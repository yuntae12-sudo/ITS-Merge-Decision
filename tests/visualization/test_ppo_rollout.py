"""Tests for src.visualization.ppo_rollout / ppo_checkpoint_policy
(docs/ppo/PPO_VISUALIZATION_GUIDE.md).

Focus areas (task spec Section 13):
  A. checkpoint policy reconstruction + deterministic reproducibility
  B. MERGE commitment semantics (pre-commit=policy decision,
     post-commit=auto-execution, never double-counted)
  D. Trace/record field correctness (step order, 14D observation
     mapping, action semantics)
"""

import dataclasses

import jax
import numpy as np
import pytest

from src.environment.behavior_action import BehaviorAction
from src.environment.merge_environment import ManeuverSpec, MergeEnvironment
from src.policies.ppo import distribution
from src.policies.ppo.networks import build_policy_network, build_value_network
from src.policies.ppo.policy import PPOPolicy
from src.training.checkpoint import CheckpointPayload, load_checkpoint, save_checkpoint
from src.training.config import load_ppo_config, load_reward_config
from src.visualization.ppo_checkpoint_policy import restore_ppo_checkpoint
from src.visualization.ppo_rollout import (
    OUTCOME_COLLISION,
    OUTCOME_OFFROAD,
    OUTCOME_SUCCESS,
    OUTCOME_TIMEOUT,
    run_ppo_episode,
)

DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"

# Same real maneuver used by tests/training/test_rollout.py (duplicated
# by value, matching that file's own stated convention).
CAUSALITY_MANEUVER = ManeuverSpec(
    maneuver_id="MAN_CAUSALITY",
    source_shard="training_tfexample.tfrecord-00001-of-01000",
    record_index=214,
    lane_chain=[527, 544],
    candidate_ids=[
        "training_tfexample.tfrecord-00001-of-01000#214__t75__527_544"
    ],
    merge_start_frame=62,
)

SINGLE_MANEUVER = ManeuverSpec(
    maneuver_id="MAN_0001",
    source_shard="training_tfexample.tfrecord-00000-of-01000",
    record_index=12,
    lane_chain=[637, 628],
    candidate_ids=[
        "training_tfexample.tfrecord-00000-of-01000#12__t49__637_628"
    ],
    merge_start_frame=36,
)


@pytest.fixture(scope="module")
def env():
    return MergeEnvironment(
        dataset_config_path=DATASET_CONFIG_PATH, downstream_mode="frenet_mpc"
    )


@pytest.fixture(scope="module")
def reward_config():
    return load_reward_config()


@pytest.fixture(scope="module")
def fresh_checkpoint_payload(tmp_path_factory):
    """Builds a real (untrained but real-shaped) checkpoint payload and
    saves it via the real save_checkpoint contract, so tests exercise
    the ACTUAL on-disk checkpoint format (not a hand-built in-memory
    substitute)."""

    ppo_config = load_ppo_config()
    reward_config = load_reward_config(ppo_config.reward_config_path)
    rng_key = jax.random.PRNGKey(ppo_config.seed)
    policy_key, value_key = jax.random.split(rng_key)

    policy_network = build_policy_network(
        hidden_sizes=ppo_config.network.hidden_sizes,
        num_actions=ppo_config.network.num_actions,
    )
    value_network = build_value_network(hidden_sizes=ppo_config.network.hidden_sizes)

    dummy_obs = np.zeros((ppo_config.network.observation_dim,), dtype=np.float32)
    policy_params = policy_network.init(policy_key, dummy_obs)
    value_params = value_network.init(value_key, dummy_obs)

    payload = CheckpointPayload(
        policy_params=policy_params,
        value_params=value_params,
        optimizer_state={"policy": None, "value": None},
        jax_rng_key=rng_key,
        global_env_step=123,
        ppo_update_step=1,
        seed=ppo_config.seed,
        config_snapshot={
            "ppo_config_path": ppo_config.source_path,
            "reward_config_path": reward_config.source_path,
            "hyperparameters": dataclasses.asdict(ppo_config.hyperparameters),
            "network_hidden_sizes": list(ppo_config.network.hidden_sizes),
            "maneuver_ids": ["MAN_0001", "MAN_CAUSALITY"],
        },
        reward_version=reward_config.reward_version,
        git_sha="test-sha",
        numpy_rng_state=None,
    )

    checkpoint_path = str(tmp_path_factory.mktemp("ckpt") / "test_checkpoint.pkl")
    save_checkpoint(payload, checkpoint_path)
    return checkpoint_path


# ======================================================================
# A. Checkpoint policy reconstruction
# ======================================================================


def test_restore_ppo_checkpoint_reconstructs_network_from_config_snapshot(fresh_checkpoint_payload):
    restored = restore_ppo_checkpoint(fresh_checkpoint_payload)

    assert restored.network_hidden_sizes == [256, 64, 32]
    assert restored.maneuver_ids == ["MAN_0001", "MAN_CAUSALITY"]
    assert restored.seed == 0
    assert restored.reward_version == "v0"
    assert restored.global_env_step == 123
    assert restored.ppo_update_step == 1
    assert isinstance(restored.policy, PPOPolicy)


def test_deterministic_action_is_reproducible(fresh_checkpoint_payload):
    restored = restore_ppo_checkpoint(fresh_checkpoint_payload)
    observation = np.zeros((14,), dtype=np.float32)

    action_1 = int(np.asarray(restored.policy.act_deterministic(observation)))
    action_2 = int(np.asarray(restored.policy.act_deterministic(observation)))

    assert action_1 == action_2
    assert action_1 in (0, 1, 2, 3)


def test_restore_falls_back_to_baseline_hidden_sizes_if_missing(fresh_checkpoint_payload):
    """An older checkpoint's config_snapshot might lack
    network_hidden_sizes -- restoration must fall back to this repo's
    fixed P0-P6 baseline shape, never crash."""

    payload = load_checkpoint(fresh_checkpoint_payload)
    stripped_snapshot = dict(payload.config_snapshot)
    del stripped_snapshot["network_hidden_sizes"]
    stripped_payload = dataclasses.replace(payload, config_snapshot=stripped_snapshot)

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "stripped.pkl")
        save_checkpoint(stripped_payload, path)
        restored = restore_ppo_checkpoint(path)

    assert restored.network_hidden_sizes == [256, 64, 32]


# ======================================================================
# B. MERGE commitment semantics
# ======================================================================


def test_prestep_commitment_only_counts_real_decisions(env, reward_config, fresh_checkpoint_payload):
    """CAUSALITY_MANEUVER's lane_chain is a single transition, so MERGE
    at decision frame 0 commits immediately and stays committed for
    every subsequent step -- exactly like
    tests/training/test_rollout.py::test_prestep_merge_policy_mask_regression's
    scripted-MERGE setup. policy_decision_count must be exactly 1 (the
    one real pre-commitment decision), never re-incremented by any
    post-commitment auto-executed MERGE step."""

    restored = restore_ppo_checkpoint(fresh_checkpoint_payload)

    class _ScriptedMergePolicy:
        def __init__(self, policy_network, policy_params):
            self._policy_network = policy_network
            self._policy_params = policy_params

        def logits(self, observation):
            return self._policy_network.apply(self._policy_params, observation)

        def act_deterministic(self, observation):
            return np.asarray(2)  # BehaviorAction.MERGE index

        def act(self, observation, rng_key):
            del rng_key
            return np.asarray(2), distribution.log_prob(self.logits(observation), 2)

    scripted_policy = _ScriptedMergePolicy(
        restored.policy._policy_network, restored.policy._policy_params
    )

    episode = run_ppo_episode(
        env=env,
        maneuver=CAUSALITY_MANEUVER,
        policy=scripted_policy,
        value_network=restored.value_network,
        value_params=restored.value_params,
        reward_config=reward_config,
        max_steps=60,
        policy_mode="deterministic",
    )

    assert episode.policy_decision_count == 1
    assert episode.merge_commit_step == 0
    assert episode.steps[0].is_policy_step is True
    assert episode.steps[0].merge_committed_before is False
    for step in episode.steps[1:]:
        assert step.is_policy_step is False
        assert step.merge_committed_before is True
        assert step.selected_action == "MERGE"
        # Post-commitment frames are diagnostic-only: probabilities
        # must be NaN, never a real decision value.
        assert np.isnan(step.keep_prob)
        assert np.isnan(step.follow_prob)
        assert np.isnan(step.merge_prob)
        assert np.isnan(step.stop_prob)

    assert episode.outcome in (OUTCOME_SUCCESS, OUTCOME_COLLISION, OUTCOME_OFFROAD, OUTCOME_TIMEOUT)


def test_commitment_uses_prestep_info_not_poststep(env, reward_config, fresh_checkpoint_payload):
    """Regression guard mirroring
    tests/training/test_rollout.py::test_prestep_merge_policy_mask_regression:
    the step ON WHICH MERGE is first selected must itself be recorded
    as a real policy decision (merge_committed_before=False), with
    commitment only taking effect from the NEXT step onward -- never
    the reversed post-step bug where the commit-triggering step itself
    gets marked as already-committed."""

    restored = restore_ppo_checkpoint(fresh_checkpoint_payload)

    episode = run_ppo_episode(
        env=env,
        maneuver=SINGLE_MANEUVER,
        policy=restored.policy,
        value_network=restored.value_network,
        value_params=restored.value_params,
        reward_config=reward_config,
        max_steps=30,
        policy_mode="deterministic",
    )

    if episode.merge_commit_step is not None:
        commit_record = next(s for s in episode.steps if s.step_index == episode.merge_commit_step)
        assert commit_record.merge_committed_before is False
        assert commit_record.is_policy_step is True


# ======================================================================
# D. Trace/record correctness
# ======================================================================


def test_step_order_and_observation_mapping(env, reward_config, fresh_checkpoint_payload):
    restored = restore_ppo_checkpoint(fresh_checkpoint_payload)

    episode = run_ppo_episode(
        env=env,
        maneuver=SINGLE_MANEUVER,
        policy=restored.policy,
        value_network=restored.value_network,
        value_params=restored.value_params,
        reward_config=reward_config,
        max_steps=30,
        policy_mode="deterministic",
    )

    assert len(episode.steps) > 0
    step_indices = [s.step_index for s in episode.steps]
    assert step_indices == list(range(len(episode.steps)))

    for record in episode.steps:
        assert record.observation.shape == (14,)
        assert record.action_index in (0, 1, 2, 3)
        assert BehaviorAction[record.selected_action].value == record.action_index

    # Only the last record may be terminal.
    for record in episode.steps[:-1]:
        assert not record.terminated
        assert not record.truncated


def test_outcome_matches_environment_termination_reason(env, reward_config, fresh_checkpoint_payload):
    """Outcome must be derived purely from MergeEnvironment's own
    terminated/truncated/termination_reason -- never re-judged by a
    separate safety rule."""

    restored = restore_ppo_checkpoint(fresh_checkpoint_payload)

    episode = run_ppo_episode(
        env=env,
        maneuver=SINGLE_MANEUVER,
        policy=restored.policy,
        value_network=restored.value_network,
        value_params=restored.value_params,
        reward_config=reward_config,
        max_steps=30,
        policy_mode="deterministic",
    )

    last = episode.steps[-1]
    if last.terminated:
        expected = {
            "success": OUTCOME_SUCCESS,
            "failure_collision": OUTCOME_COLLISION,
            "failure_offroad": OUTCOME_OFFROAD,
        }[last.termination_reason]
        assert episode.outcome == expected
    elif last.truncated:
        assert episode.outcome == OUTCOME_TIMEOUT
    else:
        # Loop exhausted max_steps with no real terminal/truncated event.
        assert episode.outcome == OUTCOME_TIMEOUT
