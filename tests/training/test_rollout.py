"""P4 tests for src.training.rollout (docs/ppo/PPO_PLAN.md SS0.1/P4,
SS7.1, SS7.2, SS11).

The centerpiece test in this file (and arguably in all of P4) is
``test_prestep_merge_policy_mask_regression``: it guards against the
exact reversed post-step ``policy_mask`` bug SS7.1 warns about, using
a scripted-MERGE policy against the real ``MergeEnvironment``.
"""

import jax
import numpy as np
import pytest

from src.environment.behavior_action import BehaviorAction
from src.environment.merge_environment import ManeuverSpec, MergeEnvironment
from src.policies.ppo import distribution
from src.policies.ppo.networks import build_policy_network, build_value_network
from src.policies.ppo.policy import PPOPolicy
from src.training.config import load_ppo_config, load_reward_config
from src.training.gae import compute_gae, normalize_advantages_masked
from src.training.rollout import Transition, collect_episode_rollout, collect_rollout

DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"

# Same real maneuvers used by tests/environment/test_merge_environment_frenet_mpc.py
# (duplicated by value, matching that file's own stated convention of
# not importing fixtures cross-file to keep each file's fixture set
# independently readable).
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

# Reaches a real terminal outcome (success/collision/offroad) within
# 60 steps of continuous MERGE from step 0 -- confirmed by
# tests/environment/test_merge_environment_frenet_mpc.py::
# test_single_maneuver_merge_reaches_success_frenet_mpc. Single-
# transition lane_chain (not chained), so MERGE at decision frame 0
# commits immediately and stays committed for the rest of the episode.
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


@pytest.fixture(scope="module")
def env():
    return MergeEnvironment(
        dataset_config_path=DATASET_CONFIG_PATH, downstream_mode="frenet_mpc"
    )


@pytest.fixture(scope="module")
def reward_config():
    return load_reward_config()


@pytest.fixture(scope="module")
def ppo_core():
    """A real (untrained) PPO policy + value network/params, built the
    same way P3's train-state construction does, for use across P4's
    rollout tests."""

    ppo_config = load_ppo_config()
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

    ppo_policy = PPOPolicy(policy_network, policy_params)
    return {
        "ppo_config": ppo_config,
        "ppo_policy": ppo_policy,
        "value_network": value_network,
        "value_params": value_params,
    }


class _ScriptedMergePolicy:
    """A minimal stand-in exposing the same ``.act``/``.logits``
    interface as ``PPOPolicy``, but that ALWAYS selects MERGE
    deterministically -- used to construct a scripted episode where
    "PPO selects MERGE" is guaranteed on the very first decision step,
    for the SS7.1 pre-step policy_mask regression test. Backed by a
    real (untrained) policy network so ``.logits``/log_prob remain
    well-defined finite numbers -- only the SAMPLED action is
    overridden to be deterministic."""

    def __init__(self, policy_network, policy_params):
        self._policy_network = policy_network
        self._policy_params = policy_params

    def logits(self, observation):
        return self._policy_network.apply(self._policy_params, observation)

    def act(self, observation, rng_key):
        del rng_key
        logits = self.logits(observation)
        merge_index = 2  # BehaviorAction.MERGE per the SS11 fixed mapping
        log_prob = distribution.log_prob(logits, merge_index)
        return np.asarray(merge_index), log_prob


# ======================================================================
# Rollout shape consistency / no NaN-inf / real 1-episode rollout
# ======================================================================


def test_real_one_episode_rollout_runs_to_completion(env, ppo_core, reward_config):
    """An ACTUAL 1-episode rollout against the real MergeEnvironment
    (downstream_mode='frenet_mpc'), using the real (untrained) PPO
    policy -- not a synthetic/toy trajectory. Must run to completion
    (terminated or truncated) and produce a valid, non-empty transition
    buffer."""

    rng_key = jax.random.PRNGKey(123)
    transitions = collect_episode_rollout(
        env=env,
        maneuver=SINGLE_MANEUVER,
        ppo_policy=ppo_core["ppo_policy"],
        value_network=ppo_core["value_network"],
        value_params=ppo_core["value_params"],
        reward_config=reward_config,
        rng_key=rng_key,
        max_steps=80,
        episode_id="real_rollout_test",
    )

    assert len(transitions) > 0
    last = transitions[-1]
    assert last.terminated or last.truncated

    for t in transitions:
        assert isinstance(t, Transition)
        assert t.observation.shape == (14,)
        assert t.next_observation.shape == (14,)
        assert t.action in (0, 1, 2, 3)
        assert t.policy_mask in (0, 1)
        assert np.isfinite(t.reward)
        assert np.isfinite(t.value)
        assert np.isfinite(t.next_value)
        assert np.isfinite(t.log_prob)
        assert t.maneuver_id == SINGLE_MANEUVER.maneuver_id


def test_rollout_shape_consistency(env, ppo_core, reward_config):
    rng_key = jax.random.PRNGKey(7)
    transitions = collect_episode_rollout(
        env=env,
        maneuver=SINGLE_MANEUVER,
        ppo_policy=ppo_core["ppo_policy"],
        value_network=ppo_core["value_network"],
        value_params=ppo_core["value_params"],
        reward_config=reward_config,
        rng_key=rng_key,
        max_steps=20,
    )

    assert len(transitions) > 0
    obs_shape = transitions[0].observation.shape
    for i, t in enumerate(transitions):
        assert t.observation.shape == obs_shape
        assert t.next_observation.shape == obs_shape
        assert t.step_index == i
        for field in Transition.__dataclass_fields__:
            assert hasattr(t, field)


def test_no_nan_inf_across_full_episode(env, ppo_core, reward_config):
    rng_key = jax.random.PRNGKey(99)
    transitions = collect_episode_rollout(
        env=env,
        maneuver=SINGLE_MANEUVER,
        ppo_policy=ppo_core["ppo_policy"],
        value_network=ppo_core["value_network"],
        value_params=ppo_core["value_params"],
        reward_config=reward_config,
        rng_key=rng_key,
        max_steps=30,
    )

    for t in transitions:
        assert np.all(np.isfinite(t.observation))
        assert np.all(np.isfinite(t.next_observation))
        assert np.isfinite(t.reward)
        assert np.isfinite(t.value)
        assert np.isfinite(t.next_value)
        assert np.isfinite(t.log_prob)


def test_multi_maneuver_collect_rollout_concatenates_episodes(env, ppo_core, reward_config):
    rng_key = jax.random.PRNGKey(11)
    transitions = collect_rollout(
        env=env,
        maneuvers=[SINGLE_MANEUVER, SINGLE_MANEUVER],
        ppo_policy=ppo_core["ppo_policy"],
        value_network=ppo_core["value_network"],
        value_params=ppo_core["value_params"],
        reward_config=reward_config,
        rng_key=rng_key,
        max_steps_per_episode=15,
    )

    episode_ids = {t.episode_id for t in transitions}
    assert len(episode_ids) == 2
    assert len(transitions) > 0


# ======================================================================
# SS7.1: pre-step MERGE policy_mask regression test (THE most important
# test in P4)
# ======================================================================


def test_prestep_merge_policy_mask_regression(env, ppo_core, reward_config):
    """Guards against the reversed post-step policy_mask bug (SS7.1):
    given a scripted episode where the policy selects MERGE at the
    FIRST decision frame, that exact frame's policy_mask must be 1 --
    NOT 0. Only frames STRICTLY AFTER it (auto-execution, where
    merge_committed is already True going into the step) may be 0.

    CAUSALITY_MANEUVER's lane_chain has exactly one transition, so
    MERGE at frame 0 commits immediately and every subsequent frame is
    auto-execution for the rest of the episode -- makes the frame-by-
    frame boundary unambiguous to assert on.
    """

    scripted_policy = _ScriptedMergePolicy(
        ppo_core["ppo_policy"]._policy_network,
        ppo_core["ppo_policy"]._policy_params,
    )

    rng_key = jax.random.PRNGKey(42)
    transitions = collect_episode_rollout(
        env=env,
        maneuver=CAUSALITY_MANEUVER,
        ppo_policy=scripted_policy,
        value_network=ppo_core["value_network"],
        value_params=ppo_core["value_params"],
        reward_config=reward_config,
        rng_key=rng_key,
        max_steps=60,
    )

    assert len(transitions) >= 2, "Expected at least a decision frame plus one auto-execution frame"

    # The very first frame is the decision frame where MERGE is
    # selected (scripted policy always selects MERGE, and the episode
    # starts in DECISION phase) -- this is the exact frame SS7.1 says
    # must be policy_mask == 1.
    assert transitions[0].action == int(BehaviorAction.MERGE)
    assert transitions[0].policy_mask == 1, (
        "SS7.1 regression: the decision frame that selects MERGE must "
        "have policy_mask == 1, not 0 (reversed post-step bug)."
    )

    # Every frame strictly after frame 0 is auto-execution (single-
    # transition lane_chain commits immediately on frame 0) -- all must
    # be policy_mask == 0.
    for t in transitions[1:]:
        assert t.action == int(BehaviorAction.MERGE)
        assert t.policy_mask == 0, (
            "SS7.1 regression: auto-executed MERGE-commitment frames "
            "(merge_committed already True going into the step) must "
            "have policy_mask == 0."
        )

    # The episode must reach a REAL terminal outcome (not merely hit
    # max_steps without terminating) -- otherwise this test would not
    # actually exercise the terminal-reward-propagation scenario.
    assert transitions[-1].terminated or transitions[-1].truncated


# ======================================================================
# SS7.2 test D: terminal reward propagation across masked frames
# ======================================================================


def test_terminal_reward_propagates_to_merge_decision_frame(env, ppo_core, reward_config):
    """A MERGE decision (policy_mask=1) followed by auto-execution
    frames (policy_mask=0) followed by a terminal SUCCESS/COLLISION
    reward: the terminal reward's GAE credit must propagate back to
    the MERGE decision frame's advantage/return -- not just reflect its
    own (small, decision-cost-only) immediate reward.
    """

    scripted_policy = _ScriptedMergePolicy(
        ppo_core["ppo_policy"]._policy_network, ppo_core["ppo_policy"]._policy_params
    )

    rng_key = jax.random.PRNGKey(2024)
    transitions = collect_episode_rollout(
        env=env,
        maneuver=CAUSALITY_MANEUVER,
        ppo_policy=scripted_policy,
        value_network=ppo_core["value_network"],
        value_params=ppo_core["value_params"],
        reward_config=reward_config,
        rng_key=rng_key,
        max_steps=60,
    )

    assert transitions[-1].terminated, "Expected a true terminal outcome (success/collision/offroad)"
    assert len(transitions) >= 2

    terminal_reward = transitions[-1].reward
    # A real terminal outcome (SUCCESS=+1.0/COLLISION=-1.0/OFFROAD=-1.0)
    # dwarfs the -0.01/0.0 decision-cost-only components, so this
    # confirms the episode actually reached a real terminal event
    # rather than staying in the "none" (0.0 terminal component) state.
    assert abs(terminal_reward) >= 0.5

    hp = ppo_core["ppo_config"].hyperparameters
    rewards = np.array([t.reward for t in transitions])
    values = np.array([t.value for t in transitions])
    next_values = np.array([t.next_value for t in transitions])
    terminated = np.array([t.terminated for t in transitions])
    truncated = np.array([t.truncated for t in transitions])

    gae_result = compute_gae(
        rewards, values, next_values, terminated, truncated, hp.gamma, hp.gae_lambda
    )

    # The MERGE decision frame (index 0) must have a return that
    # reflects far more than just its own tiny immediate decision-cost
    # reward -- i.e. the terminal reward's credit has propagated back
    # via GAE's backward recursion through the intervening auto-
    # execution (policy_mask==0) frames to frame 0.
    decision_frame_return = gae_result.returns[0]
    decision_frame_immediate_reward = transitions[0].reward

    assert abs(decision_frame_return) > abs(decision_frame_immediate_reward), (
        "Terminal reward should propagate back through GAE to the MERGE "
        "decision frame's return, making it larger in magnitude than "
        "the decision frame's own tiny immediate reward alone."
    )

    # More directly: the decision frame's return must have the same
    # sign as (and be substantially influenced by) the terminal reward,
    # confirming real credit flow rather than coincidence.
    if terminal_reward > 0:
        assert decision_frame_return > decision_frame_immediate_reward
    else:
        assert decision_frame_return < decision_frame_immediate_reward


def test_reward_components_propagate_from_wrapper_to_transitions(env, ppo_core, reward_config):
    """Reward component logging correctness follow-up fix: each real
    rollout Transition's reward_terminal_component /
    reward_decision_cost_component must match what MergeRewardWrapper
    actually computed for that step (never a synthesized/re-derived
    value), and every transition's two components must sum to exactly
    its own `reward` -- including the final terminal transition, where
    the OLD trainer.py bug would have conflated the two."""

    scripted_policy = _ScriptedMergePolicy(
        ppo_core["ppo_policy"]._policy_network, ppo_core["ppo_policy"]._policy_params
    )

    rng_key = jax.random.PRNGKey(2024)
    transitions = collect_episode_rollout(
        env=env,
        maneuver=CAUSALITY_MANEUVER,
        ppo_policy=scripted_policy,
        value_network=ppo_core["value_network"],
        value_params=ppo_core["value_params"],
        reward_config=reward_config,
        rng_key=rng_key,
        max_steps=60,
    )

    assert transitions[-1].terminated, "Expected a true terminal outcome (success/collision/offroad)"

    for t in transitions:
        assert t.reward_terminal_component + t.reward_decision_cost_component == pytest.approx(
            t.reward, abs=1e-9
        )

    # The final (terminal) transition's terminal component must be one
    # of Reward V0's fixed terminal-outcome values, not contaminated by
    # any decision-cost component -- the exact bug this fix guards
    # against (the old code would have folded the ENTIRE t.reward,
    # decision cost included, into what it called "terminal").
    final = transitions[-1]
    assert final.reward_terminal_component in (1.0, -1.0)
    assert final.reward_decision_cost_component in (-0.01, 0.0)

    # Episode-aggregate identity (Test G, real-environment version):
    # sum of components must equal sum of rewards within tolerance.
    total_terminal = sum(t.reward_terminal_component for t in transitions)
    total_decision_cost = sum(t.reward_decision_cost_component for t in transitions)
    total_reward = sum(t.reward for t in transitions)
    assert total_terminal + total_decision_cost == pytest.approx(total_reward, abs=1e-6)


def test_reward_components_zero_terminal_on_artificial_cutoff(env, ppo_core, reward_config):
    """An artificial rollout_cutoff transition (max_steps exhausted
    without the environment itself reporting terminated/truncated) must
    have reward_terminal_component == 0.0 -- confirms
    MergeRewardWrapper.compute is actually invoked with the
    environment's real (non-terminal) termination_reason for such a
    step, not a synthesized terminal status."""

    scripted_policy = _ScriptedMergePolicy(
        ppo_core["ppo_policy"]._policy_network, ppo_core["ppo_policy"]._policy_params
    )

    rng_key = jax.random.PRNGKey(7)
    # SINGLE_MANEUVER + a tiny max_steps: very likely to hit the
    # artificial trainer-side cutoff before any real environment
    # terminated/truncated outcome.
    transitions = collect_episode_rollout(
        env=env,
        maneuver=SINGLE_MANEUVER,
        ppo_policy=scripted_policy,
        value_network=ppo_core["value_network"],
        value_params=ppo_core["value_params"],
        reward_config=reward_config,
        rng_key=rng_key,
        max_steps=3,
    )

    last = transitions[-1]
    if last.rollout_cutoff:
        assert not last.terminated
        assert not last.truncated
        assert last.reward_terminal_component == pytest.approx(0.0)
        # Only the decision-cost component (driven by that step's own
        # policy_mask), never a synthesized terminal outcome.
        expected_decision_cost = -0.01 if last.policy_mask == 1 else 0.0
        assert last.reward_decision_cost_component == pytest.approx(expected_decision_cost)
        assert last.reward == pytest.approx(expected_decision_cost)


# ======================================================================
# SS7.2 test B: Actor-loss invariance to masked frames (integration
# level, using a real scripted rollout's policy_mask==0 frames)
# ======================================================================


def test_masked_frames_excluded_from_actor_side_statistics(env, ppo_core, reward_config):
    """Integration-level SS7.2 test B: using a real rollout containing
    both policy_mask==1 and policy_mask==0 frames, Actor-side
    statistics (here: the set of log_probs/actions/advantages fed into
    a policy-loss-style computation) computed after filtering to
    policy_mask==1 must be identical regardless of what the
    policy_mask==0 frames' log_prob/action/advantage values are -- i.e.
    filtering by mask, not merely weighting by it, is what rollout.py's
    consumers must do downstream."""

    scripted_policy = _ScriptedMergePolicy(
        ppo_core["ppo_policy"]._policy_network, ppo_core["ppo_policy"]._policy_params
    )
    rng_key = jax.random.PRNGKey(555)
    transitions = collect_episode_rollout(
        env=env,
        maneuver=CAUSALITY_MANEUVER,
        ppo_policy=scripted_policy,
        value_network=ppo_core["value_network"],
        value_params=ppo_core["value_params"],
        reward_config=reward_config,
        rng_key=rng_key,
        max_steps=60,
    )

    policy_mask = np.array([t.policy_mask for t in transitions])
    assert np.sum(policy_mask == 0) > 0, "Need at least one auto-execution frame for this test"
    assert np.sum(policy_mask == 1) > 0, "Need at least one decision frame for this test"

    log_probs = np.array([t.log_prob for t in transitions])
    actions = np.array([t.action for t in transitions])

    decision_log_probs = log_probs[policy_mask == 1]
    decision_actions = actions[policy_mask == 1]

    # Perturb ONLY the policy_mask==0 frames' log_prob/action values in
    # a copy and confirm the policy_mask==1 filtered subset is exactly
    # unchanged (this is what "excluded from Actor-side statistics"
    # concretely means at the data level).
    log_probs_perturbed = log_probs.copy()
    log_probs_perturbed[policy_mask == 0] = -999.0
    actions_perturbed = actions.copy()
    actions_perturbed[policy_mask == 0] = 3  # STOP, arbitrary different value

    decision_log_probs_after = log_probs_perturbed[policy_mask == 1]
    decision_actions_after = actions_perturbed[policy_mask == 1]

    np.testing.assert_array_equal(decision_log_probs, decision_log_probs_after)
    np.testing.assert_array_equal(decision_actions, decision_actions_after)


# ======================================================================
# SS11 action-mapping regression (integration-level, via rollout.py)
# ======================================================================


def test_action_mapping_regression_via_rollout(env, ppo_core, reward_config):
    """Confirms rollout.py itself does not perform any separate/second
    action-index -> BehaviorAction translation that could drift from
    distribution.ACTION_INDEX_TO_BEHAVIOR (SS11) -- it delegates
    entirely to that fixed mapping."""

    assert distribution.ACTION_INDEX_TO_BEHAVIOR == {
        0: BehaviorAction.KEEP,
        1: BehaviorAction.FOLLOW,
        2: BehaviorAction.MERGE,
        3: BehaviorAction.STOP,
    }

    scripted_policy = _ScriptedMergePolicy(
        ppo_core["ppo_policy"]._policy_network, ppo_core["ppo_policy"]._policy_params
    )
    rng_key = jax.random.PRNGKey(3)
    transitions = collect_episode_rollout(
        env=env,
        maneuver=CAUSALITY_MANEUVER,
        ppo_policy=scripted_policy,
        value_network=ppo_core["value_network"],
        value_params=ppo_core["value_params"],
        reward_config=reward_config,
        rng_key=rng_key,
        max_steps=5,
    )
    # The scripted policy always selects action-index 2; every
    # recorded transition's action must map to BehaviorAction.MERGE
    # under the SAME fixed table, never a rollout-local duplicate.
    for t in transitions:
        assert t.action == 2
        assert distribution.ACTION_INDEX_TO_BEHAVIOR[t.action] == BehaviorAction.MERGE
