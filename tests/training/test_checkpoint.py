"""P5 tests for src.training.checkpoint against REAL PPO train state
(docs/ppo/PPO_PLAN.md SS0.1/P5, SS10).

P1's own checkpoint test (``tests/training/test_config.py::
test_checkpoint_save_load_roundtrip``) only exercised plain-Python
field round-tripping (dicts of floats/ints), per that test's own
docstring note. This file verifies the full SS10 contract against
REAL JAX pytrees (Flax params, optax opt_state) and the full
save -> load -> resume -> additional-update cycle SS10 requires.
"""

import dataclasses

import jax
import jax.tree_util as jtu
import numpy as np
import pytest

from src.policies.ppo.state import create_train_state
from src.training.checkpoint import (
    CheckpointPayload,
    get_git_sha,
    load_checkpoint,
    save_checkpoint,
)
from src.training.config import load_ppo_config, load_reward_config
from src.training.seeding import make_seed_state


def _leaves(pytree):
    return [np.array(x) for x in jtu.tree_leaves(pytree)]


def _make_training_state(ppo_config, seed=0):
    rng_key = jax.random.PRNGKey(seed)
    return create_train_state(
        rng_key,
        learning_rate=ppo_config.hyperparameters.learning_rate,
        max_grad_norm=ppo_config.hyperparameters.max_grad_norm,
        policy_hidden_sizes=ppo_config.network.hidden_sizes,
        value_hidden_sizes=ppo_config.network.hidden_sizes,
    )


@pytest.fixture(scope="module")
def ppo_config():
    return load_ppo_config("configs/ppo/ppo_smoke.yaml")


@pytest.fixture(scope="module")
def reward_config(ppo_config):
    return load_reward_config(ppo_config.reward_config_path)


def test_checkpoint_roundtrips_real_jax_pytrees(tmp_path, ppo_config):
    """The full SS10 contract fields, populated from a REAL
    PPOTrainingState (Flax params + optax opt_state pytrees), must
    round-trip byte-identically through save_checkpoint/load_checkpoint
    -- not just plain Python values."""

    training_state = _make_training_state(ppo_config)
    rng_key = jax.random.PRNGKey(42)

    payload = CheckpointPayload(
        policy_params=training_state.policy_state.params,
        value_params=training_state.value_state.params,
        optimizer_state={
            "policy": training_state.policy_state.opt_state,
            "value": training_state.value_state.opt_state,
        },
        jax_rng_key=rng_key,
        global_env_step=123,
        ppo_update_step=4,
        seed=0,
        config_snapshot={
            "hyperparameters": dataclasses.asdict(ppo_config.hyperparameters),
            "network_hidden_sizes": list(ppo_config.network.hidden_sizes),
        },
        reward_version="v0",
        git_sha=get_git_sha(),
    )

    path = tmp_path / "real_ckpt.pkl"
    save_checkpoint(payload, str(path))
    assert path.exists() and path.stat().st_size > 0

    loaded = load_checkpoint(str(path))

    # Policy/value params: every leaf array must match exactly, and
    # the ARRAY TYPE must survive (not silently degrade to a plain
    # list/tuple losing shape/dtype information).
    for before, after in zip(
        jtu.tree_leaves(payload.policy_params), jtu.tree_leaves(loaded.policy_params)
    ):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    for before, after in zip(
        jtu.tree_leaves(payload.value_params), jtu.tree_leaves(loaded.value_params)
    ):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))

    # Optimizer state (optax pytree, includes nested NamedTuples like
    # ScaleByAdamState) must also round-trip exactly.
    for before, after in zip(
        jtu.tree_leaves(payload.optimizer_state), jtu.tree_leaves(loaded.optimizer_state)
    ):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))

    # JAX PRNG key round-trips exactly (critical for SS10 resume).
    np.testing.assert_array_equal(np.asarray(payload.jax_rng_key), np.asarray(loaded.jax_rng_key))

    assert loaded.global_env_step == 123
    assert loaded.ppo_update_step == 4
    assert loaded.reward_version == "v0"
    assert loaded.git_sha == payload.git_sha
    assert loaded.config_snapshot == payload.config_snapshot


def test_checkpoint_loaded_params_are_usable_for_a_real_forward_pass(tmp_path, ppo_config):
    """Round-tripped params must not just LOOK equal -- they must be
    usable directly by the real Flax network's .apply() (confirms no
    hidden structural corruption, e.g. FrozenDict vs plain dict
    mismatches that happen to compare equal leaf-by-leaf but break
    ``.apply``)."""

    training_state = _make_training_state(ppo_config)
    payload = CheckpointPayload(
        policy_params=training_state.policy_state.params,
        value_params=training_state.value_state.params,
        optimizer_state={
            "policy": training_state.policy_state.opt_state,
            "value": training_state.value_state.opt_state,
        },
        jax_rng_key=jax.random.PRNGKey(0),
        global_env_step=0,
        ppo_update_step=0,
        seed=0,
        config_snapshot={},
        reward_version="v0",
        git_sha=get_git_sha(),
    )
    path = tmp_path / "ckpt.pkl"
    save_checkpoint(payload, str(path))
    loaded = load_checkpoint(str(path))

    dummy_obs = np.zeros((14,), dtype=np.float32)
    logits_before = training_state.policy_network.apply(training_state.policy_state.params, dummy_obs)
    logits_after = training_state.policy_network.apply(loaded.policy_params, dummy_obs)
    np.testing.assert_allclose(np.asarray(logits_before), np.asarray(logits_after))

    value_before = training_state.value_network.apply(training_state.value_state.params, dummy_obs)
    value_after = training_state.value_network.apply(loaded.value_params, dummy_obs)
    np.testing.assert_allclose(np.asarray(value_before), np.asarray(value_after))


def test_full_save_load_resume_additional_update_cycle(tmp_path, ppo_config, reward_config):
    """SS10's strongest requirement: save -> load -> resume ->
    additional PPO update must actually work, end to end, against a
    REAL environment rollout -- not merely a structural round-trip.
    Confirms the resumed run performs a genuinely ADDITIONAL update
    (step counters continue, params change again) rather than
    restarting from scratch.
    """

    from src.environment.merge_environment import MergeEnvironment
    from tests.training.test_rollout import SINGLE_MANEUVER
    from src.training.trainer import run_training

    env = MergeEnvironment(
        dataset_config_path="outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml",
        downstream_mode="frenet_mpc",
    )

    seed_state = make_seed_state(ppo_config.seed)
    init_key, run_key = jax.random.split(seed_state.jax_key)
    training_state = _make_training_state(ppo_config)

    # --- First training call + checkpoint save.
    result_1 = run_training(
        ppo_config=ppo_config,
        reward_config=reward_config,
        env=env,
        maneuvers=[SINGLE_MANEUVER],
        training_state=training_state,
        rng_key=run_key,
        numpy_rng=seed_state.numpy_rng,
        num_updates=1,
        max_steps_per_episode=20,
    )

    payload = CheckpointPayload(
        policy_params=result_1["training_state"].policy_state.params,
        value_params=result_1["training_state"].value_state.params,
        optimizer_state={
            "policy": result_1["training_state"].policy_state.opt_state,
            "value": result_1["training_state"].value_state.opt_state,
        },
        jax_rng_key=result_1["rng_key"],
        global_env_step=result_1["global_env_step"],
        ppo_update_step=result_1["ppo_update_step"],
        seed=ppo_config.seed,
        config_snapshot={"hyperparameters": dataclasses.asdict(ppo_config.hyperparameters)},
        reward_version=reward_config.reward_version,
        git_sha=get_git_sha(),
    )
    path = tmp_path / "resume_ckpt.pkl"
    save_checkpoint(payload, str(path))

    # --- Simulate a brand-new process: load the checkpoint fresh.
    loaded = load_checkpoint(str(path))
    resumed_training_state = _make_training_state(ppo_config)  # fresh init (shape only)
    resumed_training_state = dataclasses.replace(
        resumed_training_state,
        policy_state=resumed_training_state.policy_state.replace(
            params=loaded.policy_params, opt_state=loaded.optimizer_state["policy"]
        ),
        value_state=resumed_training_state.value_state.replace(
            params=loaded.value_params, opt_state=loaded.optimizer_state["value"]
        ),
    )

    policy_before_resume = _leaves(resumed_training_state.policy_state.params)
    value_before_resume = _leaves(resumed_training_state.value_state.params)

    result_2 = run_training(
        ppo_config=ppo_config,
        reward_config=reward_config,
        env=env,
        maneuvers=[SINGLE_MANEUVER],
        training_state=resumed_training_state,
        rng_key=loaded.jax_rng_key,
        numpy_rng=seed_state.numpy_rng,
        num_updates=1,
        max_steps_per_episode=20,
        global_env_step=loaded.global_env_step,
        ppo_update_step=loaded.ppo_update_step,
    )

    policy_after_resume = _leaves(result_2["training_state"].policy_state.params)
    value_after_resume = _leaves(result_2["training_state"].value_state.params)

    # An ADDITIONAL real update happened on top of the resumed state.
    assert any(
        not np.array_equal(a, b) for a, b in zip(policy_before_resume, policy_after_resume)
    )
    assert any(
        not np.array_equal(a, b) for a, b in zip(value_before_resume, value_after_resume)
    )

    # Step counters CONTINUED from the checkpoint, never restarted.
    assert result_2["ppo_update_step"] == loaded.ppo_update_step + 1
    assert result_2["global_env_step"] > loaded.global_env_step
    assert loaded.ppo_update_step == result_1["ppo_update_step"] == 1
    assert result_2["ppo_update_step"] == 2
