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
import math

import jax
import jax.tree_util as jtu
import numpy as np
import pytest

from src.policies.ppo.state import create_train_state
from src.training.checkpoint import (
    CheckpointPayload,
    get_git_sha,
    load_checkpoint,
    restore_numpy_rng,
    save_checkpoint,
)
from src.training.config import load_ppo_config, load_reward_config
from src.training.seeding import make_seed_state
from src.training.trainer import run_update


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


# ======================================================================
# Fix 5 (pre-P6 hardening): NumPy RNG checkpoint/resume.
#
# CheckpointPayload.numpy_rng_state (the tuple from
# numpy.random.RandomState.get_state()) must round-trip through
# save_checkpoint/load_checkpoint, and restoring it must reproduce the
# EXACT minibatch shuffle order (and therefore identical resulting
# params) a continuous run would have produced -- not merely restore
# byte-identical state that happens never to get exercised.
# ======================================================================


def _make_synthetic_batch(n=12, seed=0):
    """A small synthetic (non-environment) PPO training batch -- fast
    and fully deterministic, isolating the NumPy-RNG-resume property
    from any real-environment rollout variance."""

    rng = np.random.RandomState(seed)
    observation = rng.uniform(-1, 1, size=(n, 14)).astype(np.float32)
    return {
        "observation": observation,
        "action": rng.randint(0, 4, size=(n,)).astype(np.int32),
        "reward": rng.uniform(-1, 1, size=(n,)).astype(np.float64),
        "next_observation": observation.copy(),
        "terminated": np.zeros((n,), dtype=bool),
        "truncated": np.zeros((n,), dtype=bool),
        "value": rng.uniform(-1, 1, size=(n,)).astype(np.float64),
        "next_value": rng.uniform(-1, 1, size=(n,)).astype(np.float64),
        "log_prob": rng.uniform(-2, 0, size=(n,)).astype(np.float64),
        "policy_mask": np.ones((n,), dtype=np.int32),
        "advantages_raw": rng.uniform(-1, 1, size=(n,)).astype(np.float64),
        "advantages": rng.uniform(-1, 1, size=(n,)).astype(np.float64),
        "returns": rng.uniform(-1, 1, size=(n,)).astype(np.float64),
        "old_logits": rng.uniform(-1, 1, size=(n, 4)).astype(np.float64),
    }


def test_numpy_rng_state_roundtrips_through_checkpoint(tmp_path, ppo_config):
    """CheckpointPayload.numpy_rng_state must save/load byte-identically
    via save_checkpoint/load_checkpoint, and restore_numpy_rng must
    reproduce the exact same subsequent random draws as the original
    RandomState would have produced."""

    original_rng = np.random.RandomState(123)
    _ = original_rng.permutation(10)  # advance the stream past init
    saved_state = original_rng.get_state()

    payload = CheckpointPayload(
        policy_params={},
        value_params={},
        optimizer_state={},
        jax_rng_key=jax.random.PRNGKey(0),
        global_env_step=0,
        ppo_update_step=0,
        seed=123,
        config_snapshot={},
        reward_version="v0",
        git_sha=get_git_sha(),
        numpy_rng_state=saved_state,
    )
    path = tmp_path / "rng_ckpt.pkl"
    save_checkpoint(payload, str(path))
    loaded = load_checkpoint(str(path))

    restored_rng = restore_numpy_rng(loaded.numpy_rng_state, fallback_seed=123)

    # Both streams, from this point on, must draw identically.
    expected_next = original_rng.permutation(10)
    actual_next = restored_rng.permutation(10)
    np.testing.assert_array_equal(expected_next, actual_next)


def test_restore_numpy_rng_falls_back_to_seed_when_state_missing():
    """An OLDER checkpoint with numpy_rng_state=None must fall back to
    re-seeding from the run's seed, not crash."""

    rng = restore_numpy_rng(None, fallback_seed=99)
    expected = np.random.RandomState(99)
    np.testing.assert_array_equal(rng.permutation(5), expected.permutation(5))


def test_numpy_rng_checkpoint_resume_reproduces_minibatch_order_and_params(
    tmp_path, ppo_config
):
    """The strongest Fix 5 test: Run A (uninterrupted) performs
    run_update at step N then step N+1 on a deterministic synthetic
    batch. Run B performs run_update at step N, SAVES the NumPy RNG
    state via a real checkpoint round-trip, builds a FRESH
    RandomState, LOADS the checkpoint, and performs step N+1 from the
    loaded state. With identical initial training state / config /
    batch, Run A and Run B's step-N+1 minibatch ordering and resulting
    policy/value parameters must match exactly -- proving the
    checkpointed NumPy RNG state resumes stochastic training state
    correctly (not just static params/optimizer state, which the
    pre-existing test_full_save_load_resume_additional_update_cycle
    above already covers)."""

    batch = _make_synthetic_batch()

    def _fresh_training_state():
        return create_train_state(
            jax.random.PRNGKey(7),
            learning_rate=ppo_config.hyperparameters.learning_rate,
            max_grad_norm=ppo_config.hyperparameters.max_grad_norm,
            policy_hidden_sizes=ppo_config.network.hidden_sizes,
            value_hidden_sizes=ppo_config.network.hidden_sizes,
        )

    # --- Run A: one continuous NumPy RandomState across both updates.
    training_state_a = _fresh_training_state()
    rng_a = np.random.RandomState(555)
    result_a1 = run_update(training_state_a, batch, ppo_config, rng_a)
    result_a2 = run_update(result_a1["training_state"], batch, ppo_config, rng_a)

    # --- Run B: same first update, but RESUME via a real checkpoint
    # round-trip before the second update (simulating a fresh process).
    training_state_b = _fresh_training_state()
    rng_b = np.random.RandomState(555)
    result_b1 = run_update(training_state_b, batch, ppo_config, rng_b)

    checkpoint_path = tmp_path / "numpy_rng_resume.pkl"
    payload = CheckpointPayload(
        policy_params=result_b1["training_state"].policy_state.params,
        value_params=result_b1["training_state"].value_state.params,
        optimizer_state={
            "policy": result_b1["training_state"].policy_state.opt_state,
            "value": result_b1["training_state"].value_state.opt_state,
        },
        jax_rng_key=jax.random.PRNGKey(0),
        global_env_step=0,
        ppo_update_step=1,
        seed=555,
        config_snapshot={},
        reward_version="v0",
        git_sha=get_git_sha(),
        numpy_rng_state=rng_b.get_state(),
    )
    save_checkpoint(payload, str(checkpoint_path))

    # Simulate a brand-new process: fresh RandomState object, loaded
    # checkpoint, restore the NumPy RNG stream from it.
    loaded = load_checkpoint(str(checkpoint_path))
    resumed_training_state_b = dataclasses.replace(
        _fresh_training_state(),
        policy_state=_fresh_training_state().policy_state.replace(
            params=loaded.policy_params, opt_state=loaded.optimizer_state["policy"]
        ),
        value_state=_fresh_training_state().value_state.replace(
            params=loaded.value_params, opt_state=loaded.optimizer_state["value"]
        ),
    )
    resumed_rng_b = restore_numpy_rng(loaded.numpy_rng_state, fallback_seed=555)

    result_b2 = run_update(resumed_training_state_b, batch, ppo_config, resumed_rng_b)

    # Resulting parameters after the SECOND update must match exactly
    # between the uninterrupted Run A and the checkpoint-resumed Run B
    # -- proving the minibatch shuffle order (driven by the NumPy RNG)
    # was reproduced exactly, not just that params/optimizer state
    # resumed.
    for a_leaf, b_leaf in zip(
        _leaves(result_a2["training_state"].policy_state.params),
        _leaves(result_b2["training_state"].policy_state.params),
    ):
        np.testing.assert_array_equal(a_leaf, b_leaf)
    for a_leaf, b_leaf in zip(
        _leaves(result_a2["training_state"].value_state.params),
        _leaves(result_b2["training_state"].value_state.params),
    ):
        np.testing.assert_array_equal(a_leaf, b_leaf)

    # Metrics (which depend on minibatch composition) must also match.
    for key in result_a2["metrics"]:
        assert math.isclose(
            result_a2["metrics"][key], result_b2["metrics"][key], rel_tol=1e-9, abs_tol=1e-9
        ), f"metric {key!r} diverged between Run A and resumed Run B"
