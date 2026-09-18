"""P5 tests for src.training.trainer.run_update / run_training (docs/ppo/
PPO_PLAN.md SS0.1/P5): the real multi-update PPO training loop against
the real ``MergeEnvironment``.

These are the "actual parameter update happens" / "finite loss and
gradients" / "no NaN/inf" tests required by SS0.1/P5's completion
checklist -- pipeline-verification scale only (a handful of steps/
updates on 1-2 real maneuvers), never a performance claim.
"""

import numpy as np
import pytest

from src.environment.merge_environment import ManeuverSpec, MergeEnvironment
from src.policies.ppo.state import create_train_state
from src.training.config import load_ppo_config, load_reward_config
from src.training.rollout import collect_rollout
from src.training.seeding import make_seed_state
from src.training.trainer import build_training_batch, run_training, run_update

from tests.training.test_rollout import CAUSALITY_MANEUVER, SINGLE_MANEUVER

DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"


@pytest.fixture(scope="module")
def env():
    return MergeEnvironment(
        dataset_config_path=DATASET_CONFIG_PATH, downstream_mode="frenet_mpc"
    )


@pytest.fixture(scope="module")
def ppo_config():
    return load_ppo_config("configs/ppo/ppo_smoke.yaml")


@pytest.fixture(scope="module")
def reward_config(ppo_config):
    return load_reward_config(ppo_config.reward_config_path)


def _tree_leaves_as_numpy(pytree):
    import jax.tree_util as jtu

    return [np.array(x) for x in jtu.tree_leaves(pytree)]


def _params_differ(before, after):
    return any(
        not np.array_equal(a, b) for a, b in zip(before, after)
    )


# ======================================================================
# run_update: one PPO update on a real rollout batch
# ======================================================================


def test_run_update_changes_both_policy_and_value_params(env, ppo_config, reward_config):
    """SS0.1/P5 item 8/15: policy AND value parameters must be
    DIFFERENT after run_update than before -- the real gradient-descent
    signal, not merely a no-op that returns an unmodified train state."""

    import jax

    seed_state = make_seed_state(ppo_config.seed)
    init_key, rollout_key = jax.random.split(seed_state.jax_key)
    training_state = create_train_state(
        init_key,
        learning_rate=ppo_config.hyperparameters.learning_rate,
        max_grad_norm=ppo_config.hyperparameters.max_grad_norm,
        policy_hidden_sizes=ppo_config.network.hidden_sizes,
        value_hidden_sizes=ppo_config.network.hidden_sizes,
    )

    from src.policies.ppo.policy import PPOPolicy

    ppo_policy = PPOPolicy(training_state.policy_network, training_state.policy_state.params)
    transitions = collect_rollout(
        env=env,
        maneuvers=[SINGLE_MANEUVER],
        ppo_policy=ppo_policy,
        value_network=training_state.value_network,
        value_params=training_state.value_state.params,
        reward_config=reward_config,
        rng_key=rollout_key,
        max_steps_per_episode=20,
    )
    batch = build_training_batch(transitions, ppo_config)

    before_policy = _tree_leaves_as_numpy(training_state.policy_state.params)
    before_value = _tree_leaves_as_numpy(training_state.value_state.params)

    numpy_rng = seed_state.numpy_rng
    result = run_update(training_state, batch, ppo_config, numpy_rng)

    after_policy = _tree_leaves_as_numpy(result["training_state"].policy_state.params)
    after_value = _tree_leaves_as_numpy(result["training_state"].value_state.params)

    assert _params_differ(before_policy, after_policy), (
        "run_update must produce different policy parameters (a real "
        "gradient step happened)."
    )
    assert _params_differ(before_value, after_value), (
        "run_update must produce different value parameters (a real "
        "gradient step happened)."
    )

    # Finite loss/gradients everywhere (SS0.1/P5 item 9).
    for key, value in result["metrics"].items():
        assert np.isfinite(value), f"run_update metric {key!r} is non-finite: {value}"


def test_run_update_raises_on_all_masked_out_batch(ppo_config):
    """A batch with no policy_mask==1 rows has no Actor-side signal to
    update on at all -- run_update must fail loudly rather than
    silently computing statistics over an empty set."""

    import jax

    seed_state = make_seed_state(ppo_config.seed)
    training_state = create_train_state(
        seed_state.jax_key,
        learning_rate=ppo_config.hyperparameters.learning_rate,
        max_grad_norm=ppo_config.hyperparameters.max_grad_norm,
        policy_hidden_sizes=ppo_config.network.hidden_sizes,
        value_hidden_sizes=ppo_config.network.hidden_sizes,
    )
    n = 5
    batch = {
        "observation": np.zeros((n, 14), dtype=np.float32),
        "action": np.zeros((n,), dtype=np.int32),
        "reward": np.zeros((n,), dtype=np.float64),
        "next_observation": np.zeros((n, 14), dtype=np.float32),
        "terminated": np.zeros((n,), dtype=bool),
        "truncated": np.zeros((n,), dtype=bool),
        "value": np.zeros((n,), dtype=np.float64),
        "next_value": np.zeros((n,), dtype=np.float64),
        "log_prob": np.zeros((n,), dtype=np.float64),
        "policy_mask": np.zeros((n,), dtype=np.int32),  # ALL masked out
        "advantages_raw": np.zeros((n,), dtype=np.float64),
        "advantages": np.zeros((n,), dtype=np.float64),
        "returns": np.zeros((n,), dtype=np.float64),
    }

    with pytest.raises(ValueError):
        run_update(training_state, batch, ppo_config, seed_state.numpy_rng)


# ======================================================================
# run_training: the full multi-update loop against the real environment
# ======================================================================


def test_run_training_end_to_end_changes_parameters(env, ppo_config, reward_config):
    """SS0.1/P5 items 1-9, 15: a real multi-update run_training call
    against the real MergeEnvironment produces finite metrics
    throughout and leaves policy/value parameters DIFFERENT from their
    initial values."""

    import jax

    seed_state = make_seed_state(ppo_config.seed)
    init_key, run_key = jax.random.split(seed_state.jax_key)
    training_state = create_train_state(
        init_key,
        learning_rate=ppo_config.hyperparameters.learning_rate,
        max_grad_norm=ppo_config.hyperparameters.max_grad_norm,
        policy_hidden_sizes=ppo_config.network.hidden_sizes,
        value_hidden_sizes=ppo_config.network.hidden_sizes,
    )

    before_policy = _tree_leaves_as_numpy(training_state.policy_state.params)
    before_value = _tree_leaves_as_numpy(training_state.value_state.params)

    result = run_training(
        ppo_config=ppo_config,
        reward_config=reward_config,
        env=env,
        maneuvers=[SINGLE_MANEUVER],
        training_state=training_state,
        rng_key=run_key,
        numpy_rng=seed_state.numpy_rng,
        num_updates=2,
        max_steps_per_episode=20,
    )

    after_policy = _tree_leaves_as_numpy(result["training_state"].policy_state.params)
    after_value = _tree_leaves_as_numpy(result["training_state"].value_state.params)

    assert _params_differ(before_policy, after_policy)
    assert _params_differ(before_value, after_value)

    assert result["global_env_step"] > 0
    assert result["ppo_update_step"] == 2
    assert len(result["updates"]) == 2

    for update_metrics in result["updates"]:
        for key, value in update_metrics.items():
            assert np.isfinite(value), f"metric {key!r} non-finite: {value}"


def test_run_training_continues_step_counters_when_given_nonzero_start(env, ppo_config, reward_config):
    """SS10 resume semantics at the function level (without going
    through a checkpoint file): passing nonzero starting
    global_env_step/ppo_update_step must make run_training CONTINUE
    those counters, not restart them at 0."""

    import jax

    seed_state = make_seed_state(ppo_config.seed)
    init_key, run_key = jax.random.split(seed_state.jax_key)
    training_state = create_train_state(
        init_key,
        learning_rate=ppo_config.hyperparameters.learning_rate,
        max_grad_norm=ppo_config.hyperparameters.max_grad_norm,
        policy_hidden_sizes=ppo_config.network.hidden_sizes,
        value_hidden_sizes=ppo_config.network.hidden_sizes,
    )

    result = run_training(
        ppo_config=ppo_config,
        reward_config=reward_config,
        env=env,
        maneuvers=[SINGLE_MANEUVER],
        training_state=training_state,
        rng_key=run_key,
        numpy_rng=seed_state.numpy_rng,
        num_updates=1,
        max_steps_per_episode=20,
        global_env_step=100,
        ppo_update_step=7,
    )

    assert result["global_env_step"] > 100
    assert result["ppo_update_step"] == 8


def test_run_training_action_stats_only_count_policy_mask_one_rows(env, ppo_config, reward_config):
    """SS7.2/SS8: action/*_ratio metrics must be computed over
    policy_mask==1 rows only. Using CAUSALITY_MANEUVER (single-
    transition lane_chain -- MERGE commits immediately, so almost every
    physical frame after frame 0 is policy_mask==0 auto-execution),
    confirm the action ratios do NOT simply reflect "every action was
    MERGE" (which is what a full-trajectory-unfiltered computation
    would report, since MERGE dominates raw frame count under
    auto-execution)."""

    import jax

    seed_state = make_seed_state(ppo_config.seed)
    init_key, run_key = jax.random.split(seed_state.jax_key)
    training_state = create_train_state(
        init_key,
        learning_rate=ppo_config.hyperparameters.learning_rate,
        max_grad_norm=ppo_config.hyperparameters.max_grad_norm,
        policy_hidden_sizes=ppo_config.network.hidden_sizes,
        value_hidden_sizes=ppo_config.network.hidden_sizes,
    )

    result = run_training(
        ppo_config=ppo_config,
        reward_config=reward_config,
        env=env,
        maneuvers=[CAUSALITY_MANEUVER],
        training_state=training_state,
        rng_key=run_key,
        numpy_rng=seed_state.numpy_rng,
        num_updates=1,
        max_steps_per_episode=60,
    )

    metrics = result["updates"][0]
    ratios = [
        metrics["action/keep_ratio"],
        metrics["action/follow_ratio"],
        metrics["action/merge_ratio"],
        metrics["action/stop_ratio"],
    ]
    assert abs(sum(ratios) - 1.0) < 1e-6, "Action ratios must sum to 1 over policy_mask==1 rows"
    # Exactly one policy_mask==1 decision frame exists for this
    # maneuver (frame 0) -- so exactly one ratio must be 1.0 and the
    # rest 0.0, never a distribution smeared across many auto-execution
    # MERGE frames.
    assert sorted(ratios) == [0.0, 0.0, 0.0, 1.0]
