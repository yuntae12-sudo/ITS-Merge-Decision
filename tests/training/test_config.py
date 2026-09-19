"""P1 tests: PPO/reward config loading (docs/ppo/PPO_PLAN.md SS0.1/P1).

Import checks + config-load checks only -- P1 explicitly has no
training loop or reward implementation yet (that's P2+).
"""

import jax
import numpy as np

from src.training.checkpoint import CheckpointPayload, get_git_sha
from src.training.config import (
    PPOConfig,
    RewardConfig,
    load_ppo_config,
    load_reward_config,
)
from src.training.seeding import make_seed_state, split_key


def test_load_ppo_base_config():
    config = load_ppo_config("configs/ppo/ppo_base.yaml")
    assert isinstance(config, PPOConfig)
    assert config.network.observation_dim == 14
    assert config.network.num_actions == 4
    assert config.network.hidden_sizes == [256, 64, 32]
    assert config.hyperparameters.learning_rate == 3.0e-4
    assert config.hyperparameters.gamma == 0.99
    assert config.hyperparameters.gae_lambda == 0.95
    assert config.hyperparameters.clip_epsilon == 0.2
    assert config.hyperparameters.value_coef == 0.5
    assert config.hyperparameters.entropy_coef == 0.01
    assert config.rollout.downstream_mode == "frenet_mpc"
    assert config.smoke is None


def test_load_ppo_smoke_config():
    config = load_ppo_config("configs/ppo/ppo_smoke.yaml")
    assert isinstance(config, PPOConfig)
    assert config.smoke is not None
    assert config.smoke.max_maneuvers >= 1
    assert config.smoke.num_updates >= 1
    assert config.rollout.downstream_mode == "frenet_mpc"


def test_load_reward_v0_config():
    config = load_reward_config("configs/reward/merge_reward_v0.yaml")
    assert isinstance(config, RewardConfig)
    assert config.reward_version == "v0"
    assert config.terminal.success == 1.0
    assert config.terminal.failure_collision == -1.0
    assert config.terminal.failure_offroad == -1.0
    assert config.terminal.truncation_horizon == -0.5
    assert config.terminal.none == 0.0
    assert config.decision_cost.real_decision_step == -0.01
    assert config.decision_cost.auto_execution_step == 0.0


def test_ppo_config_reward_path_resolves():
    ppo_config = load_ppo_config("configs/ppo/ppo_base.yaml")
    reward_config = load_reward_config(ppo_config.reward_config_path)
    assert reward_config.reward_version == "v0"


def test_config_values_are_finite():
    config = load_ppo_config("configs/ppo/ppo_base.yaml")
    h = config.hyperparameters
    for value in (
        h.learning_rate,
        h.gamma,
        h.gae_lambda,
        h.clip_epsilon,
        h.value_coef,
        h.entropy_coef,
        h.max_grad_norm,
    ):
        assert np.isfinite(value)


def test_seed_state_deterministic():
    state_a = make_seed_state(42)
    state_b = make_seed_state(42)
    assert np.array_equal(state_a.jax_key, state_b.jax_key)
    assert state_a.numpy_rng.uniform() == state_b.numpy_rng.uniform()


def test_seed_state_different_seeds_differ():
    state_a = make_seed_state(1)
    state_b = make_seed_state(2)
    assert not np.array_equal(state_a.jax_key, state_b.jax_key)


def test_split_key_returns_requested_count():
    key = jax.random.PRNGKey(0)
    keys = split_key(key, num=3)
    assert keys.shape[0] == 3


def test_git_sha_is_nonempty_string():
    sha = get_git_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0


def test_checkpoint_save_load_roundtrip(tmp_path):
    from src.training.checkpoint import load_checkpoint, save_checkpoint

    payload = CheckpointPayload(
        policy_params={"w": [1.0, 2.0]},
        value_params={"w": [3.0]},
        optimizer_state={"step": 5},
        jax_rng_key=[0, 0],
        global_env_step=100,
        ppo_update_step=3,
        seed=7,
        config_snapshot={"lr": 3e-4},
        reward_version="v0",
        git_sha="deadbeef",
    )
    path = tmp_path / "ckpt.pkl"
    save_checkpoint(payload, str(path))
    loaded = load_checkpoint(str(path))

    assert loaded.global_env_step == 100
    assert loaded.ppo_update_step == 3
    assert loaded.seed == 7
    assert loaded.reward_version == "v0"
    assert loaded.git_sha == "deadbeef"
    assert loaded.policy_params == {"w": [1.0, 2.0]}
