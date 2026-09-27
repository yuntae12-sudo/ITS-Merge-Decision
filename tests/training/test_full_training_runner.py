"""Targeted tests for full-TRAIN sampling and periodic checkpoints."""

import dataclasses

import jax
import numpy as np

from scripts.train_ppo import (
    FULL_TRAIN_SAMPLING_STRATEGY,
    deterministic_full_train_batch_ids,
    make_periodic_checkpoint_callback,
)
from src.policies.ppo.state import create_train_state
from src.scenarios.merge_v2 import MERGE_DATASET_SCHEMA
from src.training.checkpoint import load_checkpoint
from src.training.config import load_ppo_config
from src.training.trainer import resolve_update_maneuvers


def test_full_train_sampler_deterministic_coverage_boundary_and_reshuffle():
    pool = [f"m{i}" for i in range(10)]

    batches_a = [deterministic_full_train_batch_ids(pool, 7, step, 4) for step in range(3)]
    batches_b = [deterministic_full_train_batch_ids(pool, 7, step, 4) for step in range(3)]
    assert batches_a == batches_b

    sweep_0 = deterministic_full_train_batch_ids(pool, 7, 0, 10)
    sweep_1 = deterministic_full_train_batch_ids(pool, 7, 1, 10)
    assert len(sweep_0) == 10
    assert set(sweep_0) == set(pool)
    assert len(set(sweep_0)) == 10
    assert sweep_0 != sweep_1
    assert batches_a[2] == sweep_0[8:] + sweep_1[:2]


def test_full_train_sampler_resume_uses_absolute_update_step():
    pool = [f"m{i}" for i in range(10)]
    seed = 19
    batch_size = 4
    uninterrupted = [
        deterministic_full_train_batch_ids(pool, seed, step, batch_size)
        for step in range(8)
    ]
    resume_step = 5
    resumed_next = deterministic_full_train_batch_ids(
        pool, seed, resume_step, batch_size
    )
    assert resumed_next == uninterrupted[resume_step]


def test_fixed_subset_resolution_is_unchanged_and_selector_gets_absolute_step():
    fixed = [object(), object()]
    assert resolve_update_maneuvers(fixed, None, 12) is fixed

    seen = []

    def selector(step):
        seen.append(step)
        return [fixed[1]]

    assert resolve_update_maneuvers(fixed, selector, 12) == [fixed[1]]
    assert seen == [12]


def test_periodic_checkpoint_interval_contract_and_resume_payload(tmp_path):
    config = load_ppo_config("configs/ppo/smoke.yaml")
    state = create_train_state(
        jax.random.PRNGKey(0),
        learning_rate=config.hyperparameters.learning_rate,
        max_grad_norm=config.hyperparameters.max_grad_norm,
        policy_hidden_sizes=config.network.hidden_sizes,
        value_hidden_sizes=config.network.hidden_sizes,
    )
    numpy_rng = np.random.RandomState(0)
    pool_ids = [f"m{i:04d}" for i in range(1097)]
    snapshot = {
        "maneuver_ids": pool_ids,
        "full_train": True,
        "train_pool_size": 1097,
        "maneuvers_per_update": 2,
        "sampling_strategy": FULL_TRAIN_SAMPLING_STRATEGY,
    }
    callback = make_periodic_checkpoint_callback(
        interval=2,
        checkpoint_dir=tmp_path,
        filename_prefix="full_seed0",
        seed=0,
        reward_version="v1",
        config_snapshot=snapshot,
        numpy_rng=numpy_rng,
    )

    callback(0, state, jax.random.PRNGKey(1), 11, 1)
    assert list(tmp_path.glob("*.pkl")) == []

    callback(1, state, jax.random.PRNGKey(2), 22, 2)
    path = tmp_path / "full_seed0_step000002.pkl"
    assert path.is_file()
    payload = load_checkpoint(str(path))
    assert payload.global_env_step == 22
    assert payload.ppo_update_step == 2
    assert payload.seed == 0
    assert payload.reward_version == "v1"
    assert payload.dataset_schema_version == MERGE_DATASET_SCHEMA
    assert payload.numpy_rng_state is not None
    assert payload.config_snapshot == snapshot
    assert payload.git_sha != "unknown"

    resumed_state = create_train_state(
        jax.random.PRNGKey(9),
        learning_rate=config.hyperparameters.learning_rate,
        max_grad_norm=config.hyperparameters.max_grad_norm,
        policy_hidden_sizes=config.network.hidden_sizes,
        value_hidden_sizes=config.network.hidden_sizes,
    )
    resumed_state = dataclasses.replace(
        resumed_state,
        policy_state=resumed_state.policy_state.replace(
            params=payload.policy_params,
            opt_state=payload.optimizer_state["policy"],
        ),
        value_state=resumed_state.value_state.replace(
            params=payload.value_params,
            opt_state=payload.optimizer_state["value"],
        ),
    )
    logits = resumed_state.policy_network.apply(
        resumed_state.policy_state.params, np.zeros((14,), dtype=np.float32)
    )
    assert logits.shape == (4,)


def test_periodic_checkpoint_zero_disables_callback(tmp_path):
    assert make_periodic_checkpoint_callback(
        interval=0,
        checkpoint_dir=tmp_path,
        filename_prefix="unused",
        seed=0,
        reward_version="v1",
        config_snapshot={},
        numpy_rng=np.random.RandomState(0),
    ) is None
    assert make_periodic_checkpoint_callback(
        interval=None,
        checkpoint_dir=tmp_path,
        filename_prefix="unused",
        seed=0,
        reward_version="v1",
        config_snapshot={},
        numpy_rng=np.random.RandomState(0),
    ) is None
