"""Visualization <-> final MERGE Dataset checkpoint compatibility.

scripts/visualize_ppo.py resolves a checkpoint's evaluation maneuvers
through the checkpoint's own recorded dataset_schema_version, which
must always be MERGE_DATASET_SCHEMA (the final canonical contract) --
any other schema is a hard, clearly-reported error, since no other
dataset/runtime contract exists in the final pipeline."""

import dataclasses

import jax
import numpy as np
import pytest

from scripts.visualize_ppo import _resolve_maneuvers
from src.policies.ppo.networks import build_policy_network, build_value_network
from src.scenarios.merge_v2 import MERGE_DATASET_SCHEMA
from src.training.checkpoint import CheckpointPayload, load_checkpoint, save_checkpoint
from src.training.config import load_ppo_config
from src.visualization.ppo_checkpoint_policy import restore_ppo_checkpoint

REAL_SMOKE_CHECKPOINT = "outputs/checkpoints/smoke_seed0.pkl"


def _build_checkpoint(tmp_path, dataset_schema_version, maneuver_ids, reward_version="v1"):
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

    payload = CheckpointPayload(
        policy_params=policy_params,
        value_params=value_params,
        optimizer_state={"policy": None, "value": None},
        jax_rng_key=rng_key,
        global_env_step=1,
        ppo_update_step=1,
        seed=ppo_config.seed,
        config_snapshot={
            "network_hidden_sizes": list(ppo_config.network.hidden_sizes),
            "maneuver_ids": maneuver_ids,
        },
        reward_version=reward_version,
        git_sha="test-sha",
        dataset_schema_version=dataset_schema_version,
    )
    checkpoint_path = str(tmp_path / "test_checkpoint.pkl")
    save_checkpoint(payload, checkpoint_path)
    return checkpoint_path


def test_final_dataset_checkpoint_schema_accepted(tmp_path):
    checkpoint_path = _build_checkpoint(tmp_path, MERGE_DATASET_SCHEMA, maneuver_ids=[])
    checkpoint_schema = load_checkpoint(checkpoint_path).dataset_schema_version
    restored = restore_ppo_checkpoint(
        checkpoint_path, expected_dataset_schema_version=checkpoint_schema
    )
    assert restored.dataset_schema_version == MERGE_DATASET_SCHEMA


def test_genuine_schema_mismatch_still_fails(tmp_path):
    checkpoint_path = _build_checkpoint(tmp_path, MERGE_DATASET_SCHEMA, maneuver_ids=[])
    with pytest.raises(ValueError, match="schema mismatch"):
        restore_ppo_checkpoint(
            checkpoint_path, expected_dataset_schema_version="some_other_schema"
        )


def test_decision_checkpoint_routes_through_decision_loader(monkeypatch, tmp_path):
    calls = []
    import scripts.visualize_ppo as viz_mod

    original = viz_mod.load_decision_dataset_maneuver_specs

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(viz_mod, "load_decision_dataset_maneuver_specs", _spy)

    checkpoint_path = _build_checkpoint(
        tmp_path, MERGE_DATASET_SCHEMA, maneuver_ids=["MAN_DOES_NOT_MATTER"]
    )
    checkpoint_schema = load_checkpoint(checkpoint_path).dataset_schema_version
    restored = restore_ppo_checkpoint(
        checkpoint_path, expected_dataset_schema_version=checkpoint_schema
    )

    class _Args:
        scope = "checkpoint"

    with pytest.raises(ValueError, match="not found in TRAIN specs"):
        # The maneuver_id is fake, so resolution fails downstream --
        # but load_decision_dataset_maneuver_specs must still have been
        # called exactly once, proving the routing itself is correct.
        viz_mod._resolve_maneuvers(_Args(), restored)
    assert len(calls) == 1


def test_reward_v1_trace_fields_present_in_ppo_step_record():
    """Reward component fields visualize_ppo.py/trace_writer expose --
    re-confirms the exact field names so a future refactor can't
    silently drop them."""

    from src.visualization.ppo_rollout import PPOStepRecord
    from src.visualization.trace_writer import TRACE_FIELDNAMES

    reward_fields = {
        "reward_safety_component", "reward_progress_component",
        "reward_decision_component", "reward_ttc_penalty_raw",
        "reward_gap_penalty_raw",
    }
    record_fields = {f.name for f in dataclasses.fields(PPOStepRecord)}
    assert reward_fields.issubset(record_fields)
    assert reward_fields.issubset(set(TRACE_FIELDNAMES))


def test_decision_scope_split_train_excludes_validation():
    from src.environment.full_split_evaluator import load_decision_dataset_maneuver_specs

    train_specs = load_decision_dataset_maneuver_specs("train")
    val_specs = load_decision_dataset_maneuver_specs("validation")
    train_ids = {s.maneuver_id for s in train_specs}
    val_ids = {s.maneuver_id for s in val_specs}
    assert train_ids.isdisjoint(val_ids)


def test_unknown_dataset_schema_fails_clearly():
    import scripts.visualize_ppo as viz_mod

    class _FakeRestored:
        dataset_schema_version = "totally_unknown_schema_v99"
        maneuver_ids = ["MAN_X"]

    class _Args:
        scope = "checkpoint"

    with pytest.raises(ValueError, match="Unknown checkpoint dataset_schema_version"):
        viz_mod._resolve_maneuvers(_Args(), _FakeRestored())


def test_checkpoint_scope_maneuver_ids_resolve_against_real_dataset(tmp_path):
    """A checkpoint whose config_snapshot names real TRAIN maneuver_ids
    resolves them all through the final decision-dataset loader."""

    from src.environment.full_split_evaluator import load_decision_dataset_maneuver_specs

    real_ids = sorted(s.maneuver_id for s in load_decision_dataset_maneuver_specs("train"))[:5]
    checkpoint_path = _build_checkpoint(tmp_path, MERGE_DATASET_SCHEMA, maneuver_ids=real_ids)
    checkpoint_schema = load_checkpoint(checkpoint_path).dataset_schema_version
    restored = restore_ppo_checkpoint(
        checkpoint_path, expected_dataset_schema_version=checkpoint_schema
    )

    class _Args:
        scope = "checkpoint"

    maneuvers = _resolve_maneuvers(_Args(), restored)
    assert len(maneuvers) == 5
    assert {m.maneuver_id for m in maneuvers} == set(real_ids)
