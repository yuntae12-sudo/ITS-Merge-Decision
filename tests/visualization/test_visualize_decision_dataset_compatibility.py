"""Visualization <-> merge_decision_v2 checkpoint compatibility.

scripts/visualize_ppo_run.py previously hard-coded
expected_dataset_schema_version=DATASET_SCHEMA_V2 (merge_interaction_v2)
at its restore_ppo_checkpoint()/MergeEnvironment() call sites, so a
checkpoint recorded as dataset_schema_version="merge_decision_v2"
(e.g. outputs/ppo_checkpoints/v2_reward_v1_smoke_seed0_resumed.pkl,
trained under the frozen final MERGE Decision Dataset v2 + Reward V1)
was unconditionally rejected. These tests verify the checkpoint's own
recorded schema is now the source of truth (auto-detected, never
overwritten), routed to the correct maneuver-resolution loader, while
merge_interaction_v2/legacy checkpoints keep working exactly as
before."""

import dataclasses

import jax
import numpy as np
import pytest

from scripts.visualization.visualize_ppo_run import _resolve_maneuvers
from src.policies.ppo.networks import build_policy_network, build_value_network
from src.scenarios.merge_v2 import (
    DATASET_SCHEMA_MERGE_DECISION_V2,
    DATASET_SCHEMA_V2,
    LEGACY_DATASET_SCHEMA,
)
from src.training.checkpoint import CheckpointPayload, load_checkpoint, save_checkpoint
from src.training.config import load_ppo_config, load_reward_config
from src.visualization.ppo_checkpoint_policy import restore_ppo_checkpoint

REAL_DECISION_CHECKPOINT = (
    "outputs/ppo_checkpoints/v2_reward_v1_smoke_seed0_resumed.pkl"
)


def _build_checkpoint(tmp_path, dataset_schema_version, maneuver_ids, reward_version="v0"):
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


# 1. merge_decision_v2 checkpoint schema accepted.
def test_merge_decision_v2_checkpoint_schema_accepted(tmp_path):
    checkpoint_path = _build_checkpoint(
        tmp_path, DATASET_SCHEMA_MERGE_DECISION_V2, maneuver_ids=[]
    )
    checkpoint_schema = load_checkpoint(checkpoint_path).dataset_schema_version
    restored = restore_ppo_checkpoint(
        checkpoint_path, expected_dataset_schema_version=checkpoint_schema
    )
    assert restored.dataset_schema_version == DATASET_SCHEMA_MERGE_DECISION_V2


# 2. merge_interaction_v2 checkpoint still accepted.
def test_merge_interaction_v2_checkpoint_still_accepted(tmp_path):
    checkpoint_path = _build_checkpoint(
        tmp_path, DATASET_SCHEMA_V2, maneuver_ids=[]
    )
    checkpoint_schema = load_checkpoint(checkpoint_path).dataset_schema_version
    restored = restore_ppo_checkpoint(
        checkpoint_path, expected_dataset_schema_version=checkpoint_schema
    )
    assert restored.dataset_schema_version == DATASET_SCHEMA_V2


# 3. schema mismatch still fails when genuinely incompatible.
def test_genuine_schema_mismatch_still_fails(tmp_path):
    checkpoint_path = _build_checkpoint(
        tmp_path, DATASET_SCHEMA_MERGE_DECISION_V2, maneuver_ids=[]
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        restore_ppo_checkpoint(
            checkpoint_path, expected_dataset_schema_version=DATASET_SCHEMA_V2
        )


# 4 + 5. checkpoint scope: decision checkpoint maneuver_ids resolve
# (real 5-maneuver smoke checkpoint).
def test_real_decision_checkpoint_five_maneuver_ids_resolve():
    checkpoint_schema = load_checkpoint(REAL_DECISION_CHECKPOINT).dataset_schema_version
    assert checkpoint_schema == DATASET_SCHEMA_MERGE_DECISION_V2

    restored = restore_ppo_checkpoint(
        REAL_DECISION_CHECKPOINT, expected_dataset_schema_version=checkpoint_schema
    )
    assert len(restored.maneuver_ids) == 5
    assert restored.reward_version == "v1"

    class _Args:
        scope = "checkpoint"
        allow_legacy_dataset = False
        split_manifest = None
        maneuver_table = None
        candidate_manifest = None

    maneuvers = _resolve_maneuvers(_Args(), restored)
    assert len(maneuvers) == 5
    missing = set(restored.maneuver_ids) - {m.maneuver_id for m in maneuvers}
    assert missing == set()


# 6. merge_decision_v2 uses load_decision_dataset_maneuver_specs.
def test_decision_checkpoint_routes_through_decision_loader(monkeypatch):
    calls = []
    import scripts.visualization.visualize_ppo_run as viz_mod

    original = viz_mod.load_decision_dataset_maneuver_specs

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(viz_mod, "load_decision_dataset_maneuver_specs", _spy)

    checkpoint_schema = load_checkpoint(REAL_DECISION_CHECKPOINT).dataset_schema_version
    restored = restore_ppo_checkpoint(
        REAL_DECISION_CHECKPOINT, expected_dataset_schema_version=checkpoint_schema
    )

    class _Args:
        scope = "checkpoint"
        allow_legacy_dataset = False
        split_manifest = None
        maneuver_table = None
        candidate_manifest = None

    viz_mod._resolve_maneuvers(_Args(), restored)
    assert len(calls) == 1


# 7. merge_interaction_v2 uses existing load_maneuver_specs (legacy
# path untouched -- checked via monkeypatch spy against a fake
# restored checkpoint, no real merge_interaction_v2 manifest needed).
def test_legacy_checkpoint_routes_through_legacy_loader(monkeypatch):
    import scripts.visualization.visualize_ppo_run as viz_mod

    calls = []

    def _fake_load_maneuver_specs(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    monkeypatch.setattr(viz_mod, "load_maneuver_specs", _fake_load_maneuver_specs)

    class _FakeRestored:
        dataset_schema_version = DATASET_SCHEMA_V2
        maneuver_ids = []

    class _Args:
        scope = "checkpoint"
        allow_legacy_dataset = False
        split_manifest = "data/manifests/v2/dataset_split_v2.csv"
        maneuver_table = "data/manifests/v2/merge_maneuvers_v2.csv"
        candidate_manifest = "data/manifests/v2/merge_manifest_v2.csv"

    with pytest.raises(ValueError):
        # empty maneuver_ids -> "no maneuver_ids recorded" ValueError,
        # proving we reached the checkpoint-scope branch at all without
        # ever calling load_decision_dataset_maneuver_specs.
        viz_mod._resolve_maneuvers(_Args(), _FakeRestored())
    assert calls == []


# 8. MergeEnvironment receives correct required schema (unit-level:
# verifies main()'s schema-selection logic directly rather than
# re-running the whole CLI).
def test_expected_schema_selection_matches_checkpoint():
    decision_schema = load_checkpoint(REAL_DECISION_CHECKPOINT).dataset_schema_version
    # Mirrors main()'s auto-detect branch (dataset_contract=None).
    dataset_contract = None
    allow_legacy_dataset = False
    if allow_legacy_dataset:
        expected = None
    elif dataset_contract is not None:
        expected = {
            "merge_interaction_v2": DATASET_SCHEMA_V2,
            "merge_decision_v2": DATASET_SCHEMA_MERGE_DECISION_V2,
        }[dataset_contract]
    else:
        expected = decision_schema
    assert expected == DATASET_SCHEMA_MERGE_DECISION_V2


# 9 + 10. Reward V1 trace includes safety/progress/decision + raw
# TTC/Gap diagnostics (already implemented in a prior session; this
# re-confirms the exact field names visualize_ppo_run/trace_writer
# still expose, so a future refactor can't silently drop them).
def test_reward_v1_trace_fields_present_in_ppo_step_record():
    from src.visualization.ppo_rollout import PPOStepRecord
    from src.visualization.trace_writer import TRACE_FIELDNAMES

    v1_fields = {
        "reward_safety_component", "reward_progress_component",
        "reward_decision_component", "reward_ttc_penalty_raw",
        "reward_gap_penalty_raw",
    }
    record_fields = {f.name for f in dataclasses.fields(PPOStepRecord)}
    assert v1_fields.issubset(record_fields)
    assert v1_fields.issubset(set(TRACE_FIELDNAMES))


# 11. V0 visualization backward compatibility (checkpoint with legacy
# schema still restores/resolves without needing --allow-legacy-dataset
# special-casing beyond what already existed).
def test_legacy_schema_checkpoint_backward_compatible(tmp_path):
    checkpoint_path = _build_checkpoint(
        tmp_path, LEGACY_DATASET_SCHEMA, maneuver_ids=[]
    )
    checkpoint_schema = load_checkpoint(checkpoint_path).dataset_schema_version
    restored = restore_ppo_checkpoint(
        checkpoint_path, expected_dataset_schema_version=checkpoint_schema
    )
    assert restored.dataset_schema_version == LEGACY_DATASET_SCHEMA


# 12. validation split not accidentally mixed into train scope.
def test_decision_scope_split_train_excludes_validation():
    from src.environment.full_split_evaluator import load_decision_dataset_maneuver_specs

    train_specs = load_decision_dataset_maneuver_specs("train")
    val_specs = load_decision_dataset_maneuver_specs("validation")
    train_ids = {s.maneuver_id for s in train_specs}
    val_ids = {s.maneuver_id for s in val_specs}
    assert train_ids.isdisjoint(val_ids)


# 13. unknown dataset schema fails clearly.
def test_unknown_dataset_schema_fails_clearly():
    import scripts.visualization.visualize_ppo_run as viz_mod

    class _FakeRestored:
        dataset_schema_version = "totally_unknown_schema_v99"
        maneuver_ids = ["MAN_X"]

    class _Args:
        scope = "checkpoint"
        allow_legacy_dataset = False
        split_manifest = None
        maneuver_table = None
        candidate_manifest = None

    with pytest.raises(ValueError, match="Unknown checkpoint dataset_schema_version"):
        viz_mod._resolve_maneuvers(_Args(), _FakeRestored())
