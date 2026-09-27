"""Frozen final MERGE Decision Dataset v2 <-> PPO runtime schema
compatibility. The frozen dataset (data/manifests/v2/merge_decision_*_
v2.csv) is a PPO decision-context candidate-level schema; runtime
ManeuverSpec expects lane_chain/candidate_ids/merge_start_frame/
topology_evidence/interaction_evidence (legacy/canonical merge table
shape). These tests verify the new
src.environment.full_split_evaluator.load_decision_dataset_maneuver_specs
loader + ManeuverSpec.from_decision_dataset_row/DATASET_SCHEMA_MERGE_
DECISION_V2 contract bridge the two WITHOUT rewriting the frozen CSVs,
and that the pre-existing legacy path is completely unaffected."""

import csv
import json

import pytest

from src.environment.full_split_evaluator import (
    DECISION_DATASET_ROLES_FOR_PPO,
    DECISION_TIERS_FOR_PPO,
    load_decision_dataset_maneuver_specs,
)
from src.environment.merge_environment import ManeuverSpec
from src.scenarios.merge_v2 import DATASET_SCHEMA_V2, MERGE_DATASET_SCHEMA


MANEUVER_FIELDS = [
    "candidate_id", "maneuver_id", "scenario_id", "scene_key",
    "source_dataset", "source_split", "source_shard", "record_index",
    "split", "decision_tier", "dataset_role", "merge_context_status",
    "decision_relevance", "num_meaningful_kfm_actions", "archetype",
]


def _maneuver_row(
    maneuver_id, split="training", decision_tier="A", dataset_role="CORE",
    merge_context_status="MERGE_CONTEXT_ELIGIBLE", record_index=0,
    source_shard="shard0",
):
    return {
        "candidate_id": maneuver_id,
        "maneuver_id": maneuver_id,
        "scenario_id": "scn0",
        "scene_key": "scene0",
        "source_dataset": "WOMD",
        "source_split": "training",
        "source_shard": source_shard,
        "record_index": record_index,
        "split": split,
        "decision_tier": decision_tier,
        "dataset_role": dataset_role,
        "merge_context_status": merge_context_status,
        "decision_relevance": "MEDIUM",
        "num_meaningful_kfm_actions": 1,
        "archetype": "FRONT_CONSTRAINED",
    }


def _evidence_row(maneuver_id, source_lane_id=100, target_lane_id=200, commit_frame=10):
    return {
        "candidate_id": maneuver_id,
        "maneuver_id": maneuver_id,
        "scenario_id": "scn0",
        "scene_key": "scene0",
        "source_dataset": "WOMD",
        "source_split": "training",
        "source_shard": "shard0",
        "record_index": 0,
        "topology_evidence": {
            "source_lane_id": source_lane_id,
            "target_lane_id": target_lane_id,
        },
        "interaction_evidence": {
            "decision_start_frame": 0,
            "commit_frame": commit_frame,
            "completion_frame": commit_frame,
            "front_vehicle_id": None,
            "rear_vehicle_id": None,
            "conflict_vehicle_ids": [],
        },
        "gap_timeseries": [],
        "automatic_decision": "accept",
        "automatic_reason": "",
        "manual_validation": "UNREVIEWED",
        "manual_note": "",
    }


def _write_dataset(tmp_path, maneuver_rows, evidence_rows, split_rows=None):
    maneuver_table = tmp_path / "maneuvers.csv"
    with open(maneuver_table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANEUVER_FIELDS)
        writer.writeheader()
        for row in maneuver_rows:
            writer.writerow(row)

    split_manifest = tmp_path / "split.csv"
    if split_rows is None:
        split_rows = [
            {"maneuver_id": r["maneuver_id"], "scene_key": r["scene_key"],
             "scenario_id": r["scenario_id"], "split": r["split"],
             "dataset_role": r["dataset_role"]}
            for r in maneuver_rows
        ]
    with open(split_manifest, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["maneuver_id", "scene_key", "scenario_id", "split", "dataset_role"]
        )
        writer.writeheader()
        for row in split_rows:
            writer.writerow(row)

    evidence_training = tmp_path / "evidence_training.jsonl"
    with open(evidence_training, "w") as f:
        for row in evidence_rows:
            f.write(json.dumps(row) + "\n")

    evidence_validation = tmp_path / "evidence_validation.jsonl"
    with open(evidence_validation, "w") as f:
        pass

    return str(maneuver_table), str(split_manifest), str(evidence_training), str(evidence_validation)


# 1. Final frozen TRAIN A+B 1097 rows load (against real repo files).
def test_final_frozen_train_ab_1097_rows_load():
    specs = load_decision_dataset_maneuver_specs("train")
    assert len(specs) == 1097


def test_final_frozen_validation_ab_388_rows_load():
    specs = load_decision_dataset_maneuver_specs("validation")
    assert len(specs) == 388


# 2. training split normalization (real repo file uses "training").
def test_training_split_label_normalizes_correctly():
    specs = load_decision_dataset_maneuver_specs("training")
    assert len(specs) == 1097


# 3. Tier A/B only.
def test_tier_c_excluded(tmp_path):
    rows = [
        _maneuver_row("MAN_A", decision_tier="A"),
        _maneuver_row("MAN_B", decision_tier="B"),
        _maneuver_row("MAN_C", decision_tier="C"),
    ]
    evidence = [_evidence_row(r["maneuver_id"]) for r in rows]
    maneuver_table, split_manifest, evidence_training, evidence_validation = _write_dataset(
        tmp_path, rows, evidence
    )
    specs = load_decision_dataset_maneuver_specs(
        "train", maneuver_table_path=maneuver_table, split_manifest_path=split_manifest,
        evidence_training_path=evidence_training, evidence_validation_path=evidence_validation,
    )
    assert {s.maneuver_id for s in specs} == {"MAN_A", "MAN_B"}
    assert set(DECISION_TIERS_FOR_PPO) == {"A", "B"}


# 4. CORE/SUPPORT only.
def test_non_core_support_role_excluded(tmp_path):
    rows = [
        _maneuver_row("MAN_CORE", dataset_role="CORE"),
        _maneuver_row("MAN_SUPPORT", dataset_role="SUPPORT"),
        _maneuver_row("MAN_OTHER", dataset_role="EXCLUDED"),
    ]
    evidence = [_evidence_row(r["maneuver_id"]) for r in rows]
    maneuver_table, split_manifest, evidence_training, evidence_validation = _write_dataset(
        tmp_path, rows, evidence
    )
    specs = load_decision_dataset_maneuver_specs(
        "train", maneuver_table_path=maneuver_table, split_manifest_path=split_manifest,
        evidence_training_path=evidence_training, evidence_validation_path=evidence_validation,
    )
    assert {s.maneuver_id for s in specs} == {"MAN_CORE", "MAN_SUPPORT"}
    assert set(DECISION_DATASET_ROLES_FOR_PPO) == {"CORE", "SUPPORT"}


# 5. INELIGIBLE/Tier C excluded.
def test_ineligible_merge_context_excluded(tmp_path):
    rows = [
        _maneuver_row("MAN_ELIGIBLE", merge_context_status="MERGE_CONTEXT_ELIGIBLE"),
        _maneuver_row("MAN_INELIGIBLE", merge_context_status="MERGE_CONTEXT_INELIGIBLE"),
    ]
    evidence = [_evidence_row(r["maneuver_id"]) for r in rows]
    maneuver_table, split_manifest, evidence_training, evidence_validation = _write_dataset(
        tmp_path, rows, evidence
    )
    specs = load_decision_dataset_maneuver_specs(
        "train", maneuver_table_path=maneuver_table, split_manifest_path=split_manifest,
        evidence_training_path=evidence_training, evidence_validation_path=evidence_validation,
    )
    assert {s.maneuver_id for s in specs} == {"MAN_ELIGIBLE"}


# 6. lane_chain/runtime topology mapping valid.
def test_lane_chain_mapped_from_topology_evidence(tmp_path):
    rows = [_maneuver_row("MAN_X")]
    evidence = [_evidence_row("MAN_X", source_lane_id=42, target_lane_id=99)]
    maneuver_table, split_manifest, evidence_training, evidence_validation = _write_dataset(
        tmp_path, rows, evidence
    )
    specs = load_decision_dataset_maneuver_specs(
        "train", maneuver_table_path=maneuver_table, split_manifest_path=split_manifest,
        evidence_training_path=evidence_training, evidence_validation_path=evidence_validation,
    )
    assert specs[0].lane_chain == [42, 99]


# 7. candidate_ids mapping valid.
def test_candidate_ids_mapped_single_candidate(tmp_path):
    rows = [_maneuver_row("MAN_X")]
    evidence = [_evidence_row("MAN_X")]
    maneuver_table, split_manifest, evidence_training, evidence_validation = _write_dataset(
        tmp_path, rows, evidence
    )
    specs = load_decision_dataset_maneuver_specs(
        "train", maneuver_table_path=maneuver_table, split_manifest_path=split_manifest,
        evidence_training_path=evidence_training, evidence_validation_path=evidence_validation,
    )
    assert specs[0].candidate_ids == ["MAN_X"]


# 8. source_shard / record_index mapping valid.
def test_source_shard_and_record_index_mapped(tmp_path):
    rows = [_maneuver_row("MAN_X", source_shard="shardZZZ", record_index=17)]
    evidence = [_evidence_row("MAN_X")]
    maneuver_table, split_manifest, evidence_training, evidence_validation = _write_dataset(
        tmp_path, rows, evidence
    )
    specs = load_decision_dataset_maneuver_specs(
        "train", maneuver_table_path=maneuver_table, split_manifest_path=split_manifest,
        evidence_training_path=evidence_training, evidence_validation_path=evidence_validation,
    )
    assert specs[0].source_shard == "shardZZZ"
    assert specs[0].record_index == 17


# 9. episode start field valid: merge_start_frame == interaction_evidence.commit_frame.
def test_merge_start_frame_uses_commit_frame(tmp_path):
    rows = [_maneuver_row("MAN_X")]
    evidence = [_evidence_row("MAN_X", commit_frame=63)]
    maneuver_table, split_manifest, evidence_training, evidence_validation = _write_dataset(
        tmp_path, rows, evidence
    )
    specs = load_decision_dataset_maneuver_specs(
        "train", maneuver_table_path=maneuver_table, split_manifest_path=split_manifest,
        evidence_training_path=evidence_training, evidence_validation_path=evidence_validation,
    )
    assert specs[0].merge_start_frame == 63


# 10. decision-context contract does NOT require CONFIRMED_MERGE.
def test_decision_context_contract_does_not_require_confirmed_merge(tmp_path):
    rows = [_maneuver_row("MAN_X")]
    evidence = [_evidence_row("MAN_X")]  # manual_validation == "UNREVIEWED"
    maneuver_table, split_manifest, evidence_training, evidence_validation = _write_dataset(
        tmp_path, rows, evidence
    )
    specs = load_decision_dataset_maneuver_specs(
        "train", maneuver_table_path=maneuver_table, split_manifest_path=split_manifest,
        evidence_training_path=evidence_training, evidence_validation_path=evidence_validation,
    )
    spec = specs[0]
    assert spec.manual_validation is None
    # require_schema must not raise for an UNREVIEWED/no-interaction-vehicle spec.
    spec.require_schema(MERGE_DATASET_SCHEMA)


# 11. legacy DATASET_SCHEMA_V2 still DOES require CONFIRMED_MERGE.
def test_legacy_schema_v2_still_requires_confirmed_merge():
    spec = ManeuverSpec(
        maneuver_id="MAN_LEGACY",
        source_shard="shard0",
        record_index=0,
        lane_chain=[1, 2],
        candidate_ids=["MAN_LEGACY"],
        merge_start_frame=10,
        schema_version=DATASET_SCHEMA_V2,
        maneuver_type="topological_merge",
        manual_validation="UNREVIEWED",
        topology_evidence={"a": 1},
        interaction_evidence={"front_vehicle_id": 5},
    )
    with pytest.raises(ValueError, match="not manually confirmed"):
        spec.require_schema(DATASET_SCHEMA_V2)


# 12. legacy ManeuverSpec.from_csv_row unchanged.
def test_legacy_from_csv_row_unchanged(tmp_path):
    maneuver_table = tmp_path / "maneuvers.csv"
    with open(maneuver_table, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["maneuver_id", "source_shard", "record_index",
                           "lane_chain", "candidate_ids"]
        )
        writer.writeheader()
        writer.writerow({
            "maneuver_id": "MAN_LEGACY", "source_shard": "shard0", "record_index": 0,
            "lane_chain": "1->2", "candidate_ids": "C1",
        })
    candidate_manifest = tmp_path / "candidates.csv"
    with open(candidate_manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "merge_start_frame"])
        writer.writeheader()
        writer.writerow({"candidate_id": "C1", "merge_start_frame": 5})

    with open(maneuver_table, newline="") as f:
        row = next(csv.DictReader(f))
    with open(candidate_manifest, newline="") as f:
        manifest_by_candidate_id = {r["candidate_id"]: r for r in csv.DictReader(f)}

    spec = ManeuverSpec.from_csv_row(row, manifest_by_candidate_id)
    assert spec.maneuver_id == "MAN_LEGACY"
    assert spec.lane_chain == [1, 2]
    assert spec.candidate_ids == ["C1"]
    assert spec.merge_start_frame == 5
    assert spec.schema_version == "merge_geometry_v1_legacy"


# 13. validation not accidentally loaded into training.
def test_validation_not_loaded_into_training(tmp_path):
    rows = [
        _maneuver_row("MAN_TRAIN", split="training"),
        _maneuver_row("MAN_VAL", split="validation"),
    ]
    evidence = [_evidence_row(r["maneuver_id"]) for r in rows]
    maneuver_table, split_manifest, evidence_training, evidence_validation = _write_dataset(
        tmp_path, rows, evidence
    )
    # validation evidence must be sourced from evidence_validation_path,
    # not evidence_training_path -- write MAN_VAL's evidence there too.
    with open(evidence_validation, "w") as f:
        f.write(json.dumps(_evidence_row("MAN_VAL")) + "\n")

    train_specs = load_decision_dataset_maneuver_specs(
        "train", maneuver_table_path=maneuver_table, split_manifest_path=split_manifest,
        evidence_training_path=evidence_training, evidence_validation_path=evidence_validation,
    )
    assert {s.maneuver_id for s in train_specs} == {"MAN_TRAIN"}

    val_specs = load_decision_dataset_maneuver_specs(
        "validation", maneuver_table_path=maneuver_table, split_manifest_path=split_manifest,
        evidence_training_path=evidence_training, evidence_validation_path=evidence_validation,
    )
    assert {s.maneuver_id for s in val_specs} == {"MAN_VAL"}


# 14. duplicate/missing zero.
def test_no_duplicate_maneuver_or_candidate_ids():
    specs = load_decision_dataset_maneuver_specs("train")
    maneuver_ids = [s.maneuver_id for s in specs]
    candidate_ids = [cid for s in specs for cid in s.candidate_ids]
    assert len(maneuver_ids) == len(set(maneuver_ids))
    assert len(candidate_ids) == len(set(candidate_ids))


def test_duplicate_candidate_id_raises(tmp_path):
    rows = [
        _maneuver_row("MAN_X", record_index=0),
        _maneuver_row("MAN_X", record_index=1),
    ]
    evidence = [_evidence_row("MAN_X")]
    maneuver_table, split_manifest, evidence_training, evidence_validation = _write_dataset(
        tmp_path, rows, evidence
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_decision_dataset_maneuver_specs(
            "train", maneuver_table_path=maneuver_table, split_manifest_path=split_manifest,
            evidence_training_path=evidence_training, evidence_validation_path=evidence_validation,
        )


# 15. train_ppo explicit IDs validation works.
def test_train_ppo_explicit_ids_validation_against_decision_dataset():
    from scripts.train_ppo import _resolve_maneuver_ids

    specs = load_decision_dataset_maneuver_specs("train")
    some_id = sorted(s.maneuver_id for s in specs)[0]
    resolved = _resolve_maneuver_ids(
        max_maneuvers=10, explicit_ids=some_id,
        train_ids=sorted(s.maneuver_id for s in specs),
    )
    assert resolved == [some_id]

    with pytest.raises(ValueError):
        _resolve_maneuver_ids(
            max_maneuvers=10, explicit_ids="NOT_A_REAL_MANEUVER_ID",
            train_ids=sorted(s.maneuver_id for s in specs),
        )


# 16. visualization checkpoint scope can resolve decision dataset maneuver IDs.
def test_visualize_resolve_maneuvers_checkpoint_scope_decision_dataset():
    import argparse
    from scripts.visualize_ppo import _resolve_maneuvers

    specs = load_decision_dataset_maneuver_specs("train")
    target_ids = sorted(s.maneuver_id for s in specs)[:2]

    class _FakeRestored:
        dataset_schema_version = MERGE_DATASET_SCHEMA
        maneuver_ids = target_ids

    args = argparse.Namespace(
        scope="checkpoint", allow_legacy_dataset=False,
        split_manifest=None, maneuver_table=None, candidate_manifest=None,
    )
    resolved = _resolve_maneuvers(args, _FakeRestored())
    assert sorted(s.maneuver_id for s in resolved) == target_ids
