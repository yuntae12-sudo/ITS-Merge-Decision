"""Split-label integration compatibility fix: the final MERGE Decision
Dataset v2 canonical split manifest
(data/manifests/v2/merge_decision_split_v2.csv) uses "training"/
"validation" while legacy Phase 2 manifests
(data/manifests/phase2_dataset_split.csv) use "train"/"validation".
PPO/evaluator code must treat "train" and "training" as the same
semantic TRAIN split without rewriting either manifest -- see
src.environment.dataset_split.normalize_split_name."""

import csv

import pytest

from src.environment.full_split_evaluator import load_maneuver_specs
from scripts.train_ppo import _resolve_maneuver_ids


def _write_manifest(tmp_path, rows, train_label):
    """Writes a minimal maneuver_table + candidate_manifest + split
    manifest triple using ``train_label`` ("train" or "training") for
    the TRAIN rows, mirroring one real canonical manifest's shape."""

    maneuver_table = tmp_path / "maneuvers.csv"
    candidate_manifest = tmp_path / "candidates.csv"
    split_manifest = tmp_path / "split.csv"

    with open(maneuver_table, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["maneuver_id", "source_shard", "record_index",
                           "lane_chain", "candidate_ids"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "maneuver_id": row["maneuver_id"],
                "source_shard": "shard0",
                "record_index": 0,
                "lane_chain": "1->2",
                "candidate_ids": row["candidate_id"],
            })

    with open(candidate_manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "merge_start_frame"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"candidate_id": row["candidate_id"], "merge_start_frame": 0})

    with open(split_manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["maneuver_id", "scene_key", "split"])
        writer.writeheader()
        for row in rows:
            label = train_label if row["split"] == "train" else row["split"]
            writer.writerow({
                "maneuver_id": row["maneuver_id"],
                "scene_key": row["maneuver_id"] + "_scene",
                "split": label,
            })

    return str(maneuver_table), str(candidate_manifest), str(split_manifest)


_ROWS = [
    {"maneuver_id": "MAN_TRAIN_1", "candidate_id": "C1", "split": "train"},
    {"maneuver_id": "MAN_TRAIN_2", "candidate_id": "C2", "split": "train"},
    {"maneuver_id": "MAN_VAL_1", "candidate_id": "C3", "split": "validation"},
]


def test_legacy_train_manifest_load(tmp_path):
    maneuver_table, candidate_manifest, split_manifest = _write_manifest(
        tmp_path, _ROWS, train_label="train"
    )
    specs = load_maneuver_specs(
        "train",
        split_manifest_path=split_manifest,
        maneuver_table_path=maneuver_table,
        candidate_manifest_path=candidate_manifest,
    )
    assert {s.maneuver_id for s in specs} == {"MAN_TRAIN_1", "MAN_TRAIN_2"}


def test_final_v2_style_training_manifest_load(tmp_path):
    maneuver_table, candidate_manifest, split_manifest = _write_manifest(
        tmp_path, _ROWS, train_label="training"
    )
    specs = load_maneuver_specs(
        "train",
        split_manifest_path=split_manifest,
        maneuver_table_path=maneuver_table,
        candidate_manifest_path=candidate_manifest,
    )
    assert {s.maneuver_id for s in specs} == {"MAN_TRAIN_1", "MAN_TRAIN_2"}


def test_validation_load_unaffected(tmp_path):
    for train_label in ("train", "training"):
        maneuver_table, candidate_manifest, split_manifest = _write_manifest(
            tmp_path, _ROWS, train_label=train_label
        )
        specs = load_maneuver_specs(
            "validation",
            split_manifest_path=split_manifest,
            maneuver_table_path=maneuver_table,
            candidate_manifest_path=candidate_manifest,
        )
        assert {s.maneuver_id for s in specs} == {"MAN_VAL_1"}


def test_explicit_maneuver_validation_with_training_label(tmp_path):
    _, _, split_manifest = _write_manifest(tmp_path, _ROWS, train_label="training")
    resolved = _resolve_maneuver_ids(
        max_maneuvers=10, explicit_ids="MAN_TRAIN_1", split_manifest_path=split_manifest
    )
    assert resolved == ["MAN_TRAIN_1"]


def test_explicit_maneuver_validation_rejects_non_train(tmp_path):
    _, _, split_manifest = _write_manifest(tmp_path, _ROWS, train_label="training")
    with pytest.raises(ValueError):
        _resolve_maneuver_ids(
            max_maneuvers=10, explicit_ids="MAN_VAL_1", split_manifest_path=split_manifest
        )


def test_unknown_split_label_never_treated_as_train(tmp_path):
    rows = list(_ROWS) + [
        {"maneuver_id": "MAN_TUNE_1", "candidate_id": "C4", "split": "tune"}
    ]
    maneuver_table = tmp_path / "maneuvers.csv"
    candidate_manifest = tmp_path / "candidates.csv"
    split_manifest = tmp_path / "split.csv"

    with open(maneuver_table, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["maneuver_id", "source_shard", "record_index",
                           "lane_chain", "candidate_ids"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "maneuver_id": row["maneuver_id"],
                "source_shard": "shard0",
                "record_index": 0,
                "lane_chain": "1->2",
                "candidate_ids": row["candidate_id"],
            })

    with open(candidate_manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "merge_start_frame"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"candidate_id": row["candidate_id"], "merge_start_frame": 0})

    with open(split_manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["maneuver_id", "scene_key", "split"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "maneuver_id": row["maneuver_id"],
                "scene_key": row["maneuver_id"] + "_scene",
                "split": row["split"],
            })

    specs = load_maneuver_specs(
        "train",
        split_manifest_path=str(split_manifest),
        maneuver_table_path=str(maneuver_table),
        candidate_manifest_path=str(candidate_manifest),
    )
    assert "MAN_TUNE_1" not in {s.maneuver_id for s in specs}
