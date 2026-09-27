"""Tests for scripts/summarize_required_proto_shards.py's integrity
checks and required-shard-set computation."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import scripts.summarize_required_proto_shards as summarizer


def _write_targets(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", "source_split", "source_shard", "candidate_count", "candidate_ids", "v1_decisions_present"])
        for sid, split in rows:
            writer.writerow([sid, split, "shardX", 1, "cand", "review"])


def _write_locator(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", "proto_split", "proto_shard_index", "proto_total_shards", "record_index", "source_object", "matched_candidate_label"])
        for sid, split, shard_idx in rows:
            writer.writerow([sid, split, shard_idx, 1000, 0, f"shard{shard_idx}", "cand"])


def test_computes_required_shard_set_and_coverage(tmp_path, capsys):
    targets = tmp_path / "targets.csv"
    locator = tmp_path / "locator.csv"
    _write_targets(targets, [("a", "training"), ("b", "training"), ("c", "training")])
    _write_locator(locator, [("a", "training", 5), ("b", "training", 5), ("c", "training", 9)])

    exit_code = summarizer.main([
        "--target-scenario-ids", str(targets),
        "--locator-csv", str(locator),
        "--unresolved-output", str(tmp_path / "unresolved.csv"),
        "--shard-list-output-prefix", str(tmp_path / "required"),
    ])
    assert exit_code == 0

    training_shards = (tmp_path / "required_training.txt").read_text().split()
    assert training_shards == ["5", "9"]

    with open(tmp_path / "unresolved.csv") as f:
        unresolved = list(csv.DictReader(f))
    assert unresolved == []


def test_reports_unresolved_targets(tmp_path):
    targets = tmp_path / "targets.csv"
    locator = tmp_path / "locator.csv"
    _write_targets(targets, [("found", "training"), ("missing", "training")])
    _write_locator(locator, [("found", "training", 3)])

    summarizer.main([
        "--target-scenario-ids", str(targets),
        "--locator-csv", str(locator),
        "--unresolved-output", str(tmp_path / "unresolved.csv"),
        "--shard-list-output-prefix", str(tmp_path / "required"),
    ])

    with open(tmp_path / "unresolved.csv") as f:
        unresolved = list(csv.DictReader(f))
    assert unresolved == [{"scenario_id": "missing", "source_split": "training"}]


def test_detects_scenario_mapped_to_multiple_shards(tmp_path):
    targets = tmp_path / "targets.csv"
    locator = tmp_path / "locator.csv"
    _write_targets(targets, [("dup", "training")])
    _write_locator(locator, [("dup", "training", 3), ("dup", "training", 7)])

    exit_code = summarizer.main([
        "--target-scenario-ids", str(targets),
        "--locator-csv", str(locator),
        "--unresolved-output", str(tmp_path / "unresolved.csv"),
        "--shard-list-output-prefix", str(tmp_path / "required"),
    ])
    assert exit_code == 1


def test_splits_never_mixed_in_required_shards(tmp_path):
    targets = tmp_path / "targets.csv"
    locator = tmp_path / "locator.csv"
    _write_targets(targets, [("t1", "training"), ("v1", "validation")])
    _write_locator(locator, [("t1", "training", 2), ("v1", "validation", 2)])

    summarizer.main([
        "--target-scenario-ids", str(targets),
        "--locator-csv", str(locator),
        "--unresolved-output", str(tmp_path / "unresolved.csv"),
        "--shard-list-output-prefix", str(tmp_path / "required"),
    ])

    assert (tmp_path / "required_training.txt").read_text().split() == ["2"]
    assert (tmp_path / "required_validation.txt").read_text().split() == ["2"]
