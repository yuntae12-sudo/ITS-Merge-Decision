"""Integrity tests for scripts/build_womd_scenario_proto_locator.py.

Covers: exact match recording, non-target rejection, dedup, DONE-shard
resume skip, FAILED-shard retry, early stop, deterministic parallel
merge, and split isolation -- see the task spec's Section 13 checklist.
"""

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import scripts.build_womd_scenario_proto_locator as locator


def _fake_scenario(scenario_id):
    return types.SimpleNamespace(scenario_id=scenario_id, map_features=[])


def _write_targets_csv(path, rows):
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", "source_split", "source_shard", "candidate_count", "candidate_ids", "v1_decisions_present"])
        for sid, split, shard, label in rows:
            writer.writerow([sid, split, shard, 1, label, "review"])


def test_scan_one_shard_records_exact_match(monkeypatch):
    monkeypatch.setattr(
        locator,
        "iter_scenario_protobufs",
        lambda paths: iter([_fake_scenario("keep-1"), _fake_scenario("skip-1"), _fake_scenario("keep-2")]),
    )
    targets = {"keep-1": {"label": "c1"}, "keep-2": {"label": "c2"}}
    record_count, matches = locator.scan_one_shard(
        "shard-path", 0, 1000, "training", {"keep-1", "keep-2"}, targets
    )
    assert record_count == 3
    assert set(matches) == {"keep-1", "keep-2"}
    assert matches["keep-1"]["record_index"] == 0
    assert matches["keep-2"]["record_index"] == 2


def test_scan_one_shard_ignores_non_target_scenarios(monkeypatch):
    monkeypatch.setattr(
        locator,
        "iter_scenario_protobufs",
        lambda paths: iter([_fake_scenario("irrelevant-1"), _fake_scenario("irrelevant-2")]),
    )
    record_count, matches = locator.scan_one_shard(
        "shard-path", 0, 1000, "training", {"wanted-only"}, {"wanted-only": {"label": "x"}}
    )
    assert record_count == 2
    assert matches == {}


def test_write_locator_csv_deduplicates_and_is_deterministic(tmp_path):
    out = tmp_path / "locator.csv"
    matches = {
        "b-id": {
            "scenario_id": "b-id", "proto_split": "training", "proto_shard_index": 5,
            "proto_total_shards": 1000, "record_index": 1, "source_object": "shard5",
            "matched_candidate_label": "c-b",
        },
        "a-id": {
            "scenario_id": "a-id", "proto_split": "training", "proto_shard_index": 2,
            "proto_total_shards": 1000, "record_index": 9, "source_object": "shard2",
            "matched_candidate_label": "c-a",
        },
    }
    locator.write_locator_csv(out, matches)
    text1 = out.read_text()
    # Rewriting the same dict (arbitrary insertion order) must reproduce
    # byte-identical output -- required for deterministic parallel merge.
    locator.write_locator_csv(out, {"a-id": matches["a-id"], "b-id": matches["b-id"]})
    text2 = out.read_text()
    assert text1 == text2
    assert text1.index("a-id") < text1.index("b-id")


def test_done_shard_is_skipped_on_resume(tmp_path):
    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        json.dumps({"split": "training", "shard_index": 3, "status": "DONE"}) + "\n"
        + json.dumps({"split": "training", "shard_index": 7, "status": "DONE"}) + "\n"
        + json.dumps({"split": "validation", "shard_index": 3, "status": "DONE"}) + "\n"
    )
    done = locator.load_done_shards(str(progress), "training")
    assert done == {3, 7}


def test_failed_shard_is_not_treated_as_done(tmp_path):
    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        json.dumps({"split": "training", "shard_index": 3, "status": "FAILED"}) + "\n"
    )
    done = locator.load_done_shards(str(progress), "training")
    assert done == set()


def test_progress_from_different_target_set_is_rejected(tmp_path):
    progress = tmp_path / "progress.jsonl"
    progress.write_text(json.dumps({
        "split": "training", "shard_index": 3, "status": "DONE",
        "target_sha256": "old-targets",
    }) + "\n")
    import pytest

    with pytest.raises(ValueError, match="target checksum mismatch"):
        locator.load_done_shards(str(progress), "training", "new-targets")


def test_progress_file_recovers_after_truncated_final_line(tmp_path):
    """A process killed mid-write can leave one truncated trailing line;
    earlier, already-flushed DONE entries must still resume correctly."""

    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        json.dumps({"split": "training", "shard_index": 0, "status": "DONE"}) + "\n"
        + '{"split": "training", "shard_index": 1, "stat'  # truncated, no trailing newline
    )
    done = locator.load_done_shards(str(progress), "training")
    assert done == {0}


def test_progress_file_rejects_malformed_nonfinal_record(tmp_path):
    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        '{"split": "training", "shard_index": 0, "stat\n'
        + json.dumps({"split": "training", "shard_index": 1, "status": "DONE"}) + "\n"
    )
    import pytest

    with pytest.raises(json.JSONDecodeError):
        locator.load_done_shards(str(progress), "training")


def test_early_stop_when_all_targets_found(monkeypatch, tmp_path):
    _write_targets_csv(
        tmp_path / "targets.csv",
        [("only-target", "training", "shardA", "cand-1")],
    )
    scanned_shards = []

    def fake_scan(shard_path, shard_index, total_shards, split, remaining_ids, targets):
        scanned_shards.append(shard_index)
        return 10, {"only-target": {
            "scenario_id": "only-target", "proto_split": split, "proto_shard_index": shard_index,
            "proto_total_shards": total_shards, "record_index": 0, "source_object": shard_path,
            "matched_candidate_label": "cand-1",
        }}

    monkeypatch.setattr(locator, "scan_one_shard", fake_scan)
    exit_code = locator.main([
        "--target-scenario-ids", str(tmp_path / "targets.csv"),
        "--split", "training",
        "--local-shard-paths", "shard0", "shard1", "shard2",
        "--locator-output", str(tmp_path / "out.csv"),
        "--progress-file", str(tmp_path / "progress.jsonl"),
        "--workers", "1",  # serial: proves early stop, not just a small in-flight batch
    ])
    assert exit_code == 0
    # Only the first shard should ever be scanned once the sole target is found.
    assert scanned_shards == [0]


def test_failed_shard_returns_nonzero_and_is_retried(monkeypatch, tmp_path):
    _write_targets_csv(
        tmp_path / "targets.csv",
        [("only-target", "training", "shardA", "cand-1")],
    )
    attempts = []

    def failing_scan(*args):
        attempts.append(args[1])
        raise OSError("temporary network failure")

    monkeypatch.setattr(locator, "scan_one_shard", failing_scan)
    common_args = [
        "--target-scenario-ids", str(tmp_path / "targets.csv"),
        "--split", "training",
        "--local-shard-paths", "shard0",
        "--locator-output", str(tmp_path / "out.csv"),
        "--progress-file", str(tmp_path / "progress.jsonl"),
        "--workers", "1",
    ]
    assert locator.main(common_args) == 1

    def succeeding_scan(shard_path, shard_index, total_shards, split, remaining_ids, targets):
        attempts.append(shard_index)
        return 1, {"only-target": {
            "scenario_id": "only-target", "proto_split": split,
            "proto_shard_index": shard_index, "proto_total_shards": total_shards,
            "record_index": 0, "source_object": shard_path,
            "matched_candidate_label": "cand-1",
        }}

    monkeypatch.setattr(locator, "scan_one_shard", succeeding_scan)
    assert locator.main(common_args) == 0
    assert attempts == [0, 0]


def test_split_isolation_never_mixes_training_and_validation(tmp_path):
    _write_targets_csv(
        tmp_path / "targets.csv",
        [
            ("train-id", "training", "shardA", "cand-t"),
            ("val-id", "validation", "shardB", "cand-v"),
        ],
    )
    training_targets = locator.load_targets(str(tmp_path / "targets.csv"), "training")
    validation_targets = locator.load_targets(str(tmp_path / "targets.csv"), "validation")
    assert set(training_targets) == {"train-id"}
    assert set(validation_targets) == {"val-id"}


def test_parallel_workers_never_double_record_same_scenario(monkeypatch, tmp_path):
    """Two shards both happen to contain the same target scenario_id
    (should not occur in real WOMD data, but the merge logic must not
    assume it never does): the parent's serial remaining_ids update must
    keep only the first shard's match, discarding the second -- proving
    the dedup guard, not relying on which shard callback lands first."""

    _write_targets_csv(
        tmp_path / "targets.csv",
        [("dup-target", "training", "shardA", "cand-1")],
    )

    def fake_scan(shard_path, shard_index, total_shards, split, remaining_ids, targets):
        # Both shard 0 and shard 1 "contain" the same scenario_id.
        if "dup-target" in remaining_ids:
            return 5, {"dup-target": {
                "scenario_id": "dup-target", "proto_split": split, "proto_shard_index": shard_index,
                "proto_total_shards": total_shards, "record_index": 0, "source_object": shard_path,
                "matched_candidate_label": "cand-1",
            }}
        return 5, {}

    monkeypatch.setattr(locator, "scan_one_shard", fake_scan)
    locator.main([
        "--target-scenario-ids", str(tmp_path / "targets.csv"),
        "--split", "training",
        "--local-shard-paths", "shard0", "shard1",
        "--locator-output", str(tmp_path / "out.csv"),
        "--progress-file", str(tmp_path / "progress.jsonl"),
        "--workers", "4",
    ])

    import csv

    with open(tmp_path / "out.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    matching = [r for r in rows if r["scenario_id"] == "dup-target"]
    assert len(matching) == 1, "duplicate scenario_id must be recorded exactly once"


def test_resume_from_existing_locator_csv_narrows_remaining_targets(tmp_path, monkeypatch):
    _write_targets_csv(
        tmp_path / "targets.csv",
        [
            ("already-found", "training", "shardA", "cand-1"),
            ("still-missing", "training", "shardA", "cand-2"),
        ],
    )
    locator_csv = tmp_path / "locator.csv"
    locator.write_locator_csv(locator_csv, {
        "already-found": {
            "scenario_id": "already-found", "proto_split": "training", "proto_shard_index": 0,
            "proto_total_shards": 1000, "record_index": 0, "source_object": "shard0",
            "matched_candidate_label": "cand-1",
        }
    })

    scanned_ids_seen = []

    def fake_scan(shard_path, shard_index, total_shards, split, remaining_ids, targets):
        scanned_ids_seen.append(set(remaining_ids))
        return 1, {}

    monkeypatch.setattr(locator, "scan_one_shard", fake_scan)
    locator.main([
        "--target-scenario-ids", str(tmp_path / "targets.csv"),
        "--split", "training",
        "--local-shard-paths", "shard1",
        "--start-shard", "1",
        "--locator-output", str(locator_csv),
        "--progress-file", str(tmp_path / "progress.jsonl"),
    ])
    assert scanned_ids_seen == [{"still-missing"}]
