"""Regression coverage for the resume/durability contract of
``scripts/extract_merge_v2_evidence.py``'s output JSONL: append-only
writes, skip-already-processed candidates, and progress-file-style
truncated-final-line recovery (matches
``scripts/build_womd_scenario_proto_locator.py``'s philosophy: a
truncated final line is recoverable, a malformed non-final line is
fatal corruption)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.extract_merge_v2_evidence import load_processed_candidates


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_no_existing_output_returns_empty_set(tmp_path):
    assert load_processed_candidates(tmp_path / "does_not_exist.jsonl") == set()


def test_processed_candidates_recovered_from_existing_output(tmp_path):
    out = tmp_path / "evidence.jsonl"
    _write_jsonl(out, [
        {"candidate_id": "A", "value": 1},
        {"candidate_id": "B", "value": 2},
    ])
    assert load_processed_candidates(out) == {"A", "B"}


def test_truncated_final_line_is_recovered_and_stripped_from_file(tmp_path):
    out = tmp_path / "evidence.jsonl"
    _write_jsonl(out, [{"candidate_id": "A", "value": 1}])
    with open(out, "a", encoding="utf-8") as f:
        f.write('{"candidate_id": "B", "val')  # truncated, no trailing newline

    processed = load_processed_candidates(out)

    # Truncated candidate B is not counted as processed -- it will be
    # retried on resume.
    assert processed == {"A"}
    # The broken partial line must be stripped from disk so a
    # subsequent append doesn't leave it stranded mid-file.
    remaining_lines = out.read_text().splitlines()
    assert len(remaining_lines) == 1
    assert json.loads(remaining_lines[0]) == {"candidate_id": "A", "value": 1}


def test_malformed_non_final_line_is_fatal(tmp_path):
    out = tmp_path / "evidence.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        f.write('{"candidate_id": "A", "broken\n')  # malformed, NOT the final line
        f.write(json.dumps({"candidate_id": "B", "value": 2}) + "\n")

    with pytest.raises(ValueError, match="malformed non-final"):
        load_processed_candidates(out)


def test_duplicate_candidate_id_in_existing_output_is_fatal(tmp_path):
    out = tmp_path / "evidence.jsonl"
    _write_jsonl(out, [
        {"candidate_id": "A", "value": 1},
        {"candidate_id": "A", "value": 2},
    ])
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        load_processed_candidates(out)


def test_resume_end_to_end_skips_existing_appends_new_no_duplicates(tmp_path, monkeypatch):
    """A+B input against an output already containing A must: skip A,
    process only B, and leave exactly one record each for A and B --
    exercising the same skip/append/no-duplicate contract main() uses,
    without invoking the full WOMD extraction pipeline."""

    out = tmp_path / "evidence.jsonl"
    _write_jsonl(out, [{"candidate_id": "A", "value": 1}])

    transition_rows = [
        {"candidate_id": "A"},
        {"candidate_id": "B"},
    ]
    processed = load_processed_candidates(out)
    remaining = [row for row in transition_rows if row["candidate_id"] not in processed]
    assert [row["candidate_id"] for row in remaining] == ["B"]

    with out.open("a", encoding="utf-8") as f:
        for row in remaining:
            f.write(json.dumps({"candidate_id": row["candidate_id"], "value": 99}) + "\n")

    final = [json.loads(line) for line in out.read_text().splitlines()]
    candidate_ids = [r["candidate_id"] for r in final]
    assert candidate_ids == ["A", "B"]
    assert len(candidate_ids) == len(set(candidate_ids))
