"""Tests for src/scenarios/dataset_builder.py using synthetic data.

No real WOMD data required: constructs CandidateRecord objects and
label dicts directly, matching the style of test_merge_detector.py.
"""

import csv

import pytest

from src.scenarios.dataset_builder import (
    CandidateRecord,
    compute_summary_statistics,
    make_candidate_id,
    sanitize_candidate_id_for_filename,
    write_candidates_csv,
)


def _make_candidate(
    scene_key="validation_tfexample.tfrecord-00000-of-00150#28",
    transition_frame=50,
    source_lane_id=485,
    target_lane_id=344,
    decision="accept",
    reason=None,
    record_index=28,
    transition_index=0,
    **overrides,
):
    candidate_id = make_candidate_id(
        scene_key, transition_frame, source_lane_id, target_lane_id
    )
    fields = dict(
        candidate_id=candidate_id,
        scene_key=scene_key,
        source_dataset="WOMD",
        source_split="validation",
        source_shard="validation_tfexample.tfrecord-00000-of-00150",
        record_index=record_index,
        transition_index=transition_index,
        transition_frame=transition_frame,
        source_lane_id=source_lane_id,
        target_lane_id=target_lane_id,
        source_start_frame=0,
        source_end_frame=transition_frame - 1,
        target_start_frame=transition_frame,
        target_end_frame=transition_frame + 9,
        decision=decision,
        reason=reason,
        source_lane_ends=True,
        source_remaining_distance_m=0.0,
        endpoint_target_distance_m=2.14,
        endpoint_target_arc_length_m=12.0,
        heading_difference_deg=0.5,
        lanes_converge=True,
        parallel_continuation=False,
        separation_reduction_m=5.0,
        decreasing_fraction=0.9,
        max_collinear_offset_m=3.0,
        upstream_separation_m=6.0,
        pre_merge_frames=20,
        target_lane_persistent=True,
    )

    if decision == "accept":
        fields.update(
            merge_start_s=10.0,
            merge_end_s=20.0,
            merge_start_frame=transition_frame - 5,
            merge_complete_frame=transition_frame,
            ego_longitudinal_speed_mps=15.0,
            merge_distance_m=5.0,
            front_vehicle_id=99,
            front_gap_m=12.0,
            front_relative_speed_mps=1.0,
            front_ttc_s=12.0,
            rear_vehicle_id=None,
            rear_gap_m=None,
            rear_relative_speed_mps=None,
            rear_ttc_s=float("inf"),
            traffic_density=3,
        )

    fields.update(overrides)
    return CandidateRecord(**fields)


def test_accept_candidate_round_trips_all_fields(tmp_path):
    candidate = _make_candidate(decision="accept")
    output_path = tmp_path / "candidates.csv"
    write_candidates_csv([candidate], output_path)

    with open(output_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["candidate_id"] == candidate.candidate_id
    assert row["scene_key"] == candidate.scene_key
    assert row["source_lane_id"] == "485"
    assert row["target_lane_id"] == "344"
    assert row["decision"] == "accept"
    assert row["reason"] == ""
    assert row["front_vehicle_id"] == "99"
    assert row["rear_vehicle_id"] == ""
    assert row["rear_ttc_s"] == "inf"
    assert row["merge_start_frame"] == str(candidate.merge_start_frame)


def test_review_candidate_stays_review_in_csv(tmp_path):
    candidate = _make_candidate(
        decision="review", reason="ambiguous_serial_or_merge",
        transition_frame=60, source_lane_id=196, target_lane_id=206,
    )
    output_path = tmp_path / "candidates.csv"
    write_candidates_csv([candidate], output_path)

    with open(output_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["decision"] == "review"
    assert rows[0]["reason"] == "ambiguous_serial_or_merge"
    # ACCEPT-only fields must be blank, not silently filled in.
    assert rows[0]["merge_start_s"] == ""
    assert rows[0]["front_vehicle_id"] == ""


def test_reject_candidate_stays_reject_in_csv(tmp_path):
    candidate = _make_candidate(
        decision="reject", reason="parallel_lane_change",
        transition_frame=70, source_lane_id=10, target_lane_id=20,
    )
    output_path = tmp_path / "candidates.csv"
    write_candidates_csv([candidate], output_path)

    with open(output_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["decision"] == "reject"
    assert rows[0]["reason"] == "parallel_lane_change"
    assert rows[0]["merge_start_s"] == ""


def test_candidate_id_determinism():
    id_a = make_candidate_id("shard#0", 10, 1, 2)
    id_a_again = make_candidate_id("shard#0", 10, 1, 2)
    assert id_a == id_a_again

    id_diff_frame = make_candidate_id("shard#0", 11, 1, 2)
    id_diff_lanes = make_candidate_id("shard#0", 10, 1, 3)

    assert id_a != id_diff_frame
    assert id_a != id_diff_lanes


def test_sanitize_candidate_id_for_filename():
    candidate_id = "validation_tfexample.tfrecord-00000-of-00150#28__t50__485_344"
    sanitized = sanitize_candidate_id_for_filename(candidate_id)
    assert "#" not in sanitized
    assert "/" not in sanitized
    assert ":" not in sanitized
    # Original id is untouched.
    assert "#" in candidate_id


def test_count_invariant_holds_for_valid_batch():
    candidates = [
        _make_candidate(decision="accept", transition_frame=1, source_lane_id=1, target_lane_id=2),
        _make_candidate(decision="reject", reason="parallel_lane_change", transition_frame=2, source_lane_id=3, target_lane_id=4),
        _make_candidate(decision="review", reason="ambiguous_serial_or_merge", transition_frame=3, source_lane_id=5, target_lane_id=6),
    ]
    summary = compute_summary_statistics(candidates)
    assert summary["accept_count"] == 1
    assert summary["reject_count"] == 1
    assert summary["review_count"] == 1
    assert summary["total_stable_transitions"] == 3
    assert summary["review_ratio"] == pytest.approx(1 / 3)
    assert summary["review_ratio_among_accept_review"] == pytest.approx(0.5)


def test_duplicate_candidate_id_raises(tmp_path):
    candidate_a = _make_candidate(
        scene_key="shard#0", transition_frame=10, source_lane_id=1, target_lane_id=2
    )
    candidate_b = _make_candidate(
        scene_key="shard#0", transition_frame=10, source_lane_id=1, target_lane_id=2
    )
    with pytest.raises(ValueError):
        write_candidates_csv([candidate_a, candidate_b], tmp_path / "out.csv")


# ---------------------------------------------------------------------
# Manual label sync / final manifest logic (exercises
# scripts/build_merge_manifest.py's pure functions directly).
# ---------------------------------------------------------------------

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

from scripts.build_merge_manifest import sync_labels


def test_manual_label_preserved_across_resync():
    existing_labels = {
        "A": {"candidate_id": "A", "manual_validation": "CONFIRMED_MERGE", "manual_note": "looks real"},
    }
    merged, stale = sync_labels(["A", "B"], existing_labels)

    assert merged["A"]["manual_validation"] == "CONFIRMED_MERGE"
    assert merged["A"]["manual_note"] == "looks real"
    assert stale == []


def test_new_candidate_gets_unreviewed():
    merged, stale = sync_labels(["A", "B"], {})
    assert merged["A"]["manual_validation"] == "UNREVIEWED"
    assert merged["B"]["manual_validation"] == "UNREVIEWED"
    assert merged["A"]["manual_note"] == ""


def test_stale_labels_kept_not_deleted():
    existing_labels = {
        "OLD": {"candidate_id": "OLD", "manual_validation": "CONFIRMED_MERGE", "manual_note": ""},
    }
    merged, stale = sync_labels(["NEW"], existing_labels)
    assert "OLD" in merged
    assert merged["OLD"]["manual_validation"] == "CONFIRMED_MERGE"
    assert stale == ["OLD"]


def test_set_label_applies_to_merged_set():
    merged, _ = sync_labels(["A"], {}, set_label=("A", "CONFIRMED_MERGE"))
    assert merged["A"]["manual_validation"] == "CONFIRMED_MERGE"


def test_set_label_rejects_invalid_value():
    with pytest.raises(ValueError):
        sync_labels(["A"], {}, set_label=("A", "NOT_A_REAL_LABEL"))


# NOTE: the final-manifest inclusion rule (ACCEPT/REVIEW x
# CONFIRMED_MERGE/CONFIRMED_NON_MERGE/UNREVIEWED) is now covered
# end-to-end in tests/scenarios/test_build_merge_manifest.py (Cases
# E/F/L), using a real reconstructable synthetic scene -- required
# since `build_manifest` now reloads and reconstructs each confirmed
# candidate's transition rather than blindly copying CSV columns (see
# that module's docstring for the fix this guards against). The
# pure-function sync_labels tests above remain here since they don't
# depend on manifest-building at all.
