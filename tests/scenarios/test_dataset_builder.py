"""Tests for src/scenarios/dataset_builder.py using synthetic data.

No real WOMD data required: constructs CandidateRecord objects and
label dicts directly, matching the style of test_merge_detector.py.
"""

import csv

import pytest

from src.scenarios.dataset_builder import (
    CandidateRecord,
    compute_multi_shard_summary,
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
# Case C: combined-CSV deterministic sort across shards.
# ---------------------------------------------------------------------


def test_case_c_combined_csv_sorted_across_shards(tmp_path):
    # Two shards, deliberately inserted out of sort order, with
    # overlapping record_index/transition_frame values so the sort key
    # must actually discriminate on source_shard, not just record_index.
    candidate_shard_b_first = _make_candidate(
        scene_key="validation_tfexample.tfrecord-00005-of-00150#28",
        source_shard="validation_tfexample.tfrecord-00005-of-00150",
        record_index=28, transition_frame=50, source_lane_id=485, target_lane_id=344,
    )
    candidate_shard_a_second = _make_candidate(
        scene_key="validation_tfexample.tfrecord-00000-of-00150#28",
        source_shard="validation_tfexample.tfrecord-00000-of-00150",
        record_index=28, transition_frame=50, source_lane_id=485, target_lane_id=344,
    )
    candidate_shard_a_early_frame = _make_candidate(
        scene_key="validation_tfexample.tfrecord-00000-of-00150#5",
        source_shard="validation_tfexample.tfrecord-00000-of-00150",
        record_index=5, transition_frame=10, source_lane_id=1, target_lane_id=2,
        decision="reject", reason="parallel_lane_change",
    )

    output_path = tmp_path / "combined.csv"
    write_candidates_csv(
        [candidate_shard_b_first, candidate_shard_a_second, candidate_shard_a_early_frame],
        output_path,
    )

    with open(output_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Expected order: shard 00000 record 5 first (lower record_index),
    # then shard 00000 record 28, then shard 00005 record 28 (higher
    # source_shard string sorts after 00000).
    assert [r["source_shard"] for r in rows] == [
        "validation_tfexample.tfrecord-00000-of-00150",
        "validation_tfexample.tfrecord-00000-of-00150",
        "validation_tfexample.tfrecord-00005-of-00150",
    ]
    assert [int(r["record_index"]) for r in rows] == [5, 28, 28]

    # Re-running the same write must produce byte-identical ordering.
    output_path_2 = tmp_path / "combined_2.csv"
    write_candidates_csv(
        [candidate_shard_a_early_frame, candidate_shard_a_second, candidate_shard_b_first],
        output_path_2,
    )
    assert output_path.read_text() == output_path_2.read_text()


# ---------------------------------------------------------------------
# Case D: duplicate candidate_id hard failure, cross-shard variant.
# ---------------------------------------------------------------------


def test_case_d_cross_shard_duplicate_candidate_id_raises(tmp_path):
    # Same scene_key (hence same candidate_id) can only arise from the
    # SAME physical shard + record_index; this test confirms the
    # duplicate check still fires even when other fields superficially
    # differ (e.g. different transition_index), guarding against a
    # regression that might only dedupe on a subset of fields.
    candidate_a = _make_candidate(
        scene_key="validation_tfexample.tfrecord-00000-of-00150#28",
        source_shard="validation_tfexample.tfrecord-00000-of-00150",
        record_index=28, transition_frame=50, source_lane_id=485, target_lane_id=344,
        transition_index=0,
    )
    candidate_b = _make_candidate(
        scene_key="validation_tfexample.tfrecord-00000-of-00150#28",
        source_shard="validation_tfexample.tfrecord-00000-of-00150",
        record_index=28, transition_frame=50, source_lane_id=485, target_lane_id=344,
        transition_index=1,
    )
    with pytest.raises(ValueError, match="Duplicate candidate_id"):
        write_candidates_csv([candidate_a, candidate_b], tmp_path / "out.csv")


# ---------------------------------------------------------------------
# Case I: summary includes per-shard statistics.
# ---------------------------------------------------------------------


def test_case_i_multi_shard_summary_includes_per_shard_stats():
    shard_a_candidates = [
        _make_candidate(
            scene_key="shardA#0", source_shard="shardA", record_index=0,
            transition_frame=1, source_lane_id=1, target_lane_id=2,
            decision="accept",
        ),
        _make_candidate(
            scene_key="shardA#1", source_shard="shardA", record_index=1,
            transition_frame=2, source_lane_id=3, target_lane_id=4,
            decision="reject", reason="parallel_lane_change",
        ),
    ]
    shard_b_candidates = [
        _make_candidate(
            scene_key="shardB#0", source_shard="shardB", record_index=0,
            transition_frame=1, source_lane_id=1, target_lane_id=2,
            decision="review", reason="ambiguous_serial_or_merge",
        ),
    ]
    all_candidates = shard_a_candidates + shard_b_candidates

    summary = compute_multi_shard_summary(
        all_candidates,
        physical_shards_scanned=2,
        scenes_scanned=3,
        scenes_failed=0,
        per_shard_scan_counts={
            ("validation", "shardA"): {"scenes_scanned": 2, "scenes_failed": 0},
            ("validation", "shardB"): {"scenes_scanned": 1, "scenes_failed": 0},
        },
    )

    assert summary["global"]["physical_shards_scanned"] == 2
    assert summary["global"]["scenes_scanned"] == 3
    assert summary["global"]["total_stable_transitions"] == 3
    assert summary["global"]["accept_count"] == 1
    assert summary["global"]["reject_count"] == 1
    assert summary["global"]["review_count"] == 1

    per_shard = {(p["source_split"], p["source_shard"]): p for p in summary["per_shard"]}
    assert per_shard[("validation", "shardA")]["total_stable_transitions"] == 2
    assert per_shard[("validation", "shardA")]["accept_count"] == 1
    assert per_shard[("validation", "shardA")]["reject_count"] == 1
    assert per_shard[("validation", "shardA")]["scenes_scanned"] == 2

    assert per_shard[("validation", "shardB")]["total_stable_transitions"] == 1
    assert per_shard[("validation", "shardB")]["review_count"] == 1
    assert per_shard[("validation", "shardB")]["scenes_scanned"] == 1

    # per_shard entries sorted by (source_split, source_shard).
    assert [p["source_shard"] for p in summary["per_shard"]] == ["shardA", "shardB"]


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


# ---------------------------------------------------------------------
# Case M: manual labels remain unique and preserved across shards.
# ---------------------------------------------------------------------


def test_case_m_labels_preserved_independently_across_shards():
    # Two candidate_ids sharing the same record_index/transition_frame/
    # lane ids but differing by source_shard (embedded in scene_key,
    # hence in candidate_id) -- confirms sync_labels/labels storage
    # treats them as fully independent identities, never conflating or
    # deduplicating them just because their non-shard fields match.
    candidate_id_shard_a = make_candidate_id(
        "validation_tfexample.tfrecord-00000-of-00150#28", 50, 485, 344
    )
    candidate_id_shard_b = make_candidate_id(
        "validation_tfexample.tfrecord-00005-of-00150#28", 50, 485, 344
    )
    assert candidate_id_shard_a != candidate_id_shard_b

    existing_labels = {
        candidate_id_shard_a: {
            "candidate_id": candidate_id_shard_a,
            "manual_validation": "CONFIRMED_MERGE",
            "manual_note": "shard A: genuine merge",
        },
    }

    merged, stale = sync_labels(
        [candidate_id_shard_a, candidate_id_shard_b], existing_labels
    )

    # Shard A's pre-existing CONFIRMED_MERGE label is preserved exactly.
    assert merged[candidate_id_shard_a]["manual_validation"] == "CONFIRMED_MERGE"
    assert merged[candidate_id_shard_a]["manual_note"] == "shard A: genuine merge"

    # Shard B's identically-shaped candidate gets its OWN independent
    # UNREVIEWED row -- not silently merged with/overwritten by shard
    # A's label.
    assert merged[candidate_id_shard_b]["manual_validation"] == "UNREVIEWED"
    assert merged[candidate_id_shard_b]["manual_note"] == ""

    assert stale == []

    # Independently setting shard B's label must not affect shard A's.
    merged_2, _ = sync_labels(
        [candidate_id_shard_a, candidate_id_shard_b],
        merged,
        set_label=(candidate_id_shard_b, "CONFIRMED_NON_MERGE"),
    )
    assert merged_2[candidate_id_shard_a]["manual_validation"] == "CONFIRMED_MERGE"
    assert merged_2[candidate_id_shard_b]["manual_validation"] == "CONFIRMED_NON_MERGE"


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
