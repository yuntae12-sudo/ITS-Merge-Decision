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


# ---------------------------------------------------------------------
# Feature-reference-frame regression tests (fix commit "materialize
# merge state at pre-merge reference frame"). Synthetic scene, matching
# the style of test_build_merge_manifest.py's _build_merge_scene /
# test_merge_detector.py's synthetic-geometry construction.
# ---------------------------------------------------------------------

import types

import numpy as np

from src.scenarios.dataset_builder import (
    _derive_merge_frames,
    _resolve_feature_reference_frame,
    build_candidate_records,
    materialize_merge_features,
)
from src.scenarios.lane_assignment import (
    LaneAssignmentConfig,
    assign_ego_lane_sequence,
    compute_stable_lane_sequence,
    find_lane_transitions,
)
from src.scenarios.lane_geometry import extract_lane_polylines
from src.scenarios.merge_detector import MergeDecision, MergeTopologyConfig, detect_merge
from src.scenarios.scenario_features import AgentSelectionConfig

_LANE_ASSIGNMENT_CONFIG = LaneAssignmentConfig(
    max_lateral_distance_m=5.0,
    max_heading_difference_deg=45.0,
    persistence_frames=5,
    max_ambiguous_gap_frames=5,
    candidate_count=8,
)

_MERGE_TOPOLOGY_CONFIG = MergeTopologyConfig(
    max_source_end_distance_m=15.0,
    max_endpoint_target_distance_m=5.0,
    max_heading_difference_deg=20.0,
    convergence_window_m=30.0,
    convergence_sample_count=7,
    min_separation_reduction_m=2.0,
    min_decreasing_fraction=0.6,
    serial_continuation_max_lateral_m=0.5,
    min_pre_merge_frames=5,
    min_target_lane_frames=5,
)

_AGENT_SELECTION_CONFIG = AgentSelectionConfig(
    max_target_lane_lateral_distance_m=5.0,
    max_target_lane_heading_difference_deg=45.0,
    max_distance_m=100.0,
    density_radius_m=50.0,
)


def _roadgraph_from_lanes(lanes):
    all_ids, all_x, all_y, all_dx, all_dy, all_types, all_valid = (
        [], [], [], [], [], [], [],
    )
    for lane_id, xy in lanes:
        xy = np.asarray(xy, dtype=np.float64)
        diffs = np.diff(xy, axis=0)
        norms = np.hypot(diffs[:, 0], diffs[:, 1])
        norms[norms == 0] = 1.0
        unit = diffs / norms[:, None]
        direction = np.zeros_like(xy)
        direction[:-1] = unit
        direction[-1] = unit[-1]

        n = xy.shape[0]
        all_ids.extend([lane_id] * n)
        all_x.extend(xy[:, 0].tolist())
        all_y.extend(xy[:, 1].tolist())
        all_dx.extend(direction[:, 0].tolist())
        all_dy.extend(direction[:, 1].tolist())
        all_types.extend([2] * n)
        all_valid.extend([True] * n)

    return types.SimpleNamespace(
        ids=np.array(all_ids),
        x=np.array(all_x),
        y=np.array(all_y),
        dir_x=np.array(all_dx),
        dir_y=np.array(all_dy),
        types=np.array(all_types),
        valid=np.array(all_valid),
    )


# The raw source->target lane switch happens at frame 20, but
# persistence_frames=5 hysteresis (see compute_stable_lane_sequence)
# delays the STABLE transition_frame to 20 + (persistence_frames - 1)
# = 24 -- confirmed empirically below via time_separated_setup.
# merge_start_frame is a function of this scene's geometry (the first
# frame whose source-lane arc length reaches merge_start_s, computed
# by compute_merge_start_end_s): with this source curve's shape it
# lands at frame 1 -- clearly separated in time from transition_frame
# (24), which is exactly what the A23 regression needs (any two
# distinct frames with deliberately different state would do; 1 and 24
# were confirmed empirically, not hand-picked to coincidentally match).
_TIME_SEPARATED_MERGE_START_FRAME = 1
_TIME_SEPARATED_TRANSITION_FRAME = 24


def _build_time_separated_merge_scene():
    """Synthetic true-merge scene (same geometric shape as
    test_build_merge_manifest.py's _build_merge_scene -- large upstream
    separation shrinking to just outside serial_continuation_max_lateral_m
    at the endpoint, an ACCEPT case) with 40 frames, engineered so
    ``merge_start_frame == 10`` and ``transition_frame == 24`` land at
    DIFFERENT frames, and with DELIBERATELY DIFFERENT ego/front state at
    frame 10 vs frame 24 (mandatory A23 regression): ego speed 5 m/s and
    front gap ~20 m at frame 10, vs ego speed 15 m/s and front gap ~2 m
    at frame 24 -- so a test that accidentally used transition_frame
    instead of the resolved feature_reference_frame would read back
    obviously wrong (frame-24) values instead of frame-10's.
    """

    source_lane_id, target_lane_id = 196, 206
    num_frames = 40
    transition_at = 20  # raw switch frame; stable transition lands at 24

    source_xs = np.linspace(0.0, 30.0, 31)
    source_xy = np.stack([source_xs, 10.0 - (source_xs / 30.0) * 9.4], axis=1)

    target_xs = np.linspace(30.0, 90.0, 61)
    target_xy = np.stack([target_xs, np.zeros_like(target_xs)], axis=1)

    roadgraph_points = _roadgraph_from_lanes(
        [(source_lane_id, source_xy), (target_lane_id, target_xy)]
    )

    num_objects = 2  # ego + front (no rear candidate)
    x = np.zeros((num_objects, num_frames))
    y = np.zeros((num_objects, num_frames))
    yaw = np.zeros((num_objects, num_frames))
    vel_x = np.zeros((num_objects, num_frames))
    vel_y = np.zeros((num_objects, num_frames))
    length = np.full((num_objects, num_frames), 4.5)
    valid = np.ones((num_objects, num_frames), dtype=bool)
    object_types = np.array([1, 1])

    ego_idx = 0
    for frame in range(num_frames):
        if frame < transition_at:
            frac = frame / max(transition_at - 1, 1)
            idx = min(int(round(frac * 30)), source_xy.shape[0] - 1)
            pos = source_xy[idx]
            x[ego_idx, frame] = pos[0]
            y[ego_idx, frame] = pos[1]
            if idx < source_xy.shape[0] - 1:
                seg = source_xy[idx + 1] - source_xy[idx]
            else:
                seg = source_xy[idx] - source_xy[idx - 1]
            yaw[ego_idx, frame] = float(np.arctan2(seg[1], seg[0]))
        else:
            frac = (frame - transition_at) / max(num_frames - transition_at - 1, 1)
            idx = min(int(round(frac * (target_xy.shape[0] - 1))), target_xy.shape[0] - 1)
            pos = target_xy[idx]
            x[ego_idx, frame] = pos[0]
            y[ego_idx, frame] = pos[1]
            yaw[ego_idx, frame] = 0.0

        # Deliberately different ego speed at frame 10 (5 m/s) vs frame
        # 24 (15 m/s) -- and everywhere else a plausible in-between/
        # matching value so this doesn't perturb the transition
        # geometry itself.
        if frame == _TIME_SEPARATED_MERGE_START_FRAME:
            vel_x[ego_idx, frame] = 5.0
        else:
            vel_x[ego_idx, frame] = 15.0
        vel_y[ego_idx, frame] = 0.0

    # Front vehicle on the target lane throughout, positioned so the
    # gap to ego is ~20 m at the merge_start_frame and ~2 m at
    # transition_frame (both measured in TARGET-lane arc length -- at
    # merge_start_frame (1) ego's x is ~2.1 (still on the source lane,
    # far upstream of the target lane's x=30 start, so its target-lane
    # projection clamps near arc length 0); at transition_frame (24)
    # ego's x is ~43.0 (already on the target lane, ~13 m of target arc
    # length covered).
    front_idx = 1
    front_x = np.linspace(50.0, 90.0, num_frames)
    front_x[_TIME_SEPARATED_MERGE_START_FRAME] = 51.0  # ~50+ m ahead of ego's clamped ~0 target-arc position
    front_x[_TIME_SEPARATED_TRANSITION_FRAME] = 45.0  # ~2 m ahead of ego's frame-24 x=43
    x[front_idx] = front_x
    y[front_idx] = 0.0
    yaw[front_idx] = 0.0
    vel_x[front_idx] = 14.0
    vel_y[front_idx] = 0.0

    log_trajectory = types.SimpleNamespace(
        x=x, y=y, yaw=yaw, vel_x=vel_x, vel_y=vel_y, length=length,
        width=np.full((num_objects, num_frames), 2.0), valid=valid,
    )
    object_metadata = types.SimpleNamespace(
        ids=np.array([1, 2]), object_types=object_types,
        is_sdc=np.array([True, False]),
    )
    state = types.SimpleNamespace(
        log_trajectory=log_trajectory, object_metadata=object_metadata,
        roadgraph_points=roadgraph_points,
    )
    scene_key = "time_separated_shard.tfrecord#0"
    record = types.SimpleNamespace(
        record_index=0, source_shard="time_separated_shard.tfrecord",
        source_dataset="WOMD", source_split="validation", scene_key=scene_key,
        state=state, num_objects=num_objects, sdc_index=ego_idx, sdc_id=1,
        valid_trajectory_length=num_frames,
        roadgraph_point_count=source_xy.shape[0] + target_xy.shape[0],
    )
    return record


def _find_all_transitions(record):
    log_trajectory = record.state.log_trajectory
    sdc_index = record.sdc_index
    ego_x = np.asarray(log_trajectory.x[sdc_index])
    ego_y = np.asarray(log_trajectory.y[sdc_index])
    ego_yaw = np.asarray(log_trajectory.yaw[sdc_index])
    ego_valid = np.asarray(log_trajectory.valid[sdc_index]).astype(bool)
    polylines = extract_lane_polylines(record.state.roadgraph_points)
    raw_assignments = assign_ego_lane_sequence(
        ego_x, ego_y, ego_yaw, ego_valid, polylines, _LANE_ASSIGNMENT_CONFIG
    )
    stable_sequence = compute_stable_lane_sequence(
        raw_assignments,
        persistence_frames=_LANE_ASSIGNMENT_CONFIG.persistence_frames,
        max_ambiguous_gap_frames=_LANE_ASSIGNMENT_CONFIG.max_ambiguous_gap_frames,
    )
    return find_lane_transitions(
        stable_sequence,
        max_bridge_gap_frames=_LANE_ASSIGNMENT_CONFIG.max_ambiguous_gap_frames,
    )


@pytest.fixture(scope="module")
def time_separated_scene():
    return _build_time_separated_merge_scene()


@pytest.fixture(scope="module")
def time_separated_setup(time_separated_scene):
    from src.scenarios.lane_geometry import project_point_to_polyline

    transitions = _find_all_transitions(time_separated_scene)
    assert len(transitions) == 1
    transition = transitions[0]
    assert transition.transition_frame == _TIME_SEPARATED_TRANSITION_FRAME, (
        f"Expected transition_frame == {_TIME_SEPARATED_TRANSITION_FRAME}, "
        f"got {transition.transition_frame}"
    )

    polylines = extract_lane_polylines(time_separated_scene.state.roadgraph_points)
    lane_by_id = {p.lane_id: p for p in polylines}
    source_polyline = lane_by_id[transition.source_lane_id]
    target_polyline = lane_by_id[transition.target_lane_id]

    log_trajectory = time_separated_scene.state.log_trajectory
    sdc_index = time_separated_scene.sdc_index
    ego_x = np.asarray(log_trajectory.x[sdc_index])
    ego_y = np.asarray(log_trajectory.y[sdc_index])
    frame = transition.transition_frame
    projection = project_point_to_polyline(
        source_polyline, float(ego_x[frame]), float(ego_y[frame])
    )
    ego_source_arc_length = projection["arc_length_m"]

    diagnostic = detect_merge(
        transition, source_polyline, target_polyline, ego_source_arc_length,
        _MERGE_TOPOLOGY_CONFIG,
    )
    assert diagnostic.decision == MergeDecision.ACCEPT

    return transition, source_polyline, target_polyline, ego_source_arc_length


def test_feature_reference_frame_time_separation_regression(
    time_separated_scene, time_separated_setup
):
    """Mandatory A23 regression: merge_start_frame=10, transition_frame=24,
    with deliberately different ego speed (5 vs 15 m/s) and front gap
    (~20m vs ~2m) at the two frames. Materialized features must reflect
    frame 10's values, not frame 20's -- proving the fix actually
    samples at feature_reference_frame and not transition_frame.
    """
    transition, source_polyline, target_polyline, ego_source_arc_length = (
        time_separated_setup
    )

    features = materialize_merge_features(
        transition, source_polyline, target_polyline, ego_source_arc_length,
        time_separated_scene, _MERGE_TOPOLOGY_CONFIG, _AGENT_SELECTION_CONFIG,
    )

    assert features["feature_reference_valid"] is True
    assert features["feature_reference_frame"] == _TIME_SEPARATED_MERGE_START_FRAME
    assert features["feature_reference_policy"] == "merge_start_frame"
    assert features["feature_reference_reason"] is None
    assert features["merge_start_frame"] == _TIME_SEPARATED_MERGE_START_FRAME
    assert transition.transition_frame == _TIME_SEPARATED_TRANSITION_FRAME  # transition_frame itself untouched

    # Frame 10's ego speed (5 m/s), not frame 20's (15 m/s).
    assert features["ego_longitudinal_speed_mps"] == pytest.approx(5.0, abs=0.5)

    # Frame 10's front gap (~20 m), not frame 20's (~2 m).
    assert features["front_gap_m"] > 10.0


def test_feature_reference_frame_equals_merge_start_frame_when_valid(
    time_separated_scene, time_separated_setup
):
    transition, source_polyline, target_polyline, ego_source_arc_length = (
        time_separated_setup
    )
    features = materialize_merge_features(
        transition, source_polyline, target_polyline, ego_source_arc_length,
        time_separated_scene, _MERGE_TOPOLOGY_CONFIG, _AGENT_SELECTION_CONFIG,
    )
    assert features["feature_reference_frame"] == features["merge_start_frame"]


def test_transition_frame_field_unchanged(time_separated_setup):
    transition = time_separated_setup[0]
    assert transition.transition_frame == _TIME_SEPARATED_TRANSITION_FRAME


def test_detector_arc_length_input_still_uses_transition_frame(
    time_separated_scene, time_separated_setup
):
    """The detector's own source arc-length input (fed into detect_merge)
    must still be computed at transition_frame -- unchanged by this fix.
    """
    from src.scenarios.lane_geometry import project_point_to_polyline

    transition, source_polyline, target_polyline, ego_source_arc_length = (
        time_separated_setup
    )
    log_trajectory = time_separated_scene.state.log_trajectory
    ego_x = np.asarray(log_trajectory.x[time_separated_scene.sdc_index])
    ego_y = np.asarray(log_trajectory.y[time_separated_scene.sdc_index])

    expected = project_point_to_polyline(
        source_polyline,
        float(ego_x[transition.transition_frame]),
        float(ego_y[transition.transition_frame]),
    )["arc_length_m"]

    assert ego_source_arc_length == pytest.approx(expected)


def test_missing_merge_start_frame_no_silent_fallback():
    from src.scenarios.lane_assignment import LaneTransition

    transition = LaneTransition(
        source_lane_id=1, target_lane_id=2, transition_frame=20,
        source_start_frame=0, source_end_frame=19,
        target_start_frame=20, target_end_frame=29,
    )
    ego_valid = np.ones(40, dtype=bool)
    frame, valid, reason = _resolve_feature_reference_frame(
        transition, None, ego_valid
    )
    assert frame is None
    assert valid is False
    assert reason == "merge_start_frame_unavailable"


def test_invalid_ego_frame_no_silent_fallback():
    from src.scenarios.lane_assignment import LaneTransition

    transition = LaneTransition(
        source_lane_id=1, target_lane_id=2, transition_frame=20,
        source_start_frame=0, source_end_frame=19,
        target_start_frame=20, target_end_frame=29,
    )
    ego_valid = np.ones(40, dtype=bool)
    ego_valid[10] = False  # merge_start_frame candidate is invalid
    frame, valid, reason = _resolve_feature_reference_frame(
        transition, 10, ego_valid
    )
    assert frame is None
    assert valid is False
    assert reason == "merge_start_frame_invalid_ego_frame"


def test_merge_start_frame_after_transition_frame_no_silent_fallback():
    from src.scenarios.lane_assignment import LaneTransition

    transition = LaneTransition(
        source_lane_id=1, target_lane_id=2, transition_frame=20,
        source_start_frame=0, source_end_frame=19,
        target_start_frame=20, target_end_frame=29,
    )
    ego_valid = np.ones(40, dtype=bool)
    frame, valid, reason = _resolve_feature_reference_frame(
        transition, 25, ego_valid  # after transition_frame
    )
    assert frame is None
    assert valid is False
    assert reason == "merge_start_frame_after_transition_frame"


def test_invalid_reference_leaves_feature_fields_blank(time_separated_scene):
    """When the reference is invalid (no valid ego frame ever reaches
    merge_start_s, so merge_start_frame comes back None), decision
    -state feature fields must be None (blank), not silently computed
    at transition_frame.
    """
    import copy
    import types as _types

    from src.scenarios.lane_geometry import extract_lane_polylines

    transitions = _find_all_transitions(time_separated_scene)
    transition = transitions[0]

    polylines = extract_lane_polylines(time_separated_scene.state.roadgraph_points)
    lane_by_id = {p.lane_id: p for p in polylines}
    source_polyline = lane_by_id[transition.source_lane_id]
    target_polyline = lane_by_id[transition.target_lane_id]

    # Force merge_start_frame -> None by making ego invalid at every
    # frame _derive_merge_frames would otherwise scan (source_start
    # through target_end) -- no valid frame ever reaches merge_start_s,
    # so _resolve_feature_reference_frame must return
    # (None, False, "merge_start_frame_unavailable"), not fall back to
    # transition_frame.
    all_invalid_ego_valid = time_separated_scene.state.log_trajectory.valid.copy()
    all_invalid_ego_valid[time_separated_scene.sdc_index, :] = False
    tampered_log_trajectory = _types.SimpleNamespace(
        **{
            **vars(time_separated_scene.state.log_trajectory),
            "valid": all_invalid_ego_valid,
        }
    )
    tampered_state = _types.SimpleNamespace(
        **{**vars(time_separated_scene.state), "log_trajectory": tampered_log_trajectory}
    )
    tampered_scene = _types.SimpleNamespace(
        **{**vars(time_separated_scene), "state": tampered_state}
    )

    features = materialize_merge_features(
        transition, source_polyline, target_polyline, None,
        tampered_scene, _MERGE_TOPOLOGY_CONFIG, _AGENT_SELECTION_CONFIG,
    )

    assert features["feature_reference_valid"] is False
    assert features["feature_reference_frame"] is None
    assert features["feature_reference_reason"] == "merge_start_frame_unavailable"
    assert features["merge_start_frame"] is None
    assert features["ego_longitudinal_speed_mps"] is None
    assert features["merge_distance_m"] is None
    assert features["front_vehicle_id"] is None
    assert features["front_ttc_s"] is None
    assert features["traffic_density"] is None


def test_accept_path_matches_direct_materialize_call(
    time_separated_scene, time_separated_setup
):
    """The detector-ACCEPT dataset-scan path (build_candidate_records)
    must produce identical feature-reference values to a direct
    materialize_merge_features call -- single source of truth.
    """
    features_direct = materialize_merge_features(
        *time_separated_setup, time_separated_scene,
        _MERGE_TOPOLOGY_CONFIG, _AGENT_SELECTION_CONFIG,
    )
    records, error_info = build_candidate_records(
        time_separated_scene, _LANE_ASSIGNMENT_CONFIG, _MERGE_TOPOLOGY_CONFIG,
        _AGENT_SELECTION_CONFIG,
    )
    assert error_info is None
    accept_records = [r for r in records if r.decision == "accept"]
    assert len(accept_records) == 1
    record = accept_records[0]

    assert record.feature_reference_frame == features_direct["feature_reference_frame"]
    assert record.feature_reference_valid == features_direct["feature_reference_valid"]
    assert record.ego_longitudinal_speed_mps == pytest.approx(
        features_direct["ego_longitudinal_speed_mps"]
    )


def test_csv_round_trip_preserves_new_fields(tmp_path, time_separated_scene, time_separated_setup):
    records, error_info = build_candidate_records(
        time_separated_scene, _LANE_ASSIGNMENT_CONFIG, _MERGE_TOPOLOGY_CONFIG,
        _AGENT_SELECTION_CONFIG,
    )
    assert error_info is None
    output_path = tmp_path / "candidates.csv"
    write_candidates_csv(records, output_path)

    with open(output_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    accept_rows = [r for r in rows if r["decision"] == "accept"]
    assert len(accept_rows) == 1
    row = accept_rows[0]
    assert row["feature_reference_frame"] == str(_TIME_SEPARATED_MERGE_START_FRAME)
    assert row["feature_reference_policy"] == "merge_start_frame"
    assert row["feature_reference_valid"] == "True"
    assert row["feature_reference_reason"] == ""
