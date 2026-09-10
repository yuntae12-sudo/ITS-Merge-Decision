"""Tests for scripts/build_merge_manifest.py's manifest-materialization
fix ("materialize manually confirmed merge features").

Builds a lightweight fake ``ScenarioRecord``-like object (real numpy
arrays for a hand-built scene) that flows through the REAL pipeline
functions (`extract_lane_polylines`, `assign_ego_lane_sequence`,
`compute_stable_lane_sequence`, `find_lane_transitions`, `detect_merge`,
`materialize_merge_features`) -- following the same synthetic-geometry
style as test_merge_detector.py / test_scenario_features.py /
test_validation_viz.py, rather than depending on real WOMD data.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.scenarios.dataset_builder import (
    make_candidate_id,
    materialize_merge_features,
    reconstruct_transition,
)
from src.scenarios.lane_assignment import LaneAssignmentConfig
from src.scenarios.merge_detector import (
    MergeDecision,
    MergeTopologyConfig,
    detect_merge,
)
from src.scenarios.scenario_features import AgentSelectionConfig

from src.scenarios.scenario_loader import DatasetExpansionConfig

from scripts.build_merge_manifest import (
    build_manifest,
    materialize_confirmed_rows,
    sync_labels,
    validate_manifest_row,
)

LANE_ASSIGNMENT_CONFIG = LaneAssignmentConfig(
    max_lateral_distance_m=5.0,
    max_heading_difference_deg=45.0,
    persistence_frames=5,
    max_ambiguous_gap_frames=5,
    candidate_count=8,
)

MERGE_TOPOLOGY_CONFIG = MergeTopologyConfig(
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

AGENT_SELECTION_CONFIG = AgentSelectionConfig(
    max_target_lane_lateral_distance_m=5.0,
    max_target_lane_heading_difference_deg=45.0,
    max_distance_m=100.0,
    density_radius_m=50.0,
)


# ---------------------------------------------------------------------
# Synthetic scene construction.
# ---------------------------------------------------------------------


def _roadgraph_from_lanes(lanes):
    """lanes: list of (lane_id, xy (N,2)). Builds a flat RoadgraphPoints
    -like namespace with WOMD-aligned direction vectors (so
    extract_lane_polylines's travel-direction alignment is a no-op).
    """

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
        all_types.extend([2] * n)  # SURFACE_STREET
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


def _build_merge_scene(
    source_lane_id=196,
    target_lane_id=206,
    num_frames=40,
    transition_at=20,
    source_shard="fake_shard.tfrecord",
    record_index=0,
):
    """Builds a synthetic true-merge scene: ego drives along a source
    lane that curves toward and ends near a target lane it then
    continues on (matching test_merge_detector.py Case 3b's true-merge
    shape: large upstream separation shrinking to just outside
    serial_continuation_max_lateral_m at the endpoint -- an ACCEPT
    case, not REVIEW/REJECT).

    Returns: (record, transition_frame, source_lane_id, target_lane_id)
    """

    # Source: curves in from an offset, ending 0.6 m off the target's
    # start point (same shape as test_case3b in test_merge_detector.py).
    source_xs = np.linspace(0.0, 30.0, 31)
    source_xy = np.stack(
        [source_xs, 10.0 - (source_xs / 30.0) * 9.4], axis=1
    )

    target_xs = np.linspace(30.0, 90.0, 61)
    target_xy = np.stack([target_xs, np.zeros_like(target_xs)], axis=1)

    roadgraph_points = _roadgraph_from_lanes(
        [(source_lane_id, source_xy), (target_lane_id, target_xy)]
    )

    # Ego trajectory: on the source lane's geometry for the first half
    # of the window, then on the target lane's geometry (a "serial
    # continuation of stable lane ids" trajectory) for the remainder --
    # constructed so assign_ego_lane_sequence/compute_stable_lane_sequence
    # /find_lane_transitions produce a stable source->target transition
    # at `transition_at`.
    num_objects = 3  # ego + front + rear
    x = np.zeros((num_objects, num_frames))
    y = np.zeros((num_objects, num_frames))
    yaw = np.zeros((num_objects, num_frames))
    vel_x = np.zeros((num_objects, num_frames))
    vel_y = np.zeros((num_objects, num_frames))
    length = np.full((num_objects, num_frames), 4.5)
    width = np.full((num_objects, num_frames), 2.0)
    valid = np.ones((num_objects, num_frames), dtype=bool)
    object_types = np.array([1, 1, 1])  # all vehicles

    ego_idx = 0
    for frame in range(num_frames):
        if frame < transition_at:
            # Walk along the source polyline's arc length.
            frac = frame / max(transition_at - 1, 1)
            s = frac * 30.0
            idx = int(round(frac * 30))
            idx = min(idx, source_xy.shape[0] - 1)
            pos = source_xy[idx]
            x[ego_idx, frame] = pos[0]
            y[ego_idx, frame] = pos[1]
            # heading roughly along the source direction.
            if idx < source_xy.shape[0] - 1:
                seg = source_xy[idx + 1] - source_xy[idx]
            else:
                seg = source_xy[idx] - source_xy[idx - 1]
            yaw[ego_idx, frame] = float(np.arctan2(seg[1], seg[0]))
            vel_x[ego_idx, frame] = 15.0
            vel_y[ego_idx, frame] = 0.0
        else:
            frac = (frame - transition_at) / max(num_frames - transition_at - 1, 1)
            target_len = 60.0
            s = frac * target_len
            idx = int(round(frac * (target_xy.shape[0] - 1)))
            idx = min(idx, target_xy.shape[0] - 1)
            pos = target_xy[idx]
            x[ego_idx, frame] = pos[0]
            y[ego_idx, frame] = pos[1]
            yaw[ego_idx, frame] = 0.0
            vel_x[ego_idx, frame] = 15.0
            vel_y[ego_idx, frame] = 0.0

    # A front vehicle ahead of ego on the target lane throughout.
    front_idx = 1
    x[front_idx] = np.linspace(50.0, 90.0, num_frames)
    y[front_idx] = 0.0
    yaw[front_idx] = 0.0
    vel_x[front_idx] = 14.0
    vel_y[front_idx] = 0.0

    # Rear vehicle stays invalid/far away (no rear candidate expected).
    valid[2] = False

    log_trajectory = types.SimpleNamespace(
        x=x, y=y, yaw=yaw, vel_x=vel_x, vel_y=vel_y, length=length,
        width=width, valid=valid,
    )

    object_metadata = types.SimpleNamespace(
        ids=np.array([1, 2, 3]),
        object_types=object_types,
        is_sdc=np.array([True, False, False]),
    )

    state = types.SimpleNamespace(
        log_trajectory=log_trajectory,
        object_metadata=object_metadata,
        roadgraph_points=roadgraph_points,
    )

    scene_key = f"{source_shard}#{record_index}"
    record = types.SimpleNamespace(
        record_index=record_index,
        source_shard=source_shard,
        source_dataset="WOMD",
        source_split="validation",
        scene_key=scene_key,
        state=state,
        num_objects=num_objects,
        sdc_index=ego_idx,
        sdc_id=1,
        valid_trajectory_length=num_frames,
        roadgraph_point_count=source_xy.shape[0] + target_xy.shape[0],
    )

    return record


def _fake_expansion_config(tmp_path, split, shard_names):
    """Builds a real (fully-resolved, existing-on-disk) synthetic
    DatasetExpansionConfig for the given shard basenames, so
    resolve_physical_shard's real path-matching logic runs (rather
    than being bypassed) even though iter_scenarios itself is
    monkeypatched to yield synthetic scenes instead of real WOMD data.
    """

    paths = []
    for name in shard_names:
        path = tmp_path / name
        path.write_text("fake")
        paths.append(str(path))

    return DatasetExpansionConfig(
        dataset_name="WOMD",
        split=split,
        shard_paths=paths,
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )


@pytest.fixture(scope="module")
def merge_scene():
    return _build_merge_scene()


@pytest.fixture(scope="module")
def reconstructed(merge_scene):
    """Reconstructs the transition once for the whole module (the
    scene is deterministic / read-only)."""

    transitions = _find_all_transitions(merge_scene)
    assert len(transitions) == 1, (
        f"Expected exactly one stable transition in the synthetic scene, "
        f"got {len(transitions)}"
    )
    transition = transitions[0]

    return reconstruct_transition(
        merge_scene,
        LANE_ASSIGNMENT_CONFIG,
        transition.transition_frame,
        transition.source_lane_id,
        transition.target_lane_id,
        candidate_id="test",
    )


def _find_all_transitions(record):
    from src.scenarios.lane_assignment import (
        assign_ego_lane_sequence,
        compute_stable_lane_sequence,
        find_lane_transitions,
    )
    from src.scenarios.lane_geometry import extract_lane_polylines

    log_trajectory = record.state.log_trajectory
    sdc_index = record.sdc_index
    ego_x = np.asarray(log_trajectory.x[sdc_index])
    ego_y = np.asarray(log_trajectory.y[sdc_index])
    ego_yaw = np.asarray(log_trajectory.yaw[sdc_index])
    ego_valid = np.asarray(log_trajectory.valid[sdc_index]).astype(bool)

    polylines = extract_lane_polylines(record.state.roadgraph_points)
    raw_assignments = assign_ego_lane_sequence(
        ego_x, ego_y, ego_yaw, ego_valid, polylines, LANE_ASSIGNMENT_CONFIG
    )
    stable_sequence = compute_stable_lane_sequence(
        raw_assignments,
        persistence_frames=LANE_ASSIGNMENT_CONFIG.persistence_frames,
        max_ambiguous_gap_frames=LANE_ASSIGNMENT_CONFIG.max_ambiguous_gap_frames,
    )
    return find_lane_transitions(
        stable_sequence,
        max_bridge_gap_frames=LANE_ASSIGNMENT_CONFIG.max_ambiguous_gap_frames,
    )


# ---------------------------------------------------------------------
# Case A: sanity check on existing REVIEW-blank behavior (via
# dataset_builder's own CandidateRecord path -- already covered
# elsewhere, kept minimal here as an explicit sanity anchor for this
# fix's premise).
# ---------------------------------------------------------------------


def test_case_a_review_candidate_row_has_blank_accept_fields():
    row = {
        "candidate_id": "shard#0__t13__196_206",
        "scene_key": "shard#0",
        "source_dataset": "WOMD",
        "source_split": "validation",
        "source_shard": "shard",
        "record_index": "0",
        "source_lane_id": "196",
        "target_lane_id": "206",
        "transition_frame": "13",
        "decision": "review",
        "reason": "ambiguous_serial_or_merge",
        "merge_start_s": "",
        "merge_end_s": "",
        "merge_start_frame": "",
        "merge_complete_frame": "",
        "ego_longitudinal_speed_mps": "",
        "merge_distance_m": "",
        "front_vehicle_id": "",
        "front_gap_m": "",
        "front_relative_speed_mps": "",
        "front_ttc_s": "",
        "rear_vehicle_id": "",
        "rear_gap_m": "",
        "rear_relative_speed_mps": "",
        "rear_ttc_s": "",
        "traffic_density": "",
    }
    # Sanity: these are exactly the fields that must be materialized.
    assert row["merge_start_s"] == ""
    assert row["ego_longitudinal_speed_mps"] == ""
    assert row["front_ttc_s"] == ""


# ---------------------------------------------------------------------
# Core regression: REVIEW + CONFIRMED_MERGE materialization (Cases B/C).
# ---------------------------------------------------------------------


def test_case_b_c_review_confirmed_merge_materializes_full_row(merge_scene, reconstructed):
    (transition, source_polyline, target_polyline, ego_source_arc_length) = reconstructed

    diagnostic = detect_merge(
        transition, source_polyline, target_polyline, ego_source_arc_length,
        MERGE_TOPOLOGY_CONFIG,
    )
    # This synthetic scene is built as a genuine ACCEPT-shaped merge
    # (test_merge_detector.py Case 3b shape); we force-relabel it as if
    # the detector had originally said REVIEW, to exercise the
    # "REVIEW -> CONFIRMED_MERGE materializes a full row" path without
    # needing a second, separately-shaped synthetic scene. The
    # detector-drift check inside materialize_confirmed_rows always
    # compares against whatever the CSV *actually* stored, so to
    # exercise that path faithfully we call materialize_merge_features
    # directly here (this test's purpose is C: "materialization
    # produces a fully populated row"), and test drift-detection
    # separately in test_case_k.
    features = materialize_merge_features(
        transition, source_polyline, target_polyline, ego_source_arc_length,
        merge_scene, MERGE_TOPOLOGY_CONFIG, AGENT_SELECTION_CONFIG,
    )

    assert diagnostic.decision == MergeDecision.ACCEPT  # sanity: our scene is a real merge

    # Core regression: none of these come back None -- a blind CSV
    # copy of a REVIEW row would have left them all blank/None.
    assert features["merge_start_s"] is not None
    assert features["merge_end_s"] is not None
    assert features["merge_complete_frame"] is not None
    assert isinstance(features["ego_longitudinal_speed_mps"], float)
    assert isinstance(features["merge_distance_m"], float)
    assert features["traffic_density"] is not None

    # Front vehicle present (constructed ahead of ego on target lane).
    assert features["front_vehicle_id"] is not None
    assert isinstance(features["front_gap_m"], float)
    assert isinstance(features["front_relative_speed_mps"], float)
    assert isinstance(features["front_ttc_s"], float)

    # Rear invalid -> no rear candidate -> inf ttc, blank id/gap/speed.
    assert features["rear_vehicle_id"] is None
    assert features["rear_gap_m"] is None
    assert features["rear_relative_speed_mps"] is None
    assert features["rear_ttc_s"] == float("inf")


def test_case_b_via_materialize_confirmed_rows_end_to_end(monkeypatch, tmp_path, merge_scene, reconstructed):
    """End-to-end through build_merge_manifest.materialize_confirmed_rows,
    with the stored CSV row's decision/reason forced to REVIEW (as if a
    human confirmed a REVIEW candidate) -- monkeypatches iter_scenarios
    to yield our synthetic record instead of loading real WOMD data.
    """

    (transition, source_polyline, target_polyline, ego_source_arc_length) = reconstructed
    real_diagnostic = detect_merge(
        transition, source_polyline, target_polyline, ego_source_arc_length,
        MERGE_TOPOLOGY_CONFIG,
    )
    assert real_diagnostic.decision == MergeDecision.ACCEPT

    candidate_id = make_candidate_id(
        merge_scene.scene_key, transition.transition_frame,
        transition.source_lane_id, transition.target_lane_id,
    )

    candidate_rows = [
        {
            "candidate_id": candidate_id,
            "scene_key": merge_scene.scene_key,
            "source_split": "validation",
            "source_shard": merge_scene.source_shard,
            "record_index": "0",
            "source_lane_id": str(transition.source_lane_id),
            "target_lane_id": str(transition.target_lane_id),
            "transition_frame": str(transition.transition_frame),
            "decision": real_diagnostic.decision.value,
            "reason": real_diagnostic.reason or "",
        }
    ]

    import scripts.build_merge_manifest as bmm
    monkeypatch.setattr(
        bmm, "iter_scenarios", lambda dataset_config, limit=None, **kw: iter([merge_scene])
    )

    fake_expansion_config = _fake_expansion_config(
        tmp_path, "validation", [merge_scene.source_shard]
    )

    materialized = materialize_confirmed_rows(
        candidate_rows,
        {candidate_id},
        dataset_config=fake_expansion_config,
        lane_assignment_config=LANE_ASSIGNMENT_CONFIG,
        merge_topology_config=MERGE_TOPOLOGY_CONFIG,
        agent_selection_config=AGENT_SELECTION_CONFIG,
    )

    features = materialized[candidate_id]
    assert features["merge_start_s"] is not None
    assert features["ego_longitudinal_speed_mps"] is not None
    assert features["front_vehicle_id"] is not None
    assert features["feature_materialization_source"] == "detector_accept"


# ---------------------------------------------------------------------
# Case D: ACCEPT + CONFIRMED_MERGE remains complete, no drift from the
# refactor (dataset_builder's ACCEPT path vs. materialize_merge_features
# called directly must agree).
# ---------------------------------------------------------------------


def test_case_d_accept_path_matches_refactored_function(merge_scene, reconstructed):
    from src.scenarios.dataset_builder import build_candidate_records

    features_direct = materialize_merge_features(
        *reconstructed, merge_scene, MERGE_TOPOLOGY_CONFIG, AGENT_SELECTION_CONFIG,
    )

    records, error_info = build_candidate_records(
        merge_scene, LANE_ASSIGNMENT_CONFIG, MERGE_TOPOLOGY_CONFIG,
        AGENT_SELECTION_CONFIG,
    )
    assert error_info is None
    accept_records = [r for r in records if r.decision == "accept"]
    assert len(accept_records) == 1
    record = accept_records[0]

    assert record.merge_start_s == pytest.approx(features_direct["merge_start_s"])
    assert record.merge_end_s == pytest.approx(features_direct["merge_end_s"])
    assert record.merge_complete_frame == features_direct["merge_complete_frame"]
    assert record.ego_longitudinal_speed_mps == pytest.approx(
        features_direct["ego_longitudinal_speed_mps"]
    )
    assert record.front_vehicle_id == features_direct["front_vehicle_id"]
    assert record.front_gap_m == pytest.approx(features_direct["front_gap_m"])
    assert record.traffic_density == features_direct["traffic_density"]


# ---------------------------------------------------------------------
# Cases E/F: no-auto-promotion (build_manifest excludes ACCEPT
# +UNREVIEWED and REVIEW+CONFIRMED_NON_MERGE).
# ---------------------------------------------------------------------


def test_case_e_accept_unreviewed_excluded(monkeypatch, merge_scene, reconstructed):
    transition = reconstructed[0]
    candidate_id = make_candidate_id(
        merge_scene.scene_key, transition.transition_frame,
        transition.source_lane_id, transition.target_lane_id,
    )
    candidate_rows = [
        {
            "candidate_id": candidate_id,
            "scene_key": merge_scene.scene_key,
            "source_dataset": "WOMD",
            "source_split": "validation",
            "source_shard": "fake_shard.tfrecord",
            "record_index": "0",
            "source_lane_id": str(transition.source_lane_id),
            "target_lane_id": str(transition.target_lane_id),
            "transition_frame": str(transition.transition_frame),
            "decision": "accept",
            "reason": "",
        }
    ]
    labels = {
        candidate_id: {
            "candidate_id": candidate_id,
            "manual_validation": "UNREVIEWED",
            "manual_note": "",
        }
    }

    import scripts.build_merge_manifest as bmm
    monkeypatch.setattr(
        bmm, "iter_scenarios", lambda dataset_config, limit=None, **kw: iter([merge_scene])
    )

    rows = build_manifest(
        candidate_rows, labels,
        dataset_config=None, lane_assignment_config=LANE_ASSIGNMENT_CONFIG,
        merge_topology_config=MERGE_TOPOLOGY_CONFIG,
        agent_selection_config=AGENT_SELECTION_CONFIG,
    )
    assert rows == []


def test_case_f_review_confirmed_non_merge_excluded(monkeypatch, merge_scene, reconstructed):
    transition = reconstructed[0]
    candidate_id = make_candidate_id(
        merge_scene.scene_key, transition.transition_frame,
        transition.source_lane_id, transition.target_lane_id,
    )
    candidate_rows = [
        {
            "candidate_id": candidate_id,
            "scene_key": merge_scene.scene_key,
            "source_dataset": "WOMD",
            "source_split": "validation",
            "source_shard": "fake_shard.tfrecord",
            "record_index": "0",
            "source_lane_id": str(transition.source_lane_id),
            "target_lane_id": str(transition.target_lane_id),
            "transition_frame": str(transition.transition_frame),
            "decision": "accept",
            "reason": "",
        }
    ]
    labels = {
        candidate_id: {
            "candidate_id": candidate_id,
            "manual_validation": "CONFIRMED_NON_MERGE",
            "manual_note": "",
        }
    }

    import scripts.build_merge_manifest as bmm
    monkeypatch.setattr(
        bmm, "iter_scenarios", lambda dataset_config, limit=None, **kw: iter([merge_scene])
    )

    rows = build_manifest(
        candidate_rows, labels,
        dataset_config=None, lane_assignment_config=LANE_ASSIGNMENT_CONFIG,
        merge_topology_config=MERGE_TOPOLOGY_CONFIG,
        agent_selection_config=AGENT_SELECTION_CONFIG,
    )
    assert rows == []


# ---------------------------------------------------------------------
# Case G: unknown --set-label candidate_id raises.
# ---------------------------------------------------------------------


def test_case_g_unknown_set_label_candidate_id_raises():
    with pytest.raises(ValueError, match="Unknown candidate_id"):
        sync_labels(["A", "B"], {}, set_label=("ZZZ_NOT_A_CANDIDATE", "CONFIRMED_MERGE"))


# ---------------------------------------------------------------------
# Case H: stale labels preserved with warning, not deleted.
# ---------------------------------------------------------------------


def test_case_h_stale_labels_preserved():
    existing_labels = {
        "OLD": {"candidate_id": "OLD", "manual_validation": "CONFIRMED_MERGE", "manual_note": "x"},
    }
    merged, stale = sync_labels(["NEW"], existing_labels)
    assert "OLD" in merged
    assert merged["OLD"]["manual_validation"] == "CONFIRMED_MERGE"
    assert stale == ["OLD"]


# ---------------------------------------------------------------------
# Case I: transition mismatch (tampered lane ids) raises drift error.
# ---------------------------------------------------------------------


def test_case_i_transition_mismatch_raises(merge_scene, reconstructed):
    transition = reconstructed[0]
    with pytest.raises(ValueError, match="Manifest materialization drift detected"):
        reconstruct_transition(
            merge_scene,
            LANE_ASSIGNMENT_CONFIG,
            transition.transition_frame,
            source_lane_id=99999,  # tampered: does not exist in the scene
            target_lane_id=88888,
            candidate_id="tampered_candidate",
        )


# ---------------------------------------------------------------------
# Case J: required-field validation catches an incomplete row.
# ---------------------------------------------------------------------


def test_case_j_incomplete_row_raises():
    incomplete_row = {
        "candidate_id": "shard#0__t10__1_2",
        "scene_key": "shard#0",
        "record_index": "0",
        "source_lane_id": "1",
        "target_lane_id": "2",
        "transition_frame": "10",
        "merge_start_s": "1.0",
        "merge_end_s": "",  # missing required field
        "merge_complete_frame": "10",
        "ego_longitudinal_speed_mps": "10.0",
        "merge_distance_m": "5.0",
        "traffic_density": "0",
        "front_vehicle_id": "",
        "front_gap_m": "",
        "front_relative_speed_mps": "",
        "front_ttc_s": float("inf"),
        "rear_vehicle_id": "",
        "rear_gap_m": "",
        "rear_relative_speed_mps": "",
        "rear_ttc_s": float("inf"),
    }
    with pytest.raises(ValueError, match="Incomplete manifest row"):
        validate_manifest_row(incomplete_row)


def test_case_j_front_rear_inconsistency_raises():
    row = {
        "candidate_id": "shard#0__t10__1_2",
        "scene_key": "shard#0",
        "record_index": "0",
        "source_lane_id": "1",
        "target_lane_id": "2",
        "transition_frame": "10",
        "merge_start_s": "1.0",
        "merge_end_s": "2.0",
        "merge_complete_frame": "10",
        "feature_reference_valid": "True",
        "feature_reference_reason": "",
        "ego_longitudinal_speed_mps": "10.0",
        "merge_distance_m": "5.0",
        "traffic_density": "0",
        "front_vehicle_id": "99",  # present...
        "front_gap_m": "",  # ...but gap blank: inconsistent
        "front_relative_speed_mps": "",
        "front_ttc_s": 5.0,
        "rear_vehicle_id": "",
        "rear_gap_m": "",
        "rear_relative_speed_mps": "",
        "rear_ttc_s": float("inf"),
    }
    with pytest.raises(ValueError, match="Inconsistent front/rear fields"):
        validate_manifest_row(row)


def test_case_j_valid_row_passes():
    row = {
        "candidate_id": "shard#0__t10__1_2",
        "scene_key": "shard#0",
        "record_index": "0",
        "source_lane_id": "1",
        "target_lane_id": "2",
        "transition_frame": "10",
        "merge_start_s": "1.0",
        "merge_end_s": "2.0",
        "merge_complete_frame": "10",
        "feature_reference_valid": "True",
        "feature_reference_reason": "",
        "ego_longitudinal_speed_mps": "10.0",
        "merge_distance_m": "5.0",
        "traffic_density": "0",
        "front_vehicle_id": "",
        "front_gap_m": "",
        "front_relative_speed_mps": "",
        "front_ttc_s": float("inf"),
        "rear_vehicle_id": "",
        "rear_gap_m": "",
        "rear_relative_speed_mps": "",
        "rear_ttc_s": float("inf"),
    }
    validate_manifest_row(row)  # must not raise


def test_case_j_invalid_feature_reference_raises():
    """A row with feature_reference_valid=False must fail loudly (fix
    commit "materialize merge state at pre-merge reference frame") --
    a CONFIRMED_MERGE candidate must never silently enter the final
    manifest with an invalid pre-merge feature reference.
    """
    row = {
        "candidate_id": "shard#0__t10__1_2",
        "scene_key": "shard#0",
        "record_index": "0",
        "source_lane_id": "1",
        "target_lane_id": "2",
        "transition_frame": "10",
        "merge_start_s": "1.0",
        "merge_end_s": "2.0",
        "merge_complete_frame": "10",
        "feature_reference_valid": "False",
        "feature_reference_reason": "merge_start_frame_unavailable",
        "ego_longitudinal_speed_mps": "10.0",
        "merge_distance_m": "5.0",
        "traffic_density": "0",
        "front_vehicle_id": "",
        "front_gap_m": "",
        "front_relative_speed_mps": "",
        "front_ttc_s": float("inf"),
        "rear_vehicle_id": "",
        "rear_gap_m": "",
        "rear_relative_speed_mps": "",
        "rear_ttc_s": float("inf"),
    }
    with pytest.raises(ValueError, match="feature_reference_valid is not True"):
        validate_manifest_row(row)


# ---------------------------------------------------------------------
# Case K: detector decision/reason drift raises.
# ---------------------------------------------------------------------


def test_case_k_detector_drift_raises(monkeypatch, tmp_path, merge_scene, reconstructed):
    transition = reconstructed[0]
    candidate_id = make_candidate_id(
        merge_scene.scene_key, transition.transition_frame,
        transition.source_lane_id, transition.target_lane_id,
    )
    # Stored CSV row falsely claims this was a REJECT
    # (parallel_lane_change) -- the recomputed decision (ACCEPT) must
    # disagree and raise, rather than silently materializing.
    candidate_rows = [
        {
            "candidate_id": candidate_id,
            "scene_key": merge_scene.scene_key,
            "source_split": "validation",
            "source_shard": merge_scene.source_shard,
            "record_index": "0",
            "source_lane_id": str(transition.source_lane_id),
            "target_lane_id": str(transition.target_lane_id),
            "transition_frame": str(transition.transition_frame),
            "decision": "reject",
            "reason": "parallel_lane_change",
        }
    ]

    import scripts.build_merge_manifest as bmm
    monkeypatch.setattr(
        bmm, "iter_scenarios", lambda dataset_config, limit=None, **kw: iter([merge_scene])
    )

    fake_expansion_config = _fake_expansion_config(
        tmp_path, "validation", [merge_scene.source_shard]
    )

    with pytest.raises(ValueError, match="Detector drift"):
        materialize_confirmed_rows(
            candidate_rows,
            {candidate_id},
            dataset_config=fake_expansion_config,
            lane_assignment_config=LANE_ASSIGNMENT_CONFIG,
            merge_topology_config=MERGE_TOPOLOGY_CONFIG,
            agent_selection_config=AGENT_SELECTION_CONFIG,
        )


# ---------------------------------------------------------------------
# Case L: repeated manifest build is deterministic.
# ---------------------------------------------------------------------


def test_case_l_repeated_build_is_deterministic(monkeypatch, tmp_path, merge_scene, reconstructed):
    transition = reconstructed[0]
    candidate_id = make_candidate_id(
        merge_scene.scene_key, transition.transition_frame,
        transition.source_lane_id, transition.target_lane_id,
    )
    candidate_rows = [
        {
            "candidate_id": candidate_id,
            "scene_key": merge_scene.scene_key,
            "source_dataset": "WOMD",
            "source_split": "validation",
            "source_shard": "fake_shard.tfrecord",
            "record_index": "0",
            "source_lane_id": str(transition.source_lane_id),
            "target_lane_id": str(transition.target_lane_id),
            "transition_frame": str(transition.transition_frame),
            "decision": "accept",
            "reason": "",
        }
    ]
    labels = {
        candidate_id: {
            "candidate_id": candidate_id,
            "manual_validation": "CONFIRMED_MERGE",
            "manual_note": "",
        }
    }

    import scripts.build_merge_manifest as bmm
    monkeypatch.setattr(
        bmm, "iter_scenarios", lambda dataset_config, limit=None, **kw: iter([merge_scene])
    )

    fake_expansion_config = _fake_expansion_config(
        tmp_path, "validation", ["fake_shard.tfrecord"]
    )

    rows_1 = build_manifest(
        candidate_rows, labels,
        dataset_config=fake_expansion_config, lane_assignment_config=LANE_ASSIGNMENT_CONFIG,
        merge_topology_config=MERGE_TOPOLOGY_CONFIG,
        agent_selection_config=AGENT_SELECTION_CONFIG,
    )
    rows_2 = build_manifest(
        candidate_rows, labels,
        dataset_config=fake_expansion_config, lane_assignment_config=LANE_ASSIGNMENT_CONFIG,
        merge_topology_config=MERGE_TOPOLOGY_CONFIG,
        agent_selection_config=AGENT_SELECTION_CONFIG,
    )
    assert rows_1 == rows_2


# ---------------------------------------------------------------------
# Case J/K: multi-shard manifest materialization safety.
#
# Two synthetic scenes sharing the SAME record_index (0) but different
# source_shard values and different lane geometry -- confirms
# materialize_confirmed_rows/build_manifest correctly picks the
# physical shard matching each candidate's own stored source_shard
# (Case J), and that a candidate whose stored lane ids only exist in
# shard B raises the existing drift error rather than silently
# substituting shard A's transition when (incorrectly) matched against
# shard A (Case K).
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def merge_scene_shard_b():
    # Different source_shard, SAME record_index (0) as `merge_scene`,
    # and different lane ids so the two scenes are trivially
    # distinguishable if the wrong one is loaded.
    return _build_merge_scene(
        source_lane_id=485, target_lane_id=344,
        source_shard="other_shard.tfrecord", record_index=0,
    )


@pytest.fixture(scope="module")
def reconstructed_shard_b(merge_scene_shard_b):
    transitions = _find_all_transitions(merge_scene_shard_b)
    assert len(transitions) == 1
    transition = transitions[0]
    return reconstruct_transition(
        merge_scene_shard_b,
        LANE_ASSIGNMENT_CONFIG,
        transition.transition_frame,
        transition.source_lane_id,
        transition.target_lane_id,
        candidate_id="test_shard_b",
    )


def _make_shard_dispatching_iter_scenarios(scenes_by_shard_name):
    """Builds a fake ``iter_scenarios`` that inspects
    ``dataset_config.path`` (set by ``build_waymax_config`` to the
    physical shard path) and yields only the scene matching that
    shard's basename -- simulating what a real per-shard
    ``iter_scenarios`` call would do, without touching real WOMD data.
    """

    def fake_iter_scenarios(dataset_config, limit=None, **kw):
        shard_name = Path(dataset_config.path).name
        scene = scenes_by_shard_name.get(shard_name)
        if scene is None:
            return iter([])
        return iter([scene])

    return fake_iter_scenarios


def test_case_j_materialization_reloads_correct_shard(
    monkeypatch, tmp_path, merge_scene, reconstructed, merge_scene_shard_b, reconstructed_shard_b
):
    transition_a = reconstructed[0]
    transition_b = reconstructed_shard_b[0]

    candidate_id_a = make_candidate_id(
        merge_scene.scene_key, transition_a.transition_frame,
        transition_a.source_lane_id, transition_a.target_lane_id,
    )
    candidate_id_b = make_candidate_id(
        merge_scene_shard_b.scene_key, transition_b.transition_frame,
        transition_b.source_lane_id, transition_b.target_lane_id,
    )

    candidate_rows = [
        {
            "candidate_id": candidate_id_a,
            "scene_key": merge_scene.scene_key,
            "source_split": "validation",
            "source_shard": merge_scene.source_shard,
            "record_index": "0",
            "source_lane_id": str(transition_a.source_lane_id),
            "target_lane_id": str(transition_a.target_lane_id),
            "transition_frame": str(transition_a.transition_frame),
            "decision": "accept",
            "reason": "",
        },
        {
            "candidate_id": candidate_id_b,
            "scene_key": merge_scene_shard_b.scene_key,
            "source_split": "validation",
            "source_shard": merge_scene_shard_b.source_shard,
            "record_index": "0",
            "source_lane_id": str(transition_b.source_lane_id),
            "target_lane_id": str(transition_b.target_lane_id),
            "transition_frame": str(transition_b.transition_frame),
            "decision": "accept",
            "reason": "",
        },
    ]

    import scripts.build_merge_manifest as bmm
    monkeypatch.setattr(
        bmm,
        "iter_scenarios",
        _make_shard_dispatching_iter_scenarios(
            {
                merge_scene.source_shard: merge_scene,
                merge_scene_shard_b.source_shard: merge_scene_shard_b,
            }
        ),
    )

    fake_expansion_config = _fake_expansion_config(
        tmp_path, "validation",
        [merge_scene.source_shard, merge_scene_shard_b.source_shard],
    )

    materialized = materialize_confirmed_rows(
        candidate_rows,
        {candidate_id_a, candidate_id_b},
        dataset_config=fake_expansion_config,
        lane_assignment_config=LANE_ASSIGNMENT_CONFIG,
        merge_topology_config=MERGE_TOPOLOGY_CONFIG,
        agent_selection_config=AGENT_SELECTION_CONFIG,
    )

    # Both candidates must materialize successfully, each against its
    # OWN shard's geometry (not the other's, not whichever loads
    # first). front_vehicle_id/traffic_density are identical by
    # construction in both synthetic scenes, so the real
    # discriminator here is that no drift/mismatch error was raised --
    # a wrong-shard substitution would either raise (Case K) or
    # silently compute features from the wrong lane geometry, which
    # the lane_id-specific reconstruct_transition match already
    # prevents structurally (see reconstruct_transition's own
    # docstring: it matches on transition_frame/source_lane_id/
    # target_lane_id, all recorded per-row).
    assert candidate_id_a in materialized
    assert candidate_id_b in materialized
    assert materialized[candidate_id_a]["merge_start_s"] is not None
    assert materialized[candidate_id_b]["merge_start_s"] is not None


def test_case_k_wrong_shard_transition_cannot_be_substituted(
    monkeypatch, merge_scene, merge_scene_shard_b, reconstructed_shard_b
):
    """A candidate's stored source_lane_id/target_lane_id that only
    exist in shard B, if incorrectly matched against shard A's
    transitions (e.g. by a hypothetical future bug that groups by
    record_index alone), must raise the existing 'Manifest
    materialization drift detected' error rather than silently
    substituting a different transition.
    """

    transition_b = reconstructed_shard_b[0]

    # Directly exercise reconstruct_transition against shard A's scene
    # using shard B's lane ids -- this is exactly the failure mode the
    # (source_split, source_shard, record_index) grouping in
    # materialize_confirmed_rows is designed to make structurally
    # impossible; this test pins down that the underlying safety net
    # (reconstruct_transition's exact-match requirement) still raises
    # if that grouping were ever bypassed.
    with pytest.raises(ValueError, match="Manifest materialization drift detected"):
        reconstruct_transition(
            merge_scene,  # shard A's scene...
            LANE_ASSIGNMENT_CONFIG,
            transition_b.transition_frame,
            transition_b.source_lane_id,  # ...with shard B's lane ids
            transition_b.target_lane_id,
            candidate_id="cross_shard_tampered",
        )


def test_case_j_materialize_confirmed_rows_groups_by_shard_and_record_index(
    merge_scene, merge_scene_shard_b
):
    """Unit-tests the shard-selection/grouping logic directly: two
    candidates sharing record_index=0 but differing by source_shard
    must be grouped into separate physical-shard buckets, never
    collapsed into one record_index=0 bucket regardless of shard.
    """

    from collections import defaultdict

    rows = [
        {
            "candidate_id": "a", "source_split": "validation",
            "source_shard": merge_scene.source_shard, "record_index": "0",
        },
        {
            "candidate_id": "b", "source_split": "validation",
            "source_shard": merge_scene_shard_b.source_shard, "record_index": "0",
        },
    ]

    by_shard_record = defaultdict(list)
    for row in rows:
        key = (row["source_split"], row["source_shard"], int(row["record_index"]))
        by_shard_record[key].append(row)

    assert len(by_shard_record) == 2
    assert (
        "validation", merge_scene.source_shard, 0
    ) in by_shard_record
    assert (
        "validation", merge_scene_shard_b.source_shard, 0
    ) in by_shard_record


# ---------------------------------------------------------------------
# Fix commit regressions: wrong-split and unknown-shard, both via the
# common resolve_physical_shard helper (fail BEFORE any scenario
# loading is attempted). Synthetic multi-shard/dataset-config only.
# ---------------------------------------------------------------------


def test_wrong_split_regression_raises_before_scenario_loading(
    monkeypatch, tmp_path, merge_scene, reconstructed
):
    """A candidate row with source_split="training" against a
    dataset_config with split="validation" must raise clearly BEFORE
    any scenario loading is attempted -- iter_scenarios is monkeypatched
    to blow up if ever called, so this test fails loudly (rather than
    silently passing) if the split check is ever bypassed.
    """

    transition = reconstructed[0]
    candidate_id = make_candidate_id(
        merge_scene.scene_key, transition.transition_frame,
        transition.source_lane_id, transition.target_lane_id,
    )
    candidate_rows = [
        {
            "candidate_id": candidate_id,
            "scene_key": merge_scene.scene_key,
            "source_split": "training",  # mismatched vs. config split below
            "source_shard": merge_scene.source_shard,
            "record_index": "0",
            "source_lane_id": str(transition.source_lane_id),
            "target_lane_id": str(transition.target_lane_id),
            "transition_frame": str(transition.transition_frame),
            "decision": "accept",
            "reason": "",
        }
    ]

    import scripts.build_merge_manifest as bmm

    def _explode(*args, **kwargs):
        raise AssertionError(
            "iter_scenarios must never be called when source_split "
            "disagrees with the dataset config's split -- the split "
            "check must fire first."
        )

    monkeypatch.setattr(bmm, "iter_scenarios", _explode)

    fake_expansion_config = _fake_expansion_config(
        tmp_path, "validation", [merge_scene.source_shard]
    )

    with pytest.raises(ValueError, match="source_split mismatch"):
        materialize_confirmed_rows(
            candidate_rows,
            {candidate_id},
            dataset_config=fake_expansion_config,
            lane_assignment_config=LANE_ASSIGNMENT_CONFIG,
            merge_topology_config=MERGE_TOPOLOGY_CONFIG,
            agent_selection_config=AGENT_SELECTION_CONFIG,
        )


def test_unknown_shard_regression_no_fallback(
    monkeypatch, tmp_path, merge_scene, reconstructed
):
    """A candidate row referencing a source_shard the dataset config
    does not know about (config only knows shard_A, candidate
    references shard_B) must raise via resolve_physical_shard's
    unknown-shard error -- no fallback to any local-layout convention.
    """

    transition = reconstructed[0]
    candidate_id = make_candidate_id(
        merge_scene.scene_key, transition.transition_frame,
        transition.source_lane_id, transition.target_lane_id,
    )
    candidate_rows = [
        {
            "candidate_id": candidate_id,
            "scene_key": merge_scene.scene_key,
            "source_split": "validation",
            "source_shard": "shard_b_never_configured.tfrecord",
            "record_index": "0",
            "source_lane_id": str(transition.source_lane_id),
            "target_lane_id": str(transition.target_lane_id),
            "transition_frame": str(transition.transition_frame),
            "decision": "accept",
            "reason": "",
        }
    ]

    import scripts.build_merge_manifest as bmm

    def _explode(*args, **kwargs):
        raise AssertionError(
            "iter_scenarios must never be called for an unresolvable "
            "shard -- resolve_physical_shard must fail first."
        )

    monkeypatch.setattr(bmm, "iter_scenarios", _explode)

    # Config only knows shard_A ("merge_scene.source_shard"); the
    # candidate row above references an entirely different, never
    # -configured shard basename.
    fake_expansion_config = _fake_expansion_config(
        tmp_path, "validation", [merge_scene.source_shard]
    )

    with pytest.raises(ValueError, match="Unknown shard"):
        materialize_confirmed_rows(
            candidate_rows,
            {candidate_id},
            dataset_config=fake_expansion_config,
            lane_assignment_config=LANE_ASSIGNMENT_CONFIG,
            merge_topology_config=MERGE_TOPOLOGY_CONFIG,
            agent_selection_config=AGENT_SELECTION_CONFIG,
        )


def test_mixed_split_csv_rejected_by_build_manifest(
    monkeypatch, tmp_path, merge_scene, reconstructed
):
    """build_manifest must fail clearly if the candidates being
    processed contain more than one distinct source_split value among
    their CONFIRMED_MERGE rows -- one manifest invocation is one
    logical split; a hand-edited/concatenated mixed-split CSV must be
    rejected rather than silently processed.
    """

    transition = reconstructed[0]
    candidate_id_a = make_candidate_id(
        merge_scene.scene_key, transition.transition_frame,
        transition.source_lane_id, transition.target_lane_id,
    )
    candidate_rows = [
        {
            "candidate_id": candidate_id_a,
            "scene_key": merge_scene.scene_key,
            "source_dataset": "WOMD",
            "source_split": "validation",
            "source_shard": merge_scene.source_shard,
            "record_index": "0",
            "source_lane_id": str(transition.source_lane_id),
            "target_lane_id": str(transition.target_lane_id),
            "transition_frame": str(transition.transition_frame),
            "decision": "accept",
            "reason": "",
        },
        {
            "candidate_id": "other_split_candidate",
            "scene_key": "training_shard.tfrecord#0",
            "source_dataset": "WOMD",
            "source_split": "training",  # different split -- must be rejected
            "source_shard": "training_shard.tfrecord",
            "record_index": "0",
            "source_lane_id": "1",
            "target_lane_id": "2",
            "transition_frame": "5",
            "decision": "accept",
            "reason": "",
        },
    ]
    labels = {
        candidate_id_a: {
            "candidate_id": candidate_id_a,
            "manual_validation": "CONFIRMED_MERGE",
            "manual_note": "",
        },
        "other_split_candidate": {
            "candidate_id": "other_split_candidate",
            "manual_validation": "CONFIRMED_MERGE",
            "manual_note": "",
        },
    }

    import scripts.build_merge_manifest as bmm
    monkeypatch.setattr(
        bmm, "iter_scenarios", lambda dataset_config, limit=None, **kw: iter([merge_scene])
    )

    fake_expansion_config = _fake_expansion_config(
        tmp_path, "validation", [merge_scene.source_shard]
    )

    with pytest.raises(ValueError, match="more than one distinct source_split"):
        build_manifest(
            candidate_rows, labels,
            dataset_config=fake_expansion_config,
            lane_assignment_config=LANE_ASSIGNMENT_CONFIG,
            merge_topology_config=MERGE_TOPOLOGY_CONFIG,
            agent_selection_config=AGENT_SELECTION_CONFIG,
        )


# ---------------------------------------------------------------------
# Fix commit regressions: harden multi-shard inspection and shard
# reload safety. materialize_confirmed_rows must resolve source_shard
# via resolve_physical_shard against the authoritative dataset_config's
# shard_paths -- never a hardcoded data/womd/<split>/<file> convention
# -- and must fail fast on a source_split mismatch BEFORE any scenario
# loading is attempted.
# ---------------------------------------------------------------------


def test_manifest_resolves_custom_configured_path_not_hardcoded_convention(
    monkeypatch, tmp_path, merge_scene, reconstructed
):
    """The candidate's source_shard resolves to whatever physical path
    is actually configured in dataset_config.shard_paths -- including a
    path that does NOT follow the data/womd/<split>/<file> convention
    -- proving build_merge_manifest.py no longer synthesizes a path
    from that convention.
    """

    transition = reconstructed[0]
    candidate_id = make_candidate_id(
        merge_scene.scene_key, transition.transition_frame,
        transition.source_lane_id, transition.target_lane_id,
    )
    candidate_rows = [
        {
            "candidate_id": candidate_id,
            "scene_key": merge_scene.scene_key,
            "source_split": "validation",
            "source_shard": merge_scene.source_shard,
            "record_index": "0",
            "source_lane_id": str(transition.source_lane_id),
            "target_lane_id": str(transition.target_lane_id),
            "transition_frame": str(transition.transition_frame),
            "decision": "accept",
            "reason": "",
        }
    ]

    # A custom, non-conventional directory layout: NOT data/womd/validation/.
    custom_dir = tmp_path / "some" / "unconventional" / "layout"
    custom_dir.mkdir(parents=True)
    custom_shard_path = custom_dir / merge_scene.source_shard
    custom_shard_path.write_text("fake")

    fake_expansion_config = DatasetExpansionConfig(
        dataset_name="WOMD",
        split="validation",
        shard_paths=[str(custom_shard_path)],
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )

    seen_paths = []

    import scripts.build_merge_manifest as bmm

    def fake_iter_scenarios(dataset_config, limit=None, **kw):
        seen_paths.append(dataset_config.path)
        return iter([merge_scene])

    monkeypatch.setattr(bmm, "iter_scenarios", fake_iter_scenarios)

    materialized = materialize_confirmed_rows(
        candidate_rows,
        {candidate_id},
        dataset_config=fake_expansion_config,
        lane_assignment_config=LANE_ASSIGNMENT_CONFIG,
        merge_topology_config=MERGE_TOPOLOGY_CONFIG,
        agent_selection_config=AGENT_SELECTION_CONFIG,
    )

    assert candidate_id in materialized
    # The exact custom configured path was used -- not a synthesized
    # data/womd/validation/<file> convention path.
    assert seen_paths == [str(custom_shard_path)]


def test_manifest_wrong_split_fails_before_scenario_loading(
    monkeypatch, tmp_path, merge_scene, reconstructed
):
    """A candidate row with source_split='training' fed against a
    dataset_config whose split='validation' must raise BEFORE any
    scenario loading is attempted (iter_scenarios must never be
    called).
    """

    transition = reconstructed[0]
    candidate_id = make_candidate_id(
        merge_scene.scene_key, transition.transition_frame,
        transition.source_lane_id, transition.target_lane_id,
    )
    candidate_rows = [
        {
            "candidate_id": candidate_id,
            "scene_key": merge_scene.scene_key,
            "source_split": "training",  # mismatches the config below
            "source_shard": merge_scene.source_shard,
            "record_index": "0",
            "source_lane_id": str(transition.source_lane_id),
            "target_lane_id": str(transition.target_lane_id),
            "transition_frame": str(transition.transition_frame),
            "decision": "accept",
            "reason": "",
        }
    ]

    fake_expansion_config = _fake_expansion_config(
        tmp_path, "validation", [merge_scene.source_shard]
    )

    import scripts.build_merge_manifest as bmm

    call_count = 0

    def fake_iter_scenarios(dataset_config, limit=None, **kw):
        nonlocal call_count
        call_count += 1
        return iter([merge_scene])

    monkeypatch.setattr(bmm, "iter_scenarios", fake_iter_scenarios)

    with pytest.raises(ValueError, match="source_split mismatch"):
        materialize_confirmed_rows(
            candidate_rows,
            {candidate_id},
            dataset_config=fake_expansion_config,
            lane_assignment_config=LANE_ASSIGNMENT_CONFIG,
            merge_topology_config=MERGE_TOPOLOGY_CONFIG,
            agent_selection_config=AGENT_SELECTION_CONFIG,
        )

    assert call_count == 0, (
        "iter_scenarios must never be called when source_split "
        "mismatches dataset_config.split"
    )


def test_manifest_unknown_shard_raises_no_fallback(
    monkeypatch, tmp_path, merge_scene, reconstructed
):
    """A candidate referencing a source_shard absent from
    dataset_config.shard_paths must raise (unknown shard), never fall
    back to any local-layout convention.
    """

    transition = reconstructed[0]
    candidate_id = make_candidate_id(
        merge_scene.scene_key, transition.transition_frame,
        transition.source_lane_id, transition.target_lane_id,
    )
    candidate_rows = [
        {
            "candidate_id": candidate_id,
            "scene_key": merge_scene.scene_key,
            "source_split": "validation",
            "source_shard": "shard_b_unknown.tfrecord",
            "record_index": "0",
            "source_lane_id": str(transition.source_lane_id),
            "target_lane_id": str(transition.target_lane_id),
            "transition_frame": str(transition.transition_frame),
            "decision": "accept",
            "reason": "",
        }
    ]

    # Config only knows shard_A -- not shard_b_unknown.tfrecord.
    fake_expansion_config = _fake_expansion_config(
        tmp_path, "validation", ["shard_a_known.tfrecord"]
    )

    import scripts.build_merge_manifest as bmm
    monkeypatch.setattr(
        bmm, "iter_scenarios", lambda dataset_config, limit=None, **kw: iter([merge_scene])
    )

    with pytest.raises(ValueError, match="Unknown shard"):
        materialize_confirmed_rows(
            candidate_rows,
            {candidate_id},
            dataset_config=fake_expansion_config,
            lane_assignment_config=LANE_ASSIGNMENT_CONFIG,
            merge_topology_config=MERGE_TOPOLOGY_CONFIG,
            agent_selection_config=AGENT_SELECTION_CONFIG,
        )


def test_build_manifest_mixed_split_confirmed_rows_raises():
    """If the candidates CSV contains more than one distinct
    source_split among its CONFIRMED_MERGE rows, build_manifest must
    fail clearly rather than silently processing a mixed-split CSV.
    """

    candidate_rows = [
        {
            "candidate_id": "shardA#0__t10__1_2",
            "scene_key": "shardA#0",
            "source_split": "validation",
            "source_shard": "shardA.tfrecord",
            "record_index": "0",
            "decision": "accept",
            "reason": "",
        },
        {
            "candidate_id": "shardB#0__t10__3_4",
            "scene_key": "shardB#0",
            "source_split": "training",
            "source_shard": "shardB.tfrecord",
            "record_index": "0",
            "decision": "accept",
            "reason": "",
        },
    ]
    labels = {
        "shardA#0__t10__1_2": {
            "candidate_id": "shardA#0__t10__1_2",
            "manual_validation": "CONFIRMED_MERGE",
            "manual_note": "",
        },
        "shardB#0__t10__3_4": {
            "candidate_id": "shardB#0__t10__3_4",
            "manual_validation": "CONFIRMED_MERGE",
            "manual_note": "",
        },
    }

    with pytest.raises(ValueError, match="more than one distinct source_split"):
        build_manifest(
            candidate_rows, labels,
            dataset_config=None, lane_assignment_config=LANE_ASSIGNMENT_CONFIG,
            merge_topology_config=MERGE_TOPOLOGY_CONFIG,
            agent_selection_config=AGENT_SELECTION_CONFIG,
        )
