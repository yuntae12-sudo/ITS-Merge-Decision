import types

import numpy as np

from src.scenarios.merge_v2 import (
    InteractionEvidence,
    ManeuverType,
    REJECT_CUT_IN_NOT_TOPOLOGY_MERGE,
    REJECT_DIVERGE,
    REJECT_GAP_ORDER_NOT_PERSISTENT,
    REJECT_MAP_ARTIFACT,
    REJECT_NO_INTERACTION,
    REJECT_SERIAL_CONTINUATION,
    REVIEW_AMBIGUOUS_TOPOLOGY,
    TopologyEvidence,
    V2Decision,
    classify_v2_merge,
)
from src.scenarios.scenario_proto_loader import iter_scenario_protobufs, scenario_to_topology
import pytest


def interaction(**overrides):
    values = dict(
        decision_start_frame=10,
        commit_frame=20,
        completion_frame=35,
        source_occupancy_frames=10,
        target_stable_frames=8,
        longitudinal_progress_m=20.0,
        lateral_displacement_m=3.5,
        front_vehicle_id=11,
        rear_vehicle_id=None,
    )
    values.update(overrides)
    return InteractionEvidence(**values)


def topology(**overrides):
    values = dict(
        source_lane_id=1,
        target_lane_id=2,
        source_exit_lane_ids=(2,),
        target_entry_lane_ids=(1, 3),
        source_exits_to_target=True,
        terminal_heading_difference_deg=10.0,
        terminal_collinear_offset_m=4.0,
        endpoint_gap_m=1.0,
    )
    values.update(overrides)
    return TopologyEvidence(**values)


def test_authoritative_multi_entry_topological_merge_accepted():
    result = classify_v2_merge(topology(), interaction())
    assert result.decision == V2Decision.ACCEPT
    assert result.maneuver_type == ManeuverType.TOPOLOGICAL_MERGE
    assert result.canonical


def test_man0013_single_collinear_path_rejected_even_with_1_to_5m_gap():
    result = classify_v2_merge(
        topology(
            target_entry_lane_ids=(1,),
            terminal_heading_difference_deg=0.066,
            terminal_collinear_offset_m=0.95,
            endpoint_gap_m=1.85,
        ),
        interaction(lateral_displacement_m=0.4),
    )
    assert result.decision == V2Decision.REJECT
    assert result.reason == REJECT_SERIAL_CONTINUATION


def test_interactive_parallel_cut_in_rejected_when_not_topology_merge():
    """A purely lateral cut-in with no map-topology convergence (no
    entry_lanes/exit_lanes link, not a roundabout) is a lane
    change/cut-in, not a MERGE, regardless of interaction strength."""

    result = classify_v2_merge(
        topology(
            source_exit_lane_ids=(), target_entry_lane_ids=(),
            source_exits_to_target=False, lanes_overlap_longitudinally=True,
        ),
        interaction(lateral_displacement_m=3.2),
    )
    assert result.decision == V2Decision.REJECT
    assert result.reason == REJECT_CUT_IN_NOT_TOPOLOGY_MERGE


def test_diverge_rejected_not_merge():
    result = classify_v2_merge(
        topology(source_exit_lane_count=2, target_entry_lane_ids=(1,)),
        interaction(),
    )
    assert result.decision == V2Decision.REJECT
    assert result.reason == REJECT_DIVERGE


def test_map_artifact_target_rejected():
    result = classify_v2_merge(
        topology(target_polyline_point_count=1),
        interaction(),
    )
    assert result.decision == V2Decision.REJECT
    assert result.reason == REJECT_MAP_ARTIFACT


def test_ambiguous_topology_without_multi_entry_or_roundabout_is_reviewed():
    result = classify_v2_merge(
        topology(target_entry_lane_ids=(1,), endpoint_gap_m=6.0),
        interaction(),
    )
    assert result.decision == V2Decision.REVIEW
    assert result.reason == REVIEW_AMBIGUOUS_TOPOLOGY


def test_roundabout_entry_requires_interaction_and_accepts_conflict_vehicle():
    no_vehicle = classify_v2_merge(
        topology(target_is_roundabout=True),
        interaction(front_vehicle_id=None),
    )
    assert no_vehicle.reason == REJECT_NO_INTERACTION

    accepted = classify_v2_merge(
        topology(target_is_roundabout=True),
        interaction(front_vehicle_id=None, conflict_vehicle_ids=(99,)),
    )
    assert accepted.decision == V2Decision.ACCEPT
    assert accepted.maneuver_type == ManeuverType.ROUNDABOUT_ENTRY


def test_two_sided_gap_must_persist():
    result = classify_v2_merge(
        topology(),
        interaction(
            rear_vehicle_id=12,
            gap_order_valid_at_entry=True,
            gap_order_persistence_frames=4,
            required_gap_order_persistence_frames=5,
        ),
    )
    assert result.reason == REJECT_GAP_ORDER_NOT_PERSISTENT


def test_proto_loader_preserves_entry_exit_topology():
    def feature(feature_id, entries, exits):
        lane = types.SimpleNamespace(
            entry_lanes=entries,
            exit_lanes=exits,
            polyline=[types.SimpleNamespace(x=0.0, y=0.0), types.SimpleNamespace(x=1.0, y=0.0)],
        )
        return types.SimpleNamespace(
            id=feature_id,
            lane=lane,
            HasField=lambda name: name == "lane",
        )

    scenario = types.SimpleNamespace(
        scenario_id="scenario-1",
        map_features=[feature(1, [], [2]), feature(2, [1, 3], [])],
    )
    loaded = scenario_to_topology(scenario)
    evidence = loaded.evidence(
        1, 2,
        terminal_heading_difference_deg=10.0,
        terminal_collinear_offset_m=4.0,
        endpoint_gap_m=1.0,
    )
    assert evidence.source_exits_to_target
    assert evidence.target_has_multiple_entries
    assert np.array_equal(loaded.lanes[1].polyline_xy, np.array([[0.0, 0.0], [1.0, 0.0]]))


def test_tfexample_cannot_masquerade_as_topology_source():
    with pytest.raises(ValueError, match="do not contain entry_lanes"):
        next(iter_scenario_protobufs(["training_tfexample.tfrecord-00000-of-01000"]))


def _proto_topology(features_by_id):
    """``{lane_id: (entries, exits, left_neighbors, right_neighbors, n_polyline_pts)}``
    -> a duck-typed ``ScenarioTopology`` exercising the real
    ``scenario_to_topology`` parse path (not hand-built dataclasses),
    so ``evidence()``'s auto-computed topology signals are covered
    end-to-end from parsed lanes."""

    def feature(feature_id, entries, exits, left_nb, right_nb, n_pts):
        lane = types.SimpleNamespace(
            entry_lanes=entries,
            exit_lanes=exits,
            left_neighbors=[types.SimpleNamespace(feature_id=n) for n in left_nb],
            right_neighbors=[types.SimpleNamespace(feature_id=n) for n in right_nb],
            polyline=[types.SimpleNamespace(x=float(i), y=0.0) for i in range(n_pts)],
        )
        return types.SimpleNamespace(
            id=feature_id, lane=lane, HasField=lambda name: name == "lane"
        )

    scenario = types.SimpleNamespace(
        scenario_id="scenario-evidence",
        map_features=[
            feature(lane_id, *spec) for lane_id, spec in features_by_id.items()
        ],
    )
    return scenario_to_topology(scenario)


def test_evidence_auto_flags_lane_change_from_neighbor_without_topology_link():
    topology = _proto_topology({
        1: ([], [], [], [2], 2),
        2: ([], [], [1], [], 2),
    })
    evidence = topology.evidence(1, 2, terminal_heading_difference_deg=0.0)
    assert evidence.is_lane_change_target
    assert not evidence.source_exits_to_target


def test_evidence_does_not_flag_lane_change_when_topology_connected():
    # Neighbors AND a real entry_lanes/exit_lanes link (e.g. a lane
    # that both borders and topologically merges into its neighbor):
    # topology connectivity wins, never treated as a plain lane change.
    topology = _proto_topology({
        1: ([], [2], [], [2], 2),
        2: ([1, 3], [], [1], [], 2),
    })
    evidence = topology.evidence(1, 2, terminal_heading_difference_deg=5.0)
    assert not evidence.is_lane_change_target
    assert evidence.source_exits_to_target


def test_evidence_auto_flags_intersection_from_sharp_heading_single_entry():
    topology = _proto_topology({
        1: ([], [2], [], [], 2),
        2: ([1], [], [], [], 2),
    })
    evidence = topology.evidence(1, 2, terminal_heading_difference_deg=75.0)
    assert evidence.is_intersection_transition


def test_evidence_does_not_flag_intersection_for_multi_entry_merge():
    topology = _proto_topology({
        1: ([], [2], [], [], 2),
        2: ([1, 3], [], [], [], 2),
    })
    evidence = topology.evidence(1, 2, terminal_heading_difference_deg=75.0)
    assert not evidence.is_intersection_transition


def test_evidence_auto_computes_exit_count_and_polyline_point_count():
    topology = _proto_topology({
        1: ([], [2, 4], [], [], 3),
        2: ([1], [], [], [], 1),
    })
    evidence = topology.evidence(1, 2, terminal_heading_difference_deg=0.0)
    assert evidence.source_exit_lane_count == 2
    assert evidence.target_polyline_point_count == 1
    assert evidence.is_map_artifact


def test_iter_scenario_protobufs_preserves_gcs_uri_double_slash():
    """Regression: ``pathlib.Path("gs://bucket/x")`` silently collapses
    to ``gs:/bucket/x`` (single slash), which TensorFlow's GCS
    filesystem then reports as NotFoundError -- this broke every
    remote-shard read while never surfacing on a local path. The path
    must reach ``tf.data.TFRecordDataset`` as the exact string given."""

    import src.scenarios.scenario_proto_loader as loader

    captured = []

    class _FakeDataset:
        def __init__(self, path):
            captured.append(path)

        def __iter__(self):
            return iter(())

    original = loader.tf if hasattr(loader, "tf") else None
    import tensorflow as tf

    real_ctor = tf.data.TFRecordDataset
    tf.data.TFRecordDataset = _FakeDataset
    try:
        gcs_uri = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/training/training.tfrecord-00500-of-01000"
        list(loader.iter_scenario_protobufs([gcs_uri]))
    finally:
        tf.data.TFRecordDataset = real_ctor

    assert captured == [gcs_uri]
