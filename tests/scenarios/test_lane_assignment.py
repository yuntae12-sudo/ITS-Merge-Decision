"""Tests for src/scenarios/lane_assignment.py using synthetic geometry.

These do not depend on any WOMD tf_example fixture -- lanes are built
directly as ``LanePolyline`` instances (bypassing roadgraph point
reconstruction, which is already covered by test_lane_geometry.py) so
assignment, temporal filtering, and transition detection can be
verified deterministically.
"""

import numpy as np
import pytest

from src.scenarios.lane_assignment import (
    LaneAssignment,
    LaneAssignmentConfig,
    assign_lane,
    compute_stable_lane_sequence,
    find_lane_transitions,
)
from src.scenarios.lane_geometry import (
    LanePolyline,
    compute_arc_length,
    nearest_lane_candidates,
)


def _make_straight_lane(lane_id: int, y: float, x_range=(0.0, 100.0)):
    """A straight lane along +x at constant y, direction (1, 0)."""

    xs = np.linspace(x_range[0], x_range[1], 50)
    xy = np.stack([xs, np.full_like(xs, y)], axis=1)
    direction = np.tile([1.0, 0.0], (xy.shape[0], 1))

    return LanePolyline(
        lane_id=lane_id,
        lane_type=2,
        xy=xy,
        direction=direction,
        arc_length=compute_arc_length(xy),
    )


DEFAULT_CONFIG = LaneAssignmentConfig(
    max_lateral_distance_m=2.0,
    max_heading_difference_deg=45.0,
    persistence_frames=3,
)


# ======================================================================
# assign_lane: candidate rejection and scoring
# ======================================================================


def test_assign_lane_picks_close_matching_lane():

    lane_a = _make_straight_lane(1, y=0.0)
    lane_b = _make_straight_lane(2, y=20.0)

    result = assign_lane(
        x=10.0, y=0.3, yaw=0.0,
        polylines=[lane_a, lane_b],
        config=DEFAULT_CONFIG,
        frame_index=0,
    )

    assert result.lane_id == 1
    assert result.lateral_distance_m == pytest.approx(0.3, abs=1e-6)
    assert result.valid is True


def test_assign_lane_rejects_out_of_range_lateral_distance():

    lane_a = _make_straight_lane(1, y=0.0)

    result = assign_lane(
        x=10.0, y=5.0, yaw=0.0,  # 5 m > max_lateral_distance_m=2.0
        polylines=[lane_a],
        config=DEFAULT_CONFIG,
        frame_index=0,
    )

    assert result.lane_id is None
    assert result.valid is True


def _make_vertical_lane(lane_id: int, x: float, y_range=(-50.0, 50.0)):
    """A straight lane along +y at constant x, direction (0, 1) --
    simulates a crossing lane whose *geometry* (not just a direction
    field) runs perpendicular to a +x-traveling ego, since
    project_point_to_polyline derives heading from segment vectors
    (xy), not from the direction field.
    """

    ys = np.linspace(y_range[0], y_range[1], 50)
    xy = np.stack([np.full_like(ys, x), ys], axis=1)
    direction = np.tile([0.0, 1.0], (xy.shape[0], 1))

    return LanePolyline(
        lane_id=lane_id,
        lane_type=2,
        xy=xy,
        direction=direction,
        arc_length=compute_arc_length(xy),
    )


def test_assign_lane_rejects_heading_mismatch_despite_proximity():
    """A lane that is physically closer but ~90 degrees off heading
    (e.g. a crossing lane at an intersection) must be rejected in favor
    of a farther lane whose heading actually matches ego -- this is the
    real-world failure mode observed at WOMD intersections during
    threshold selection (see configs/phase1_merge.yaml).
    """

    close_crossing_lane = _make_vertical_lane(1, x=10.0)
    farther_matching_lane = _make_straight_lane(2, y=1.8)

    result = assign_lane(
        x=10.0, y=0.1, yaw=0.0,
        polylines=[close_crossing_lane, farther_matching_lane],
        config=DEFAULT_CONFIG,
        frame_index=0,
    )

    assert result.lane_id == 2


def test_assign_lane_no_candidates_returns_none():

    result = assign_lane(
        x=0.0, y=0.0, yaw=0.0,
        polylines=[],
        config=DEFAULT_CONFIG,
        frame_index=0,
    )

    assert result.lane_id is None
    assert result.valid is True


def test_assign_lane_segment_projection_beats_vertex_distance():
    """Constructs a case where nearest_lane_candidates' cheap *sampled
    vertex* distance would rank lane A above lane B, but the exact
    segment-projected geometry (project_point_to_polyline) shows lane B
    is the true ego lane -- proving assign_lane must use segment
    projection for the final decision rather than stopping at vertex
    preselection (project plan Commit C section 4 / test case 7).

    Lane A is a short, steeply-angled (near-vertical) segment whose one
    endpoint sits very close to the query point: by raw vertex
    distance alone A looks like the best candidate (closer than any
    vertex of B). But that endpoint is a clamped segment projection
    with a heading ~90 degrees from ego, and B -- a sparse, longer,
    ego-heading-aligned lane whose *vertices* are individually farther
    away -- has the smaller true segment-projected lateral distance
    and a matching heading. Vertex-distance preselection alone cannot
    tell these apart; only project_point_to_polyline's exact geometry
    can.
    """

    lane_a_xy = np.array([[10.0, 0.5], [10.0, 20.0]])
    lane_a = LanePolyline(
        lane_id=1,
        lane_type=2,
        xy=lane_a_xy,
        direction=np.array([[0.0, 1.0]] * 2),
        arc_length=compute_arc_length(lane_a_xy),
    )

    lane_b_xs = np.array([0.0, 8.0, 12.0, 20.0])  # sparse: no vertex at x=10
    lane_b_xy = np.stack(
        [lane_b_xs, np.full_like(lane_b_xs, 0.3)], axis=1
    )
    lane_b = LanePolyline(
        lane_id=2,
        lane_type=2,
        xy=lane_b_xy,
        direction=np.tile([1.0, 0.0], (4, 1)),
        arc_length=compute_arc_length(lane_b_xy),
    )

    config = LaneAssignmentConfig(
        max_lateral_distance_m=6.0,
        max_heading_difference_deg=45.0,
        persistence_frames=3,
    )

    # Sanity-check the premise: vertex-distance preselection alone
    # would rank lane A (endpoint 0.15 m away) above lane B (nearest
    # vertex 2.0 m away).
    ranked = nearest_lane_candidates([lane_a, lane_b], x=10.0, y=0.35, k=2)
    assert ranked[0][0] == 1
    assert ranked[1][0] == 2

    result = assign_lane(
        x=10.0, y=0.35, yaw=0.0,
        polylines=[lane_a, lane_b],
        config=config,
        frame_index=0,
    )

    # Despite lane A ranking first on raw vertex distance, its segment
    # heading (~90 deg from ego) fails max_heading_difference_deg, so
    # the final decision correctly picks lane B.
    assert result.lane_id == 2


# ======================================================================
# compute_stable_lane_sequence: persistence / hysteresis
# ======================================================================


def _assignments_from_lane_ids(lane_ids):
    return [
        LaneAssignment(
            frame_index=i,
            lane_id=lane_id,
            lateral_distance_m=0.0 if lane_id is not None else None,
            heading_difference_rad=0.0 if lane_id is not None else None,
            arc_length_m=0.0 if lane_id is not None else None,
            score=0.0 if lane_id is not None else None,
            valid=True,
        )
        for i, lane_id in enumerate(lane_ids)
    ]


def test_stable_sequence_case1_stable_single_lane():

    raw = ["A"] * 6
    assignments = _assignments_from_lane_ids(raw)

    stable = compute_stable_lane_sequence(assignments, persistence_frames=3)
    transitions = find_lane_transitions(stable)

    assert stable == ["A"] * 6
    assert transitions == []


def test_stable_sequence_case2_stable_transition():

    raw = ["A", "A", "A", "A", "B", "B", "B", "B", "B"]
    assignments = _assignments_from_lane_ids(raw)

    stable = compute_stable_lane_sequence(assignments, persistence_frames=3)
    transitions = find_lane_transitions(stable)

    assert len(transitions) == 1
    assert transitions[0].source_lane_id == "A"
    assert transitions[0].target_lane_id == "B"


def test_stable_sequence_case3_one_frame_jitter_rejected():

    raw = ["A", "A", "A", "B", "A", "A", "A"]
    assignments = _assignments_from_lane_ids(raw)

    stable = compute_stable_lane_sequence(assignments, persistence_frames=3)
    transitions = find_lane_transitions(stable)

    assert transitions == []
    assert stable == ["A"] * 7


def test_stable_sequence_case4_two_short_oscillations_rejected():

    raw = ["A", "A", "B", "A", "B", "A", "A"]
    assignments = _assignments_from_lane_ids(raw)

    stable = compute_stable_lane_sequence(assignments, persistence_frames=3)
    transitions = find_lane_transitions(stable)

    assert transitions == []
    assert stable == ["A"] * 7


def test_stable_sequence_case5_stable_transition_after_noise():

    raw = ["A", "A", "A", "B", "A", "B", "B", "B", "B"]
    assignments = _assignments_from_lane_ids(raw)

    stable = compute_stable_lane_sequence(assignments, persistence_frames=3)
    transitions = find_lane_transitions(stable)

    assert len(transitions) == 1
    assert transitions[0].source_lane_id == "A"
    assert transitions[0].target_lane_id == "B"
    # raw: A A A B A B B B B (indices 0-8). The B at index 3 starts a
    # pending run of 1, but index 4 (A) matches the still-current
    # stable lane A and resets the pending count. B then restarts a
    # fresh pending run at index 5 and reaches persistence_frames=3
    # consecutive B's at index 7, where the stable switch lands.
    assert transitions[0].transition_frame == 7


def test_stable_sequence_none_frames_bridge_short_gap():

    raw = ["A", "A", "A", None, None, "A", "A"]
    assignments = _assignments_from_lane_ids(raw)

    stable = compute_stable_lane_sequence(
        assignments, persistence_frames=3, max_ambiguous_gap_frames=5
    )

    assert stable == ["A", "A", "A", "A", "A", "A", "A"]


def test_stable_sequence_none_frames_drop_after_long_gap():

    raw = ["A", "A", "A"] + [None] * 10 + ["A", "A"]
    assignments = _assignments_from_lane_ids(raw)

    stable = compute_stable_lane_sequence(
        assignments, persistence_frames=3, max_ambiguous_gap_frames=3
    )

    # After 3 valid A frames, the ambiguous gap exceeds the configured
    # limit (3), so the stable lane must drop to None rather than
    # bridging an unsupported 10-frame gap. It should not silently
    # re-establish as "A" without a fresh persistent run -- exactly 3
    # A-frames only satisfies persistence_frames=3 starting at the
    # first of them, so it re-establishes on the last frame.
    assert stable[:3] == ["A", "A", "A"]
    assert None in stable[4:12]
    assert stable[-1] == "A"


def test_stable_sequence_none_at_start_has_no_stable_lane():

    raw = [None, None, "A", "A", "A", "A"]
    assignments = _assignments_from_lane_ids(raw)

    stable = compute_stable_lane_sequence(assignments, persistence_frames=3)

    assert stable[0] is None
    assert stable[1] is None
    # Lane A needs persistence_frames=3 consecutive raw frames before
    # becoming stable, so it should not appear as stable until index 4.
    assert stable[4] == "A"
    assert stable[5] == "A"


# ======================================================================
# find_lane_transitions
# ======================================================================


def test_find_lane_transitions_no_transition_for_all_none():

    stable = [None, None, None]

    assert find_lane_transitions(stable) == []


def test_find_lane_transitions_multiple_transitions():

    stable = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]

    transitions = find_lane_transitions(stable)

    assert len(transitions) == 2
    assert (transitions[0].source_lane_id, transitions[0].target_lane_id) == (
        "A",
        "B",
    )
    assert (transitions[1].source_lane_id, transitions[1].target_lane_id) == (
        "B",
        "C",
    )
    assert transitions[0].source_start_frame == 0
    assert transitions[0].source_end_frame == 2
    assert transitions[0].target_start_frame == 3
    assert transitions[1].target_start_frame == 6


def test_find_lane_transitions_rejects_transition_across_long_gap():
    """A -> None -> B must NOT be automatically confirmed as an A -> B
    transition just because B happens to follow eventually -- a long
    ambiguous gap is not evidence the two lanes are related.
    """

    stable = ["A", "A", "A"] + [None] * 10 + ["B", "B", "B"]

    unbounded = find_lane_transitions(stable)
    bounded = find_lane_transitions(stable, max_bridge_gap_frames=3)

    assert len(unbounded) == 1  # default: any gap length is bridged
    assert bounded == []  # explicit limit correctly rejects it


def test_find_lane_transitions_accepts_transition_across_short_gap():

    stable = ["A", "A", "A", None, None, "B", "B", "B"]

    transitions = find_lane_transitions(stable, max_bridge_gap_frames=2)

    assert len(transitions) == 1
    assert transitions[0].source_lane_id == "A"
    assert transitions[0].target_lane_id == "B"

    # A tighter limit than the actual 2-frame gap must reject it.
    assert find_lane_transitions(stable, max_bridge_gap_frames=1) == []
