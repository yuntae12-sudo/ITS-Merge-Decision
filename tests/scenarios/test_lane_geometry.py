"""Tests for src/scenarios/lane_geometry.py using synthetic geometry.

These tests do not depend on any WOMD tf_example fixture -- they build
minimal ``RoadgraphPoints`` instances directly so lane reconstruction
logic can be verified deterministically and independent of dataset
availability.
"""

import numpy as np
import pytest
from waymax.datatypes.roadgraph import RoadgraphPoints

from src.scenarios.lane_geometry import (
    compute_arc_length,
    extract_lane_polylines,
    nearest_lane_candidates,
    project_point_to_polyline,
)


def _make_roadgraph_points(
    xy_by_id: dict, lane_type: int = 2
) -> RoadgraphPoints:
    """Builds a RoadgraphPoints from {lane_id: [(x, y), ...]} in the
    given per-id order (this order is what the test intentionally
    shuffles to prove reconstruction doesn't trust raw array order).
    """

    xs, ys, ids, types = [], [], [], []

    for lane_id, points in xy_by_id.items():
        for x, y in points:
            xs.append(x)
            ys.append(y)
            ids.append(lane_id)
            types.append(lane_type)

    num_points = len(xs)

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)

    # Direction vectors are derived from consecutive points within each
    # id's *intended* order (not the possibly-shuffled storage order),
    # matching how WOMD provides a genuine local heading per point.
    dir_x = np.zeros(num_points, dtype=np.float32)
    dir_y = np.zeros(num_points, dtype=np.float32)

    cursor = 0

    for _, points in xy_by_id.items():
        points_arr = np.asarray(points, dtype=np.float32)
        n = points_arr.shape[0]

        for local_i in range(n):
            if local_i < n - 1:
                dx = points_arr[local_i + 1, 0] - points_arr[local_i, 0]
                dy = points_arr[local_i + 1, 1] - points_arr[local_i, 1]
            else:
                dx = points_arr[local_i, 0] - points_arr[local_i - 1, 0]
                dy = points_arr[local_i, 1] - points_arr[local_i - 1, 1]

            norm = np.hypot(dx, dy) or 1.0
            dir_x[cursor] = dx / norm
            dir_y[cursor] = dy / norm
            cursor += 1

    return RoadgraphPoints(
        x=xs,
        y=ys,
        z=np.zeros(num_points, dtype=np.float32),
        dir_x=dir_x,
        dir_y=dir_y,
        dir_z=np.zeros(num_points, dtype=np.float32),
        types=np.asarray(types, dtype=np.int32),
        ids=np.asarray(ids, dtype=np.int32),
        valid=np.ones(num_points, dtype=bool),
    )


def test_compute_arc_length_straight_line():

    xy = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])

    arc_length = compute_arc_length(xy)

    np.testing.assert_allclose(arc_length, [0.0, 1.0, 3.0])


def test_compute_arc_length_empty():

    xy = np.zeros((0, 2))

    arc_length = compute_arc_length(xy)

    assert arc_length.shape == (0,)


def test_extract_lane_polylines_reconstructs_shuffled_order():
    """Points sharing an id are NOT stored in polyline order; the
    module must reconstruct the order via chaining, not trust the
    array order.
    """

    straight_line = [(float(i), 0.0) for i in range(10)]

    shuffled = straight_line.copy()
    rng = np.random.default_rng(seed=0)
    rng.shuffle(shuffled)

    roadgraph_points = _make_roadgraph_points({7: shuffled})

    polylines = extract_lane_polylines(roadgraph_points)

    assert len(polylines) == 1

    polyline = polylines[0]
    assert polyline.lane_id == 7

    # Reconstructed order must be spatially monotonic along x, in
    # either direction, even though storage order was shuffled.
    xs = polyline.xy[:, 0]
    is_ascending = np.all(np.diff(xs) > 0)
    is_descending = np.all(np.diff(xs) < 0)

    assert is_ascending or is_descending


def test_extract_lane_polylines_filters_non_lane_types():

    roadgraph_points = _make_roadgraph_points(
        {1: [(0.0, 0.0), (1.0, 0.0)]}, lane_type=17  # STOP_SIGN
    )

    polylines = extract_lane_polylines(roadgraph_points)

    assert polylines == []


def test_extract_lane_polylines_separates_ids():

    roadgraph_points = _make_roadgraph_points(
        {
            1: [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            2: [(0.0, 10.0), (1.0, 10.0)],
        }
    )

    polylines = extract_lane_polylines(roadgraph_points)

    lane_ids = sorted(polyline.lane_id for polyline in polylines)

    assert lane_ids == [1, 2]


def test_nearest_lane_candidates_orders_by_distance():

    roadgraph_points = _make_roadgraph_points(
        {
            1: [(0.0, 0.0), (10.0, 0.0)],  # close to query point
            2: [(0.0, 100.0), (10.0, 100.0)],  # far from query point
        }
    )

    polylines = extract_lane_polylines(roadgraph_points)

    ranked = nearest_lane_candidates(polylines, x=0.0, y=1.0)

    assert ranked[0][0] == 1
    assert ranked[1][0] == 2
    assert ranked[0][1] < ranked[1][1]


def test_project_point_to_polyline_on_straight_segment():

    roadgraph_points = _make_roadgraph_points(
        {1: [(0.0, 0.0), (10.0, 0.0)]}
    )

    polyline = extract_lane_polylines(roadgraph_points)[0]

    projection = project_point_to_polyline(polyline, x=5.0, y=2.0)

    assert projection["arc_length_m"] == pytest.approx(5.0, abs=1e-3)
    assert projection["lateral_distance_m"] == pytest.approx(2.0, abs=1e-3)
    assert projection["heading_rad"] == pytest.approx(0.0, abs=1e-3)


def test_project_point_to_polyline_negative_lateral_side():

    roadgraph_points = _make_roadgraph_points(
        {1: [(0.0, 0.0), (10.0, 0.0)]}
    )

    polyline = extract_lane_polylines(roadgraph_points)[0]

    projection = project_point_to_polyline(polyline, x=5.0, y=-2.0)

    assert projection["lateral_distance_m"] == pytest.approx(
        -2.0, abs=1e-3
    )


def test_project_point_to_polyline_requires_two_points():

    roadgraph_points = _make_roadgraph_points({1: [(0.0, 0.0)]})

    polyline = extract_lane_polylines(roadgraph_points)[0]

    with pytest.raises(ValueError):
        project_point_to_polyline(polyline, x=0.0, y=0.0)
