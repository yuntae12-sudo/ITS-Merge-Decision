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
    ALL_LANE_TYPE_IDS,
    compute_arc_length,
    extract_lane_polylines,
    nearest_lane_candidates,
    project_point_to_polyline,
    project_point_to_polyline_signed,
)


def _make_roadgraph_points(
    xy_by_id: dict,
    lane_type: int = 2,
    direction_by_id: dict = None,
) -> RoadgraphPoints:
    """Builds a RoadgraphPoints from {lane_id: [(x, y), ...]}.

    Point storage order for each id is exactly what's passed in
    ``xy_by_id`` -- tests intentionally shuffle it to prove
    reconstruction doesn't trust raw array order.

    ``direction_by_id`` independently specifies the *true* WOMD
    per-point travel direction for each id, as a single (dx, dy) unit
    vector applied to every point of that id (WOMD direction is
    locally near-constant on a straight lane segment, which is all
    these tests use). This is deliberately decoupled from
    ``xy_by_id``'s storage order: unlike a real WOMD scenario, this
    test data does not derive direction from consecutive stored
    points, so a test can shuffle storage order and/or reverse the true
    direction independently and verify the reconstructed polyline
    still ends up aligned with the given direction. If omitted for an
    id, direction defaults to (1, 0).
    """

    direction_by_id = direction_by_id or {}

    xs, ys, ids, types, dir_xs, dir_ys = [], [], [], [], [], []

    for lane_id, points in xy_by_id.items():

        dx, dy = direction_by_id.get(lane_id, (1.0, 0.0))
        norm = np.hypot(dx, dy) or 1.0
        dx, dy = dx / norm, dy / norm

        for x, y in points:
            xs.append(x)
            ys.append(y)
            ids.append(lane_id)
            types.append(lane_type)
            dir_xs.append(dx)
            dir_ys.append(dy)

    num_points = len(xs)

    return RoadgraphPoints(
        x=np.asarray(xs, dtype=np.float32),
        y=np.asarray(ys, dtype=np.float32),
        z=np.zeros(num_points, dtype=np.float32),
        dir_x=np.asarray(dir_xs, dtype=np.float32),
        dir_y=np.asarray(dir_ys, dtype=np.float32),
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


def test_extract_lane_polylines_aligns_to_positive_travel_direction():
    """The shuffled-order test above only checks monotonicity, so it
    cannot catch a polyline reconstructed backwards relative to WOMD's
    own travel direction. This pins the direction independently of
    storage order: true WOMD direction is (+1, 0), so the reconstructed
    polyline must run with increasing x regardless of how the points
    were shuffled in storage.
    """

    straight_line = [(float(i), 0.0) for i in range(10)]

    shuffled = straight_line.copy()
    rng = np.random.default_rng(seed=1)
    rng.shuffle(shuffled)

    roadgraph_points = _make_roadgraph_points(
        {7: shuffled}, direction_by_id={7: (1.0, 0.0)}
    )

    polyline = extract_lane_polylines(roadgraph_points)[0]

    xs = polyline.xy[:, 0]
    assert np.all(np.diff(xs) > 0)
    assert polyline.arc_length[0] == pytest.approx(0.0)
    assert polyline.xy[0, 0] == pytest.approx(0.0)


def test_extract_lane_polylines_aligns_to_negative_travel_direction():
    """Same geometry as above, but with WOMD direction reversed to
    (-1, 0): the reconstructed polyline must now run with decreasing x,
    proving orientation is driven by WOMD direction and not by an
    arbitrary geometric chaining endpoint.
    """

    straight_line = [(float(i), 0.0) for i in range(10)]

    shuffled = straight_line.copy()
    rng = np.random.default_rng(seed=1)
    rng.shuffle(shuffled)

    roadgraph_points = _make_roadgraph_points(
        {7: shuffled}, direction_by_id={7: (-1.0, 0.0)}
    )

    polyline = extract_lane_polylines(roadgraph_points)[0]

    xs = polyline.xy[:, 0]
    assert np.all(np.diff(xs) < 0)
    assert polyline.arc_length[0] == pytest.approx(0.0)
    assert polyline.xy[0, 0] == pytest.approx(9.0)


def test_extract_lane_polylines_filters_non_lane_types():

    roadgraph_points = _make_roadgraph_points(
        {1: [(0.0, 0.0), (1.0, 0.0)]}, lane_type=17  # STOP_SIGN
    )

    polylines = extract_lane_polylines(roadgraph_points)

    assert polylines == []


def test_extract_lane_polylines_default_excludes_bike_lane():
    """Ego lane assignment / merge detection default to vehicle-only
    lane types: a vehicle cannot be assigned to or merge into a bike
    lane in this study.

    ``_make_roadgraph_points`` assigns one ``lane_type`` per call, so
    a freeway-lane id and a bike-lane id are built separately here and
    their point arrays concatenated to give each id its own type.
    """

    freeway_points = _make_roadgraph_points(
        {1: [(0.0, 0.0), (1.0, 0.0)]}, lane_type=1
    )
    bike_points = _make_roadgraph_points(
        {3: [(0.0, 10.0), (1.0, 10.0)]}, lane_type=3
    )

    combined = RoadgraphPoints(
        x=np.concatenate([freeway_points.x, bike_points.x]),
        y=np.concatenate([freeway_points.y, bike_points.y]),
        z=np.concatenate([freeway_points.z, bike_points.z]),
        dir_x=np.concatenate([freeway_points.dir_x, bike_points.dir_x]),
        dir_y=np.concatenate([freeway_points.dir_y, bike_points.dir_y]),
        dir_z=np.concatenate([freeway_points.dir_z, bike_points.dir_z]),
        types=np.concatenate([freeway_points.types, bike_points.types]),
        ids=np.concatenate([freeway_points.ids, bike_points.ids]),
        valid=np.concatenate([freeway_points.valid, bike_points.valid]),
    )

    default_polylines = extract_lane_polylines(combined)
    all_lane_polylines = extract_lane_polylines(
        combined, lane_type_ids=ALL_LANE_TYPE_IDS
    )

    default_ids = sorted(p.lane_id for p in default_polylines)
    all_ids = sorted(p.lane_id for p in all_lane_polylines)

    assert default_ids == [1]
    assert all_ids == [1, 3]


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


# --- Stage B-0 fix: project_point_to_polyline_signed ---
#
# project_point_to_polyline CLAMPS arc_length_m to the polyline's own
# domain [0, total_length] -- correct for merge_detector.py's
# endpoint/d_m measurements, but wrong for relative front/rear
# ordering: two physically distinct points both upstream (or both
# downstream) of the polyline collapse to the SAME clamped arc
# length, making their difference exactly 0.0 instead of reflecting
# their true relative order. project_point_to_polyline_signed fixes
# this by extending the boundary segment's own local tangent instead
# of clamping.


def test_signed_projection_matches_clamped_for_interior_point():

    roadgraph_points = _make_roadgraph_points(
        {1: [(0.0, 0.0), (10.0, 0.0)]}
    )
    polyline = extract_lane_polylines(roadgraph_points)[0]

    clamped = project_point_to_polyline(polyline, x=5.0, y=2.0)
    signed = project_point_to_polyline_signed(polyline, x=5.0, y=2.0)

    assert signed["arc_length_m"] == pytest.approx(
        clamped["arc_length_m"], abs=1e-6
    )
    assert signed["lateral_distance_m"] == pytest.approx(
        clamped["lateral_distance_m"], abs=1e-6
    )
    assert signed["heading_rad"] == pytest.approx(
        clamped["heading_rad"], abs=1e-6
    )


def test_signed_projection_goes_negative_before_polyline_start():

    roadgraph_points = _make_roadgraph_points(
        {1: [(0.0, 0.0), (10.0, 0.0)]}
    )
    polyline = extract_lane_polylines(roadgraph_points)[0]

    # 5 m before the polyline's own start (x=0), still on the lane's
    # centerline extended backward.
    clamped = project_point_to_polyline(polyline, x=-5.0, y=0.0)
    signed = project_point_to_polyline_signed(polyline, x=-5.0, y=0.0)

    assert clamped["arc_length_m"] == pytest.approx(0.0, abs=1e-6)
    assert signed["arc_length_m"] == pytest.approx(-5.0, abs=1e-3)


def test_signed_projection_exceeds_length_past_polyline_end():

    roadgraph_points = _make_roadgraph_points(
        {1: [(0.0, 0.0), (10.0, 0.0)]}
    )
    polyline = extract_lane_polylines(roadgraph_points)[0]

    clamped = project_point_to_polyline(polyline, x=15.0, y=0.0)
    signed = project_point_to_polyline_signed(polyline, x=15.0, y=0.0)

    assert clamped["arc_length_m"] == pytest.approx(10.0, abs=1e-6)
    assert signed["arc_length_m"] == pytest.approx(15.0, abs=1e-3)


def test_signed_projection_preserves_relative_order_before_start():
    """The exact bug this fix targets: two points both upstream of the
    polyline's start must NOT both collapse to arc_length_m=0.0 --
    their true relative order (5 m apart) must be preserved."""

    roadgraph_points = _make_roadgraph_points(
        {1: [(0.0, 0.0), (10.0, 0.0)]}
    )
    polyline = extract_lane_polylines(roadgraph_points)[0]

    near = project_point_to_polyline(polyline, x=-2.0, y=0.0)
    far = project_point_to_polyline(polyline, x=-7.0, y=0.0)
    assert near["arc_length_m"] == far["arc_length_m"] == pytest.approx(
        0.0, abs=1e-6
    )  # the bug: both clamp to the same value

    near_signed = project_point_to_polyline_signed(polyline, x=-2.0, y=0.0)
    far_signed = project_point_to_polyline_signed(polyline, x=-7.0, y=0.0)
    assert near_signed["arc_length_m"] > far_signed["arc_length_m"]
    assert near_signed["arc_length_m"] - far_signed["arc_length_m"] == (
        pytest.approx(5.0, abs=1e-3)
    )


def test_signed_projection_requires_two_points():

    roadgraph_points = _make_roadgraph_points({1: [(0.0, 0.0)]})
    polyline = extract_lane_polylines(roadgraph_points)[0]

    with pytest.raises(ValueError):
        project_point_to_polyline_signed(polyline, x=0.0, y=0.0)


# --- Stage 3-H regression: dedicated lock-in for the Stage 3-G bug fix
# in project_point_to_polyline_signed's extrapolation branches.
#
# The bug (see docs/phase3/OVERNIGHT_PROGRESS.md Stage 3-G entry, and
# the fix's own inline comment in lane_geometry.py):
# `lateral_distance_m` was computed as the Euclidean distance from the
# query point to the CLAMPED boundary-endpoint projection, not the
# true perpendicular distance to the segment's infinite tangent line.
# For an on-centerline query (y=0.0, the ONLY case every pre-existing
# extrapolation test above uses) this distinction is invisible -- the
# lateral offset is exactly 0.0 either way. The bug only manifests for
# a query point that is BOTH far outside the polyline's domain along
# the tangent AND offset perpendicular to it (e.g. "~1m off centerline
# but ~45m before/after the polyline"), which is exactly the blind
# spot none of the tests above cover. These tests close that gap with
# hand-computed independent expected values (never calling the
# function under test to derive its own expected output).


def test_signed_projection_extrapolation_upstream_true_perpendicular_offset():
    """Query point far upstream of the first segment, with a nonzero
    perpendicular offset. On a straight polyline along +x from (0,0)
    to (10,0), the segment's tangent is (1, 0) and its left-hand
    perpendicular is (0, 1) (matching this module's own sign
    convention: perpendicular = (-dir_y, dir_x)). A query point at
    (x=-45.0, y=1.0) sits 45 m before the polyline's own domain, but
    only 1.0 m off the extended centerline -- independently computed
    by basic vector projection, not by calling the function under
    test."""

    roadgraph_points = _make_roadgraph_points(
        {1: [(0.0, 0.0), (10.0, 0.0)]}
    )
    polyline = extract_lane_polylines(roadgraph_points)[0]

    query_x, query_y = -45.0, 1.0

    # Independent hand computation: tangent = (1, 0), origin = (0, 0).
    # t_raw = dot(query - origin, tangent) / |tangent|^2 = -45.0.
    # unclamped_projected = origin + t_raw * tangent = (-45.0, 0.0).
    # offset = query - unclamped_projected = (0.0, 1.0).
    # perpendicular = (-tangent_y, tangent_x) = (0.0, 1.0).
    # true_lateral = dot(offset, perpendicular) = 1.0.
    expected_arc_length_m = -45.0
    expected_lateral_distance_m = 1.0

    result = project_point_to_polyline_signed(polyline, x=query_x, y=query_y)

    assert result["arc_length_m"] == pytest.approx(
        expected_arc_length_m, abs=1e-6
    )
    assert result["lateral_distance_m"] == pytest.approx(
        expected_lateral_distance_m, abs=1e-6
    )
    # The buggy value (Euclidean distance to the clamped endpoint
    # (0,0)) would have been hypot(45.0, 1.0) ~= 45.011 -- assert the
    # true value is nowhere near that, as an explicit anti-regression
    # tripwire in addition to the exact equality above.
    assert abs(result["lateral_distance_m"]) < 2.0


def test_signed_projection_extrapolation_downstream_true_perpendicular_offset():
    """Query point far downstream of the last segment, with a nonzero
    perpendicular offset. Mirror of the upstream case above, past the
    polyline's end (x=10.0)."""

    roadgraph_points = _make_roadgraph_points(
        {1: [(0.0, 0.0), (10.0, 0.0)]}
    )
    polyline = extract_lane_polylines(roadgraph_points)[0]

    query_x, query_y = 55.0, -2.0

    # Independent hand computation: tangent = (1, 0), last point
    # (10, 0). t_raw = dot((55,-2) - (10,0), (1,0)) / 1.0 = 45.0.
    # unclamped_projected = (10, 0) + 45.0 * (1, 0) = (55.0, 0.0).
    # offset = (55.0, -2.0) - (55.0, 0.0) = (0.0, -2.0).
    # perpendicular = (0.0, 1.0). true_lateral = dot(offset, perp) = -2.0.
    expected_arc_length_m = 10.0 + 45.0  # arc_end + t_raw * seg_len
    expected_lateral_distance_m = -2.0

    result = project_point_to_polyline_signed(polyline, x=query_x, y=query_y)

    assert result["arc_length_m"] == pytest.approx(
        expected_arc_length_m, abs=1e-6
    )
    assert result["lateral_distance_m"] == pytest.approx(
        expected_lateral_distance_m, abs=1e-6
    )
    # Buggy value would have been hypot(45.0, 2.0) ~= 45.044.
    assert abs(result["lateral_distance_m"]) < 3.0


def test_signed_projection_lateral_distance_invariant_to_extrapolation_distance():
    """The exact bug class that occurred: lateral_distance_m must NOT
    scale with how far upstream/downstream the query point is. Varying
    ONLY the along-tangent extrapolation distance while holding the
    perpendicular offset fixed at 1.0 m must leave lateral_distance_m
    unchanged (up to floating-point tolerance) -- this is precisely
    the invariant the buggy Euclidean-to-clamped-endpoint computation
    violated (it grew with the along-tangent distance instead of
    staying constant)."""

    roadgraph_points = _make_roadgraph_points(
        {1: [(0.0, 0.0), (10.0, 0.0)]}
    )
    polyline = extract_lane_polylines(roadgraph_points)[0]

    fixed_perpendicular_offset = 1.0
    upstream_extrapolation_distances = [0.5, 5.0, 45.0, 500.0]

    lateral_values = [
        project_point_to_polyline_signed(
            polyline,
            x=-distance,
            y=fixed_perpendicular_offset,
        )["lateral_distance_m"]
        for distance in upstream_extrapolation_distances
    ]

    for value in lateral_values:
        assert value == pytest.approx(fixed_perpendicular_offset, abs=1e-6)

    # Same invariant downstream of the last point (x=10.0).
    downstream_lateral_values = [
        project_point_to_polyline_signed(
            polyline,
            x=10.0 + distance,
            y=fixed_perpendicular_offset,
        )["lateral_distance_m"]
        for distance in upstream_extrapolation_distances
    ]

    for value in downstream_lateral_values:
        assert value == pytest.approx(fixed_perpendicular_offset, abs=1e-6)
