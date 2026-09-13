"""Stage 3-A tests for src/planning/reference.py.

Synthetic hand-built geometries only (straight line, left/right arcs,
non-uniform spacing, duplicates, short-but-valid lines) -- the real-
WOMD audit lives in scripts/audit_phase3_reference_geometry.py per
the Stage 3-A gate, not here.
"""

import numpy as np
import pytest

from src.planning.reference import (
    DUPLICATE_POINT_EPS_M,
    InvalidReferenceGeometryError,
    ReferenceLine,
)
from src.scenarios.lane_geometry import LanePolyline, compute_arc_length


def _make_polyline(xy: np.ndarray, lane_id: int = 1) -> LanePolyline:
    xy = np.asarray(xy, dtype=np.float64)
    diffs = np.diff(xy, axis=0)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1e-9, norms)
    unit = diffs / norms
    direction = np.vstack([unit, unit[-1:]]) if len(unit) else np.zeros((1, 2))
    return LanePolyline(
        lane_id=lane_id,
        lane_type=1,
        xy=xy,
        direction=direction,
        arc_length=compute_arc_length(xy),
    )


def _straight_line(n=10, spacing=1.0):
    x = np.arange(n) * spacing
    y = np.zeros(n)
    return np.stack([x, y], axis=1)


def _arc(n=50, radius=20.0, turn_left=True, total_angle=np.pi / 2):
    angles = np.linspace(0.0, total_angle, n)
    if turn_left:
        x = radius * np.sin(angles)
        y = radius * (1 - np.cos(angles))
    else:
        x = radius * np.sin(angles)
        y = -radius * (1 - np.cos(angles))
    return np.stack([x, y], axis=1)


# ======================================================================
# straight line
# ======================================================================


def test_straight_line_curvature_near_zero():
    ref = ReferenceLine.from_lane_polyline(_make_polyline(_straight_line()))
    assert np.allclose(ref.curvature, 0.0, atol=1e-9)


def test_straight_line_yaw_constant():
    ref = ReferenceLine.from_lane_polyline(_make_polyline(_straight_line()))
    assert np.allclose(ref.yaw, 0.0, atol=1e-9)


def test_straight_line_arc_length_matches_spacing():
    ref = ReferenceLine.from_lane_polyline(_make_polyline(_straight_line(n=10, spacing=1.0)))
    assert np.isclose(ref.length_m, 9.0)
    assert np.allclose(np.diff(ref.s), 1.0)


# ======================================================================
# arcs: signed curvature
# ======================================================================


def test_left_arc_has_consistent_sign_curvature():
    ref = ReferenceLine.from_lane_polyline(
        _make_polyline(_arc(turn_left=True))
    )
    # Interior points only -- endpoints use one-sided differences and
    # can be noisier; the sign should still be consistent throughout,
    # but check interior strictly for magnitude too.
    interior = ref.curvature[2:-2]
    assert np.all(interior > 0.0)


def test_right_arc_has_opposite_sign_curvature():
    ref = ReferenceLine.from_lane_polyline(
        _make_polyline(_arc(turn_left=False))
    )
    interior = ref.curvature[2:-2]
    assert np.all(interior < 0.0)


def test_arc_curvature_magnitude_matches_radius():
    radius = 20.0
    ref = ReferenceLine.from_lane_polyline(
        _make_polyline(_arc(radius=radius, turn_left=True))
    )
    interior = ref.curvature[5:-5]
    expected = 1.0 / radius
    assert np.allclose(interior, expected, rtol=0.05)


# ======================================================================
# non-uniform spacing
# ======================================================================


def test_non_uniform_spacing_does_not_crash_and_is_finite():
    x = np.array([0.0, 0.5, 3.0, 3.2, 10.0, 10.1, 25.0])
    y = np.array([0.0, 0.1, 0.5, 0.55, 1.0, 1.02, 2.0])
    xy = np.stack([x, y], axis=1)
    ref = ReferenceLine.from_lane_polyline(_make_polyline(xy))
    assert np.all(np.isfinite(ref.curvature))
    assert np.all(np.isfinite(ref.curvature_derivative))
    assert np.all(np.isfinite(ref.yaw))
    assert np.all(np.isfinite(ref.s))


# ======================================================================
# duplicate points
# ======================================================================


def test_duplicate_points_are_cleaned_up():
    xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0 + 1e-6, 0.0 + 1e-6],  # near-duplicate of previous
            [2.0, 0.0],
            [3.0, 0.0],
        ]
    )
    ref = ReferenceLine.from_lane_polyline(_make_polyline(xy))
    # The near-duplicate point should have been dropped.
    assert ref.x.shape[0] == 4
    assert np.all(np.isfinite(ref.curvature))


def test_all_duplicate_points_raises():
    xy = np.array([[5.0, 5.0]] * 6)
    with pytest.raises(InvalidReferenceGeometryError):
        ReferenceLine.from_lane_polyline(_make_polyline(xy))


def test_non_finite_input_raises():
    xy = np.array([[0.0, 0.0], [1.0, np.nan], [2.0, 0.0]])
    with pytest.raises(InvalidReferenceGeometryError):
        ReferenceLine.from_lane_polyline(_make_polyline(xy))


# ======================================================================
# short-but-valid line (exactly 2 points)
# ======================================================================


def test_two_point_line_is_valid():
    xy = np.array([[0.0, 0.0], [5.0, 0.0]])
    ref = ReferenceLine.from_lane_polyline(_make_polyline(xy))
    assert ref.x.shape[0] == 2
    assert np.isclose(ref.length_m, 5.0)
    assert np.allclose(ref.curvature, 0.0)


# ======================================================================
# interpolation
# ======================================================================


def test_interpolation_at_sample_point_matches_exactly():
    ref = ReferenceLine.from_lane_polyline(_make_polyline(_straight_line(n=10)))
    state = ref.interpolate(3.0)
    assert np.isclose(state.x, 3.0)
    assert np.isclose(state.y, 0.0)


def test_interpolation_between_samples_is_continuous():
    ref = ReferenceLine.from_lane_polyline(_make_polyline(_straight_line(n=10)))
    prev = ref.interpolate(2.9)
    at = ref.interpolate(3.0)
    nxt = ref.interpolate(3.1)
    # Continuity: small step in s -> small step in x/y/yaw.
    assert abs(at.x - prev.x) < 0.2
    assert abs(nxt.x - at.x) < 0.2
    assert abs(at.yaw - prev.yaw) < 1e-6
    assert abs(nxt.yaw - at.yaw) < 1e-6


def test_interpolation_on_arc_is_sane():
    ref = ReferenceLine.from_lane_polyline(_make_polyline(_arc()))
    mid_s = ref.length_m / 2.0
    state = ref.interpolate(mid_s)
    assert np.isfinite(state.x)
    assert np.isfinite(state.y)
    assert np.isfinite(state.yaw)
    assert np.isfinite(state.curvature)


# ======================================================================
# upstream/downstream (open-path) queries
# ======================================================================


def test_query_upstream_of_start_extrapolates_along_tangent():
    ref = ReferenceLine.from_lane_polyline(_make_polyline(_straight_line(n=10)))
    state = ref.interpolate(-2.0)
    assert np.isclose(state.x, -2.0)
    assert np.isclose(state.y, 0.0)
    assert np.isclose(state.yaw, ref.yaw[0])


def test_query_downstream_of_end_extrapolates_along_tangent():
    ref = ReferenceLine.from_lane_polyline(_make_polyline(_straight_line(n=10)))
    end_s = ref.length_m
    state = ref.interpolate(end_s + 5.0)
    assert np.isclose(state.x, end_s + 5.0)
    assert np.isclose(state.y, 0.0)
    assert np.isclose(state.yaw, ref.yaw[-1])


def test_query_upstream_curvature_held_at_boundary():
    ref = ReferenceLine.from_lane_polyline(_make_polyline(_arc()))
    state = ref.interpolate(-1.0)
    assert np.isclose(state.curvature, ref.curvature[0])


# ======================================================================
# critical invariant: d positive = LEFT of travel direction
# ======================================================================


def test_projection_left_of_travel_direction_is_positive_d():
    # Straight line traveling in +x: a point above the line (larger y)
    # is to the LEFT of travel direction, and must have positive
    # lateral_distance_m -- matching lane_geometry.py's own
    # project_point_to_polyline convention
    # (perpendicular = [-dir_y, dir_x]).
    ref = ReferenceLine.from_lane_polyline(_make_polyline(_straight_line(n=10)))
    result = ref.project(x=3.0, y=1.0)
    assert result["lateral_distance_m"] > 0.0
    assert np.isclose(result["lateral_distance_m"], 1.0)


def test_projection_right_of_travel_direction_is_negative_d():
    ref = ReferenceLine.from_lane_polyline(_make_polyline(_straight_line(n=10)))
    result = ref.project(x=3.0, y=-1.0)
    assert result["lateral_distance_m"] < 0.0
    assert np.isclose(result["lateral_distance_m"], -1.0)


def test_projection_arc_length_matches_expected_s():
    ref = ReferenceLine.from_lane_polyline(_make_polyline(_straight_line(n=10)))
    result = ref.project(x=4.0, y=0.5)
    assert np.isclose(result["arc_length_m"], 4.0, atol=1e-6)


def test_projection_upstream_of_start_is_negative_s():
    ref = ReferenceLine.from_lane_polyline(_make_polyline(_straight_line(n=10)))
    result = ref.project(x=-3.0, y=0.0)
    assert result["arc_length_m"] < 0.0
