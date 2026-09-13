"""Stage 3-B tests for src/planning/frenet_transform.py.

Synthetic straight/curved reference lines here; a real-WOMD spot-check
lives in scripts/audit_phase3_frenet_transform.py per the Stage 3-B
gate, not here.
"""

import numpy as np
import pytest

from src.planning.frenet_transform import (
    LOW_SPEED_THRESHOLD_MPS,
    cartesian_to_frenet,
    frenet_to_cartesian,
)
from src.planning.frenet_types import FrenetState
from src.planning.reference import ReferenceLine
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


def _straight_line(n=200, spacing=1.0):
    x = np.arange(n) * spacing
    y = np.zeros(n)
    return np.stack([x, y], axis=1)


def _arc(n=4000, radius=50.0, turn_left=True, total_angle=np.pi / 2):
    angles = np.linspace(0.0, total_angle, n)
    if turn_left:
        x = radius * np.sin(angles)
        y = radius * (1 - np.cos(angles))
    else:
        x = radius * np.sin(angles)
        y = -radius * (1 - np.cos(angles))
    return np.stack([x, y], axis=1)


@pytest.fixture
def straight_ref():
    return ReferenceLine.from_lane_polyline(_make_polyline(_straight_line()))


@pytest.fixture
def curved_ref():
    # Constant-curvature left-turning arc, radius 50m -> kappa = 1/50 = 0.02
    return ReferenceLine.from_lane_polyline(_make_polyline(_arc(turn_left=True)))


@pytest.fixture
def curved_ref_right():
    return ReferenceLine.from_lane_polyline(_make_polyline(_arc(turn_left=False)))


ROUND_TRIP_TOL = 1e-4

# Curved-reference round trip has an additional, expected error source
# beyond floating point: ReferenceLine's yaw/curvature are themselves
# finite-difference ESTIMATES over a discretely-sampled polyline (see
# Stage 3-A docstring), not the arc's true closed-form curvature. That
# discretization error shows up only in reconstructed (x, y) (yaw,
# velocity, acceleration all round-trip to floating-point precision;
# verified separately below), grows with the lateral offset |d| being
# tested (error is ~curvature-estimate-error * |d|), and shrinks with
# sample density. Measured directly on this fixture at the largest |d|
# case exercised below (|d| ~= 10 m): ~1.7e-2 m residual at 400 points,
# ~3e-3 at 2000, ~4e-4 at 8000 (falls off much faster than a linear
# ~1/n curvature error alone would predict, consistent with the
# second-order-accurate central-difference scheme). The curved fixture
# below uses 4000 points (denser than Stage 3-A's own curved-reference
# tests) specifically so this stays comfortably small even for the
# larger |d| cases tested here.
CURVED_ROUND_TRIP_POSITION_TOL = 3e-3


def _roundtrip(cart_x, cart_y, cart_yaw, cart_v, cart_a, reference):
    frenet = cartesian_to_frenet(cart_x, cart_y, cart_yaw, cart_v, cart_a, reference)
    cart2 = frenet_to_cartesian(frenet, reference)
    return frenet, cart2


class TestRoundTripStraight:
    def test_roundtrip_various_states(self, straight_ref):
        cases = [
            (50.0, 2.0, 0.05, 12.0, 1.0),
            (10.0, -1.5, -0.1, 8.0, -0.5),
            (100.0, 0.0, 0.0, 15.0, 0.0),
            (5.0, 3.0, 0.2, 5.0, 2.0),
        ]
        for x, y, yaw, v, a in cases:
            frenet, cart2 = _roundtrip(x, y, yaw, v, a, straight_ref)
            assert cart2.x == pytest.approx(x, abs=ROUND_TRIP_TOL)
            assert cart2.y == pytest.approx(y, abs=ROUND_TRIP_TOL)
            assert cart2.yaw == pytest.approx(yaw, abs=ROUND_TRIP_TOL)
            assert cart2.velocity == pytest.approx(v, abs=ROUND_TRIP_TOL)
            assert cart2.acceleration == pytest.approx(a, abs=ROUND_TRIP_TOL)


class TestRoundTripCurved:
    def test_roundtrip_various_states(self, curved_ref):
        cases = [
            (20.0, 5.0, 0.3, 10.0, 0.5),
            (30.0, -2.0, 0.1, 12.0, -1.0),
            (10.0, 0.0, np.pi / 6, 8.0, 0.0),
        ]
        for x, y, yaw, v, a in cases:
            frenet, cart2 = _roundtrip(x, y, yaw, v, a, curved_ref)
            assert cart2.x == pytest.approx(x, abs=CURVED_ROUND_TRIP_POSITION_TOL)
            assert cart2.y == pytest.approx(y, abs=CURVED_ROUND_TRIP_POSITION_TOL)
            assert cart2.yaw == pytest.approx(yaw, abs=ROUND_TRIP_TOL)
            assert cart2.velocity == pytest.approx(v, abs=ROUND_TRIP_TOL)
            assert cart2.acceleration == pytest.approx(a, abs=ROUND_TRIP_TOL)


class TestSignConvention:
    def test_positive_d_is_left_of_travel(self, straight_ref):
        # Straight line travels in +x direction; left-of-travel is +y.
        state = FrenetState(s=10.0, s_d=10.0, s_dd=0.0, d=2.0, d_d=0.0, d_dd=0.0)
        cart = frenet_to_cartesian(state, straight_ref)
        assert cart.y == pytest.approx(2.0, abs=1e-6)

        state_neg = FrenetState(s=10.0, s_d=10.0, s_dd=0.0, d=-2.0, d_d=0.0, d_dd=0.0)
        cart_neg = frenet_to_cartesian(state_neg, straight_ref)
        assert cart_neg.y == pytest.approx(-2.0, abs=1e-6)

    def test_consistent_with_reference_projection_sign(self, straight_ref):
        # Directly verify against ReferenceLine.project's own sign
        # convention (Stage 3-A), rather than assuming.
        projection = straight_ref.project(10.0, 2.0)
        assert projection["lateral_distance_m"] > 0.0


class TestLowSpeed:
    def test_frenet_to_cartesian_zero_speed_finite(self, straight_ref):
        state = FrenetState(s=5.0, s_d=0.0, s_dd=0.0, d=1.0, d_d=0.0, d_dd=0.0)
        cart = frenet_to_cartesian(state, straight_ref)
        assert np.isfinite(cart.x)
        assert np.isfinite(cart.y)
        assert np.isfinite(cart.yaw)
        assert np.isfinite(cart.curvature)
        assert np.isfinite(cart.velocity)
        assert np.isfinite(cart.acceleration)

    def test_frenet_to_cartesian_tiny_speed_finite(self, straight_ref):
        state = FrenetState(s=5.0, s_d=0.01, s_dd=0.1, d=0.5, d_d=0.0, d_dd=0.0)
        cart = frenet_to_cartesian(state, straight_ref)
        assert np.isfinite(cart.x)
        assert np.isfinite(cart.y)
        assert np.isfinite(cart.yaw)
        assert np.isfinite(cart.curvature)
        assert np.isfinite(cart.velocity)
        assert np.isfinite(cart.acceleration)

    def test_cartesian_to_frenet_zero_speed_finite(self, straight_ref):
        frenet = cartesian_to_frenet(5.0, 1.0, 0.0, 0.0, 0.0, straight_ref)
        assert np.isfinite(frenet.s)
        assert np.isfinite(frenet.s_d)
        assert np.isfinite(frenet.s_dd)
        assert np.isfinite(frenet.d)
        assert np.isfinite(frenet.d_d)
        assert np.isfinite(frenet.d_dd)

    def test_cartesian_to_frenet_tiny_speed_finite(self, straight_ref):
        frenet = cartesian_to_frenet(5.0, 1.0, 0.05, 0.01, 0.1, straight_ref)
        assert np.isfinite(frenet.s)
        assert np.isfinite(frenet.s_d)
        assert np.isfinite(frenet.s_dd)
        assert np.isfinite(frenet.d)
        assert np.isfinite(frenet.d_d)
        assert np.isfinite(frenet.d_dd)

    def test_stop_maneuver_terminal_state_finite(self, straight_ref):
        # STOP_TARGET_SPEED_MPS = 0.0 from behavior_action.py: verify
        # the exact terminal state of a STOP is handled cleanly.
        state = FrenetState(s=30.0, s_d=0.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
        cart = frenet_to_cartesian(state, straight_ref)
        assert np.isfinite(cart.x) and np.isfinite(cart.y)
        frenet_back = cartesian_to_frenet(
            cart.x, cart.y, cart.yaw, cart.velocity, cart.acceleration, straight_ref
        )
        assert np.isfinite(frenet_back.s)
        assert np.isfinite(frenet_back.d)

    def test_threshold_is_positive_and_reasonable(self):
        assert LOW_SPEED_THRESHOLD_MPS > 0.0
        assert LOW_SPEED_THRESHOLD_MPS < 15.0  # NOMINAL_CRUISE_SPEED_MPS


class TestYawWrap:
    def test_yaw_stays_wrapped(self, straight_ref):
        # Force a large heading offset that would naively exceed pi.
        state = FrenetState(s=10.0, s_d=1.0, s_dd=0.0, d=0.0, d_d=50.0, d_dd=0.0)
        cart = frenet_to_cartesian(state, straight_ref)
        assert -np.pi <= cart.yaw <= np.pi

    def test_yaw_wrap_near_boundary(self, straight_ref):
        for raw_yaw in [np.pi - 0.01, -np.pi + 0.01, 3.0, -3.0]:
            frenet = cartesian_to_frenet(10.0, 0.0, raw_yaw, 5.0, 0.0, straight_ref)
            cart = frenet_to_cartesian(frenet, straight_ref)
            assert -np.pi <= cart.yaw <= np.pi


class TestCurvatureFiniteness:
    def test_finite_across_d_range(self, curved_ref):
        for d in np.linspace(-5.0, 5.0, 11):
            state = FrenetState(s=30.0, s_d=10.0, s_dd=0.0, d=float(d), d_d=0.0, d_dd=0.0)
            cart = frenet_to_cartesian(state, curved_ref)
            assert np.isfinite(cart.curvature)

    def test_curvature_sign_matches_turn_direction(self, curved_ref, curved_ref_right):
        state = FrenetState(s=30.0, s_d=10.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
        left_cart = frenet_to_cartesian(state, curved_ref)
        right_cart = frenet_to_cartesian(state, curved_ref_right)
        assert left_cart.curvature > 0.0
        assert right_cart.curvature < 0.0


class TestReferenceBoundary:
    def test_s_before_start(self, straight_ref):
        state = FrenetState(s=-5.0, s_d=10.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
        cart = frenet_to_cartesian(state, straight_ref)
        assert np.isfinite(cart.x) and np.isfinite(cart.y)
        # Extrapolated backwards along +x tangent -> negative x.
        assert cart.x < 0.0

    def test_s_after_end(self, straight_ref):
        far_s = straight_ref.length_m + 50.0
        state = FrenetState(s=far_s, s_d=10.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
        cart = frenet_to_cartesian(state, straight_ref)
        assert np.isfinite(cart.x) and np.isfinite(cart.y)
        assert cart.x > straight_ref.x[-1]

    def test_s_slightly_beyond_end_roundtrip(self, straight_ref):
        far_s = straight_ref.length_m + 1.0
        state = FrenetState(s=far_s, s_d=10.0, s_dd=0.0, d=1.0, d_d=0.0, d_dd=0.0)
        cart = frenet_to_cartesian(state, straight_ref)
        assert np.isfinite(cart.x) and np.isfinite(cart.y)
