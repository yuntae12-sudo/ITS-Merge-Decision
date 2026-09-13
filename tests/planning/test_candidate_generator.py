"""Stage 3-C tests for src/planning/candidate_generator.py.

Synthetic straight/curved references (same construction pattern as
Stage 3-A/3-B's own tests). Real-WOMD maneuver coverage is exercised
in test_frenet_planner.py (the higher-level entry point), per the
task brief's "at least the physically-different-actions checks" real
-data requirement.
"""

import numpy as np
import pytest

from src.planning.candidate_generator import (
    generate_follow_or_merge_candidate,
    generate_keep_candidate,
    generate_stop_candidate,
    project_cartesian_to_frame,
)
from src.planning.frenet_transform import frenet_to_cartesian
from src.planning.frenet_types import FrenetState
from src.planning.reference import ReferenceLine
from src.scenarios.lane_geometry import LanePolyline, compute_arc_length

HORIZON_S = 3.0
DT_S = 0.1
COMFORTABLE_DECEL = 2.0


def _make_polyline(xy: np.ndarray, lane_id: int = 1) -> LanePolyline:
    xy = np.asarray(xy, dtype=np.float64)
    diffs = np.diff(xy, axis=0)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1e-9, norms)
    unit = diffs / norms
    direction = np.vstack([unit, unit[-1:]]) if len(unit) else np.zeros((1, 2))
    return LanePolyline(
        lane_id=lane_id, lane_type=1, xy=xy, direction=direction,
        arc_length=compute_arc_length(xy),
    )


def _straight_line(n=500, spacing=1.0):
    x = np.arange(n) * spacing
    y = np.zeros(n)
    return np.stack([x, y], axis=1)


@pytest.fixture
def straight_ref():
    return ReferenceLine.from_lane_polyline(_make_polyline(_straight_line()))


def _straight_ego_state(s=10.0, s_d=15.0, d=0.5):
    return FrenetState(s=s, s_d=s_d, s_dd=0.0, d=d, d_d=0.0, d_dd=0.0)


def test_keep_converges_toward_reference_speed(straight_ref):
    ego = _straight_ego_state(s_d=10.0)
    path = generate_keep_candidate(ego, reference_speed_mps=15.0, horizon_s=HORIZON_S, dt_s=DT_S)
    assert np.isclose(path.s_d[-1], 15.0, atol=1e-6)
    assert np.isclose(path.d[-1], 0.0, atol=1e-6)
    assert np.isclose(path.d_d[-1], 0.0, atol=1e-6)


def test_keep_lateral_converges_to_zero_from_nonzero_offset(straight_ref):
    ego = _straight_ego_state(d=1.2)
    path = generate_keep_candidate(ego, reference_speed_mps=15.0, horizon_s=HORIZON_S, dt_s=DT_S)
    assert np.isclose(path.d[0], 1.2, atol=1e-6)
    assert np.isclose(path.d[-1], 0.0, atol=1e-8)
    # monotonically converging, not oscillating wildly
    assert np.all(np.abs(path.d) <= 1.2 + 1e-6)


def test_stop_target_computed_from_formula(straight_ref):
    ego = _straight_ego_state(s=10.0, s_d=12.0)
    result = generate_stop_candidate(
        ego, COMFORTABLE_DECEL, HORIZON_S, DT_S, straight_ref,
    )
    expected_braking_distance = (12.0 ** 2) / (2 * COMFORTABLE_DECEL)
    expected_s_stop = 10.0 + expected_braking_distance
    assert result.ok
    assert not result.stop_target_clamped
    assert np.isclose(result.terminal_frenet_state.s, expected_s_stop, atol=1e-6)
    assert np.isclose(result.frenet_path.s[-1], expected_s_stop, atol=1e-3)
    assert np.isclose(result.frenet_path.s_d[-1], 0.0, atol=1e-6)


def test_stop_clamps_when_target_exceeds_reference_domain():
    # A short reference (20m) with a high current speed that would
    # otherwise require a stop point far beyond the reference's end.
    short_ref = ReferenceLine.from_lane_polyline(_make_polyline(_straight_line(n=21, spacing=1.0)))
    ego = FrenetState(s=5.0, s_d=30.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
    result = generate_stop_candidate(
        ego, COMFORTABLE_DECEL, HORIZON_S, DT_S, short_ref,
    )
    assert result.ok
    assert result.stop_target_clamped is True
    assert np.isclose(result.terminal_frenet_state.s, short_ref.s[-1], atol=1e-8)


def test_keep_and_stop_produce_physically_different_longitudinal_trajectories(straight_ref):
    ego = _straight_ego_state(s=10.0, s_d=15.0)
    keep_path = generate_keep_candidate(ego, reference_speed_mps=15.0, horizon_s=HORIZON_S, dt_s=DT_S)
    stop_result = generate_stop_candidate(ego, COMFORTABLE_DECEL, HORIZON_S, DT_S, straight_ref)
    stop_path = stop_result.frenet_path

    # STOP's longitudinal horizon is the physically natural stopping
    # time (v/comfortable_deceleration), which may differ from KEEP's
    # generic horizon_s -- compare over the overlapping time range
    # only.
    n_common = min(len(keep_path.t), len(stop_path.t))
    assert not np.allclose(keep_path.s[:n_common], stop_path.s[:n_common], atol=1e-2)
    assert not np.allclose(keep_path.s_d[:n_common], stop_path.s_d[:n_common], atol=1e-2)
    # KEEP maintains speed; STOP decelerates to 0.
    assert keep_path.s_d[-1] > 10.0
    assert np.isclose(stop_path.s_d[-1], 0.0, atol=1e-6)


def test_follow_reacts_to_closer_faster_closing_lead_with_lower_target_speed():
    ego = _straight_ego_state(s_d=15.0)

    # Emulate BehaviorExecutor._compute_follow_speed with two distinct
    # gap/relative-speed inputs (closer & faster-closing vs. farther &
    # non-closing) -- this test exercises the SAME follow-speed
    # computation the frozen BehaviorExecutor uses, applied at the
    # planner level, to confirm FOLLOW's generated trajectory actually
    # differs when the causal inputs differ.
    def follow_speed(relative_speed_mps):
        closing_speed = max(relative_speed_mps, 0.0)
        return max(15.0 - closing_speed, 0.0)

    close_fast_speed = follow_speed(relative_speed_mps=8.0)  # closing fast
    far_slow_speed = follow_speed(relative_speed_mps=0.0)  # not closing

    close_path = generate_follow_or_merge_candidate(
        ego, close_fast_speed, HORIZON_S, DT_S,
    )
    far_path = generate_follow_or_merge_candidate(
        ego, far_slow_speed, HORIZON_S, DT_S,
    )

    assert close_path.s_d[-1] < far_path.s_d[-1]
    assert np.isclose(close_path.s_d[-1], 7.0, atol=1e-6)
    assert np.isclose(far_path.s_d[-1], 15.0, atol=1e-6)


def test_merge_terminal_lateral_position_converges_to_zero_in_target_frame():
    # Two parallel straight lanes offset laterally by 3.5m (a typical
    # lane width), representing source/target.
    source_xy = _straight_line(n=200)
    target_xy = source_xy.copy()
    target_xy[:, 1] += 3.5
    source_ref = ReferenceLine.from_lane_polyline(_make_polyline(source_xy, lane_id=1))
    target_ref = ReferenceLine.from_lane_polyline(_make_polyline(target_xy, lane_id=2))

    # Ego starts physically ON the source lane's centerline.
    ego_x, ego_y, ego_yaw, ego_speed = 10.0, 0.0, 0.0, 15.0
    ego_frenet_in_target = project_cartesian_to_frame(
        ego_x, ego_y, ego_yaw, ego_speed, target_ref,
    )
    # Ego should show up ~3.5m to the right (d<0) of the target lane.
    assert ego_frenet_in_target.d < -3.0

    merge_path = generate_follow_or_merge_candidate(
        ego_frenet_in_target, reference_speed_mps=15.0, horizon_s=HORIZON_S, dt_s=DT_S,
    )
    assert np.isclose(merge_path.d[-1], 0.0, atol=1e-6)

    # Verify by transforming the terminal Cartesian point back into
    # the target reference's Frenet frame.
    terminal_state = FrenetState(
        s=float(merge_path.s[-1]), s_d=float(merge_path.s_d[-1]), s_dd=float(merge_path.s_dd[-1]),
        d=float(merge_path.d[-1]), d_d=float(merge_path.d_d[-1]), d_dd=float(merge_path.d_dd[-1]),
    )
    cart = frenet_to_cartesian(terminal_state, target_ref)
    from src.planning.frenet_transform import cartesian_to_frenet
    reprojected = cartesian_to_frenet(cart.x, cart.y, cart.yaw, cart.velocity, cart.acceleration, target_ref)
    assert abs(reprojected.d) < 1e-3


def test_merge_keep_stop_produce_meaningfully_different_trajectories():
    source_xy = _straight_line(n=200)
    target_xy = source_xy.copy()
    target_xy[:, 1] += 3.5
    source_ref = ReferenceLine.from_lane_polyline(_make_polyline(source_xy, lane_id=1))
    target_ref = ReferenceLine.from_lane_polyline(_make_polyline(target_xy, lane_id=2))

    ego_x, ego_y, ego_yaw, ego_speed = 10.0, 0.0, 0.0, 15.0

    ego_frenet_source = project_cartesian_to_frame(ego_x, ego_y, ego_yaw, ego_speed, source_ref)
    ego_frenet_target = project_cartesian_to_frame(ego_x, ego_y, ego_yaw, ego_speed, target_ref)

    keep_path = generate_keep_candidate(ego_frenet_source, 15.0, HORIZON_S, DT_S)
    stop_result = generate_stop_candidate(ego_frenet_source, COMFORTABLE_DECEL, HORIZON_S, DT_S, source_ref)
    merge_path = generate_follow_or_merge_candidate(ego_frenet_target, 15.0, HORIZON_S, DT_S)

    # Convert all three to Cartesian (their own respective frames) to
    # compare physical trajectories on common ground.
    def to_cartesian_xy(path, reference):
        xs, ys = [], []
        for i in range(len(path.t)):
            state = FrenetState(
                s=float(path.s[i]), s_d=float(path.s_d[i]), s_dd=float(path.s_dd[i]),
                d=float(path.d[i]), d_d=float(path.d_d[i]), d_dd=float(path.d_dd[i]),
            )
            cart = frenet_to_cartesian(state, reference)
            xs.append(cart.x)
            ys.append(cart.y)
        return np.array(xs), np.array(ys)

    keep_x, keep_y = to_cartesian_xy(keep_path, source_ref)
    stop_x, stop_y = to_cartesian_xy(stop_result.frenet_path, source_ref)
    merge_x, merge_y = to_cartesian_xy(merge_path, target_ref)

    # STOP's longitudinal horizon is the natural stopping time, which
    # may differ in length from KEEP/MERGE's shared generic horizon_s
    # -- compare over each pair's overlapping time range only.
    n_keep_stop = min(len(keep_x), len(stop_x))
    n_keep_merge = min(len(keep_x), len(merge_x))
    n_stop_merge = min(len(stop_x), len(merge_x))
    assert not np.allclose(keep_y[:n_keep_merge], merge_y[:n_keep_merge], atol=0.5)
    assert not np.allclose(keep_x[:n_keep_stop], stop_x[:n_keep_stop], atol=0.5)
    assert not np.allclose(stop_x[:n_stop_merge], merge_x[:n_stop_merge], atol=0.5)
    # MERGE ends up laterally near the target lane (y~3.5), KEEP/STOP
    # stay near the source lane (y~0).
    assert merge_y[-1] > 3.0
    assert abs(keep_y[-1]) < 0.5
    assert abs(stop_y[-1]) < 0.5
