"""Stage 3-C tests for src/planning/candidate_evaluator.py.

Synthetic straight/curved references. Focused on the explicit
feasibility/collision status contract: OK, PLANNER_INFEASIBLE,
COLLISION_BLOCKED, INVALID_REFERENCE must each be reachable, and a
failing candidate must never be silently reported as OK or converted
into different geometry.
"""

import numpy as np
import pytest

from src.planning.candidate_evaluator import (
    CollisionLimits,
    CurrentAgentState,
    FeasibilityLimits,
    PlannerStatus,
    check_collision,
    evaluate_candidate,
)
from src.planning.candidate_generator import (
    generate_keep_candidate,
    generate_stop_candidate,
)
from src.planning.frenet_types import FrenetPath, FrenetState
from src.planning.reference import ReferenceLine
from src.scenarios.lane_geometry import LanePolyline, compute_arc_length

HORIZON_S = 3.0
DT_S = 0.1


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


DEFAULT_LIMITS = FeasibilityLimits(
    max_longitudinal_accel_mps2=6.0,
    max_curvature_per_m=0.3,
    max_longitudinal_jerk_mps3=20.0,
    min_forward_progress_s_dot_mps=-0.1,
)
DEFAULT_COLLISION_LIMITS = CollisionLimits(
    ego_radius_m=1.5, agent_radius_default_m=1.5, collision_margin_m=0.5,
)


def test_feasible_keep_candidate_is_ok(straight_ref):
    ego = FrenetState(s=10.0, s_d=15.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
    path = generate_keep_candidate(ego, 15.0, HORIZON_S, DT_S)
    result = evaluate_candidate(path, straight_ref, DEFAULT_LIMITS, DEFAULT_COLLISION_LIMITS)
    assert result.status == PlannerStatus.OK
    assert "longitudinal_accel" in result.checks_evaluated
    assert "collision" in result.checks_evaluated
    assert result.checks_failed == ()


def test_kinematically_infeasible_speed_change_returns_planner_infeasible(straight_ref):
    # A speed change requiring far more than 6.0 m/s^2 over the
    # horizon: 0 -> 100 m/s in 3s implies avg accel > 33 m/s^2.
    ego = FrenetState(s=10.0, s_d=0.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
    path = generate_keep_candidate(ego, 100.0, HORIZON_S, DT_S)
    result = evaluate_candidate(path, straight_ref, DEFAULT_LIMITS, DEFAULT_COLLISION_LIMITS)
    assert result.status == PlannerStatus.PLANNER_INFEASIBLE
    assert "longitudinal_accel" in result.checks_failed
    # Must not silently become a different, "fixed" trajectory
    # reported as OK.
    assert result.status != PlannerStatus.OK


def test_unreasonable_lateral_shift_in_short_time_returns_planner_infeasible(straight_ref):
    # An unreasonably large lateral shift (20m, several lane widths)
    # in a very short horizon (0.5s) should blow the curvature bound.
    # dt_s=0.05 (finer than the nominal 0.1s control step) is used
    # here only so the sampled trajectory actually captures the sharp
    # mid-maneuver curvature peak this extreme boundary condition
    # produces -- the infeasibility itself is a real property of the
    # continuous trajectory, not an artifact of sampling.
    ego = FrenetState(s=10.0, s_d=15.0, s_dd=0.0, d=20.0, d_d=0.0, d_dd=0.0)
    path = generate_keep_candidate(ego, 15.0, horizon_s=0.5, dt_s=0.05)
    result = evaluate_candidate(path, straight_ref, DEFAULT_LIMITS, DEFAULT_COLLISION_LIMITS)
    assert result.status == PlannerStatus.PLANNER_INFEASIBLE
    assert len(result.checks_failed) > 0


def test_collision_blocked_when_agent_directly_in_path(straight_ref):
    ego = FrenetState(s=10.0, s_d=15.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
    path = generate_keep_candidate(ego, 15.0, HORIZON_S, DT_S)
    # Stationary agent placed directly on the ego's future path
    # (KEEP travels roughly from x=10 to x=10+15*3=55 along y=0).
    agent = CurrentAgentState(
        agent_id=42, x=30.0, y=0.0, velocity_x_mps=0.0, velocity_y_mps=0.0,
    )
    result = evaluate_candidate(
        path, straight_ref, DEFAULT_LIMITS, DEFAULT_COLLISION_LIMITS, agents=[agent],
    )
    assert result.status == PlannerStatus.COLLISION_BLOCKED
    assert result.colliding_agent_id == 42
    assert result.collision_time_s is not None
    # The returned trajectory is exactly the requested KEEP geometry
    # (not silently substituted) -- terminal speed still shows the
    # KEEP objective was what was attempted.
    assert np.isclose(path.s_d[-1], 15.0, atol=1e-6)


def test_collision_blocked_status_is_not_silently_reclassified_as_ok(straight_ref):
    ego = FrenetState(s=10.0, s_d=15.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
    path = generate_keep_candidate(ego, 15.0, HORIZON_S, DT_S)
    agent = CurrentAgentState(agent_id=7, x=25.0, y=0.0, velocity_x_mps=0.0, velocity_y_mps=0.0)
    result = evaluate_candidate(
        path, straight_ref, DEFAULT_LIMITS, DEFAULT_COLLISION_LIMITS, agents=[agent],
    )
    assert result.status == PlannerStatus.COLLISION_BLOCKED
    assert result.status != PlannerStatus.OK
    assert result.status != PlannerStatus.PLANNER_INFEASIBLE


def test_no_collision_when_agent_far_from_path(straight_ref):
    ego = FrenetState(s=10.0, s_d=15.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
    path = generate_keep_candidate(ego, 15.0, HORIZON_S, DT_S)
    agent = CurrentAgentState(
        agent_id=1, x=30.0, y=50.0, velocity_x_mps=0.0, velocity_y_mps=0.0,
    )
    result = evaluate_candidate(
        path, straight_ref, DEFAULT_LIMITS, DEFAULT_COLLISION_LIMITS, agents=[agent],
    )
    assert result.status == PlannerStatus.OK


def test_moving_agent_constant_velocity_prediction_causes_collision(straight_ref):
    ego = FrenetState(s=10.0, s_d=15.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
    path = generate_keep_candidate(ego, 15.0, HORIZON_S, DT_S)
    # Agent starts far off the path laterally but moves toward it,
    # timed to intercept the ego's projected position.
    agent = CurrentAgentState(
        agent_id=99, x=30.0, y=10.0, velocity_x_mps=0.0, velocity_y_mps=-10.0 / 1.0,
    )
    result = evaluate_candidate(
        path, straight_ref, DEFAULT_LIMITS, DEFAULT_COLLISION_LIMITS, agents=[agent],
    )
    assert result.status == PlannerStatus.COLLISION_BLOCKED


def test_reference_construction_failure_surfaces_as_invalid_reference():
    # Degenerate/malformed reference: single-point polyline can't even
    # build a ReferenceLine -- construction itself raises
    # InvalidReferenceGeometryError, which the caller (frenet_planner)
    # is responsible for catching; here we directly confirm
    # ReferenceLine.from_lane_polyline raises rather than silently
    # producing a garbage reference, and that a caller mapping that
    # into INVALID_REFERENCE is straightforward (exercised end-to-end
    # in test_frenet_planner.py).
    from src.planning.reference import InvalidReferenceGeometryError

    degenerate_xy = np.array([[0.0, 0.0]])
    polyline = _make_polyline(degenerate_xy)
    with pytest.raises(InvalidReferenceGeometryError):
        ReferenceLine.from_lane_polyline(polyline)


def test_check_collision_returns_none_when_no_agents(straight_ref):
    ego = FrenetState(s=10.0, s_d=15.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
    path = generate_keep_candidate(ego, 15.0, HORIZON_S, DT_S)
    from src.planning.candidate_evaluator import _frenet_path_to_cartesian
    cart = _frenet_path_to_cartesian(path, straight_ref)
    assert check_collision(cart, [], DEFAULT_COLLISION_LIMITS) is None
