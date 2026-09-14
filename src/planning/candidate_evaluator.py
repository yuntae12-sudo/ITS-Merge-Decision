"""Phase 3 Stage 3-C: explicit feasibility/collision evaluation of one
generated candidate ``FrenetPath``.

This module NEVER decides whether a BehaviorAction should have been
chosen, and never silently substitutes a different action's geometry
when a candidate fails a check. It only classifies one already
-generated candidate into an explicit status:

  - OK: passed every check below.
  - PLANNER_INFEASIBLE: violates a kinematic/curvature/jerk bound.
  - COLLISION_BLOCKED: overlaps a constant-velocity-predicted current
    agent position at a matching time.
  - INVALID_REFERENCE: the reference line itself could not be used
    (e.g. non-finite candidate output caused by a degenerate
    reference -- surfaced explicitly rather than crashing).

Checks performed (all thresholds from ``configs/phase3_downstream.yaml``,
see that file for the justification of each number):
  1. Longitudinal acceleration bound: |s_dd(t)| <= max_longitudinal_accel_mps2.
  2. Non-negative forward progress: s_d(t) >= min_forward_progress_s_dot_mps.
  3. Curvature bound: |curvature(t)| <= max_curvature_per_m, computed by
     converting the candidate to Cartesian via Stage 3-B's
     ``frenet_to_cartesian`` (the same transform the candidate will
     actually be executed through).
  4. Longitudinal jerk bound: |s_ddd(t)| <= max_longitudinal_jerk_mps3.
  5. Collision: current-position + constant-velocity prediction only,
     circular-proxy overlap check (see ``check_collision`` docstring
     for the full simplification rationale).

Collision-check simplification (Stage 3-C scope, explicitly documented
per the task brief): both ego and every surrounding agent are modeled
as CIRCLES (not oriented rectangles), and future agent position is
predicted via CURRENT position + CURRENT velocity * elapsed time only
(no logged/future trajectory is ever read for any agent, ego or
otherwise -- prediction is purely a function of the CURRENT-frame
inputs the caller supplies). This is a conservative, fast proxy
sufficient for Stage 3-C's correctness bar (the task brief explicitly
permits "a reasonably conservative circular or axis-aligned
approximation"); it is not a full oriented-bounding-box/SAT polygon
check, and does not need to be for this stage -- a tighter geometric
model can be layered in later without changing this module's status
contract (OK/PLANNER_INFEASIBLE/COLLISION_BLOCKED/INVALID_REFERENCE).
"""

import dataclasses
from typing import List, Optional

import numpy as np

from src.planning.frenet_transform import frenet_to_cartesian
from src.planning.frenet_types import CartesianTrajectory, FrenetPath, FrenetState
from src.planning.reference import ReferenceLine


class PlannerStatus:
    """Explicit evaluation-outcome statuses (Stage 3-C requirement:
    infeasibility/collision must be reported, never silently hidden or
    converted into a different action's geometry)."""

    OK = "OK"
    PLANNER_INFEASIBLE = "PLANNER_INFEASIBLE"
    COLLISION_BLOCKED = "COLLISION_BLOCKED"
    INVALID_REFERENCE = "INVALID_REFERENCE"


@dataclasses.dataclass(frozen=True)
class FeasibilityLimits:
    max_longitudinal_accel_mps2: float
    max_curvature_per_m: float
    max_longitudinal_jerk_mps3: float
    min_forward_progress_s_dot_mps: float


@dataclasses.dataclass(frozen=True)
class CollisionLimits:
    ego_radius_m: float
    agent_radius_default_m: float
    collision_margin_m: float


@dataclasses.dataclass(frozen=True)
class CurrentAgentState:
    """One surrounding agent's CURRENT (never future/logged) state,
    used only for the constant-velocity collision prediction."""

    agent_id: int
    x: float
    y: float
    velocity_x_mps: float
    velocity_y_mps: float
    radius_m: Optional[float] = None  # falls back to agent_radius_default_m


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    status: str
    reason: Optional[str] = None
    checks_evaluated: tuple = ()
    checks_failed: tuple = ()
    cartesian_trajectory: Optional[CartesianTrajectory] = None
    colliding_agent_id: Optional[int] = None
    collision_time_s: Optional[float] = None


def _frenet_path_to_cartesian(
    frenet_path: FrenetPath, reference: ReferenceLine
) -> CartesianTrajectory:
    n = frenet_path.t.shape[0]
    x = np.empty(n)
    y = np.empty(n)
    yaw = np.empty(n)
    curvature = np.empty(n)
    velocity = np.empty(n)
    acceleration = np.empty(n)

    for i in range(n):
        state = FrenetState(
            s=float(frenet_path.s[i]), s_d=float(frenet_path.s_d[i]),
            s_dd=float(frenet_path.s_dd[i]), d=float(frenet_path.d[i]),
            d_d=float(frenet_path.d_d[i]), d_dd=float(frenet_path.d_dd[i]),
        )
        cart = frenet_to_cartesian(state, reference)
        x[i], y[i] = cart.x, cart.y
        yaw[i] = cart.yaw
        curvature[i] = cart.curvature
        velocity[i] = cart.velocity
        acceleration[i] = cart.acceleration

    return CartesianTrajectory(
        t=frenet_path.t.copy(), x=x, y=y, yaw=yaw,
        curvature=curvature, velocity=velocity, acceleration=acceleration,
    )


def check_collision(
    cartesian_trajectory: CartesianTrajectory,
    agents: List[CurrentAgentState],
    limits: CollisionLimits,
) -> Optional[tuple]:
    """Constant-velocity, circular-proxy collision check (see module
    docstring for the full simplification rationale). Returns
    ``(agent_id, collision_time_s)`` for the FIRST (earliest-time)
    colliding agent found, or ``None`` if no collision is predicted.
    """

    for i, t in enumerate(cartesian_trajectory.t):
        ego_x, ego_y = cartesian_trajectory.x[i], cartesian_trajectory.y[i]
        for agent in agents:
            predicted_x = agent.x + agent.velocity_x_mps * t
            predicted_y = agent.y + agent.velocity_y_mps * t
            agent_radius = (
                agent.radius_m if agent.radius_m is not None
                else limits.agent_radius_default_m
            )
            distance = np.hypot(ego_x - predicted_x, ego_y - predicted_y)
            threshold = limits.ego_radius_m + agent_radius + limits.collision_margin_m
            if distance < threshold:
                return agent.agent_id, float(t)

    return None


def evaluate_candidate(
    frenet_path: FrenetPath,
    reference: ReferenceLine,
    limits: FeasibilityLimits,
    collision_limits: CollisionLimits,
    agents: Optional[List[CurrentAgentState]] = None,
) -> EvaluationResult:
    """Evaluates one candidate ``FrenetPath`` against every check
    listed in the module docstring, returning an explicit
    ``EvaluationResult``. Never mutates or reshapes the candidate --
    only classifies it."""

    agents = agents or []
    checks_evaluated = []
    checks_failed = []

    # --- Reference/geometry validity: convert to Cartesian first,
    # since curvature feasibility and collision both need it. Any
    # non-finite output here means the reference/candidate combination
    # itself is invalid geometry, not a kinematic infeasibility.
    checks_evaluated.append("reference_conversion")
    try:
        cartesian_trajectory = _frenet_path_to_cartesian(frenet_path, reference)
    except Exception as exc:  # noqa: BLE001 - surface as INVALID_REFERENCE, never crash
        checks_failed.append("reference_conversion")
        return EvaluationResult(
            status=PlannerStatus.INVALID_REFERENCE,
            reason=f"Cartesian conversion failed: {type(exc).__name__}: {exc}",
            checks_evaluated=tuple(checks_evaluated),
            checks_failed=tuple(checks_failed),
        )

    if not (
        np.all(np.isfinite(cartesian_trajectory.x))
        and np.all(np.isfinite(cartesian_trajectory.y))
        and np.all(np.isfinite(cartesian_trajectory.yaw))
        and np.all(np.isfinite(cartesian_trajectory.curvature))
    ):
        checks_failed.append("reference_conversion")
        return EvaluationResult(
            status=PlannerStatus.INVALID_REFERENCE,
            reason="Cartesian conversion produced non-finite output.",
            checks_evaluated=tuple(checks_evaluated),
            checks_failed=tuple(checks_failed),
            cartesian_trajectory=cartesian_trajectory,
        )

    # --- Kinematic/curvature/jerk feasibility checks.
    checks_evaluated.append("longitudinal_accel")
    if np.any(np.abs(frenet_path.s_dd) > limits.max_longitudinal_accel_mps2):
        checks_failed.append("longitudinal_accel")

    checks_evaluated.append("forward_progress")
    if np.any(frenet_path.s_d < limits.min_forward_progress_s_dot_mps):
        checks_failed.append("forward_progress")

    checks_evaluated.append("curvature")
    if np.any(np.abs(cartesian_trajectory.curvature) > limits.max_curvature_per_m):
        checks_failed.append("curvature")

    checks_evaluated.append("longitudinal_jerk")
    if frenet_path.s_ddd is not None and np.any(
        np.abs(frenet_path.s_ddd) > limits.max_longitudinal_jerk_mps3
    ):
        checks_failed.append("longitudinal_jerk")

    if checks_failed:
        return EvaluationResult(
            status=PlannerStatus.PLANNER_INFEASIBLE,
            reason=f"Failed checks: {checks_failed}",
            checks_evaluated=tuple(checks_evaluated),
            checks_failed=tuple(checks_failed),
            cartesian_trajectory=cartesian_trajectory,
        )

    # --- Collision check (only reached if kinematically feasible;
    # order does not change the OUTCOME classification of a candidate
    # that fails both -- it is simply reported as PLANNER_INFEASIBLE
    # first since that was already known -- but does mean a candidate
    # that is kinematically fine and only fails collision is reported
    # as exactly COLLISION_BLOCKED, never conflated with infeasibility).
    checks_evaluated.append("collision")
    collision = check_collision(cartesian_trajectory, agents, collision_limits)
    if collision is not None:
        agent_id, collision_time_s = collision
        checks_failed.append("collision")
        return EvaluationResult(
            status=PlannerStatus.COLLISION_BLOCKED,
            reason=f"Predicted collision with agent {agent_id} at t={collision_time_s:.2f}s",
            checks_evaluated=tuple(checks_evaluated),
            checks_failed=tuple(checks_failed),
            cartesian_trajectory=cartesian_trajectory,
            colliding_agent_id=agent_id,
            collision_time_s=collision_time_s,
        )

    return EvaluationResult(
        status=PlannerStatus.OK,
        checks_evaluated=tuple(checks_evaluated),
        checks_failed=(),
        cartesian_trajectory=cartesian_trajectory,
    )
