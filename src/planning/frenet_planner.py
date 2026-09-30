"""Phase 3 Stage 3-C: common Frenet behavior-execution planner --
top-level entry point.

Wires ``candidate_generator.py`` (one trajectory per already-decided
``BehaviorAction``) and ``candidate_evaluator.py`` (explicit
feasibility/collision status) into a single ``plan`` function. This is
the "common downstream" planner both the FSM baseline and PPO policy
will eventually call (wiring into ``MergeEnvironment`` is Stage 3-F,
NOT done here) -- this module contains no policy logic of its own, only
already-fixed per-action execution semantics, exactly mirroring how
``BehaviorExecutor``/``LowLevelController`` are shared unconditionally
in Phase 2.

Hard fairness constraint (Stage 3-0 audit, restated here because this
is the module a future caller will actually import): this planner
NEVER decides whether an action should have been taken. It only
executes the ``BehaviorAction``/``BehaviorObjective`` it is handed.
MERGE is generated and evaluated exactly like the other three actions
-- if it is infeasible or would collide, the caller gets an explicit
``PlannerStatus`` back, never a silent substitution of a different
action's geometry. See ``candidate_generator.py`` and
``candidate_evaluator.py``'s own module docstrings for the specific
code proving this, and the Stage 3-C completion report's self-audit
section for a consolidated citation list.
"""

import dataclasses
import math
import time
from typing import List, Optional

import numpy as np

from src.environment.behavior_action import BehaviorAction, BehaviorObjective
from src.planning.candidate_evaluator import (
    CollisionLimits,
    CurrentAgentState,
    EvaluationResult,
    FeasibilityLimits,
    PlannerStatus,
    evaluate_candidate,
)
from src.planning.candidate_generator import (
    generate_follow_or_merge_candidate,
    generate_keep_candidate,
    generate_stop_candidate,
    project_cartesian_to_frame,
)
from src.planning.frenet_transform import frenet_to_cartesian
from src.planning.frenet_types import CartesianTrajectory, FrenetPath
from src.planning.reference import InvalidReferenceGeometryError, ReferenceLine


@dataclasses.dataclass(frozen=True)
class PlannerConfig:
    """Loaded from ``configs/downstream.yaml`` by
    ``load_planner_config`` below; also constructible directly (e.g.
    in tests) without touching the file."""

    dt_s: float
    trajectory_horizon_s: float
    comfortable_deceleration_mps2: float
    feasibility_limits: FeasibilityLimits
    collision_limits: CollisionLimits


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def load_planner_config(path: str = "configs/downstream.yaml") -> PlannerConfig:
    import yaml

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    planner = raw["planner"]
    feasibility = raw["feasibility"]
    collision = raw["collision"]

    return PlannerConfig(
        dt_s=float(planner["dt_s"]),
        trajectory_horizon_s=float(planner["trajectory_horizon_s"]),
        comfortable_deceleration_mps2=float(planner["comfortable_deceleration_mps2"]),
        feasibility_limits=FeasibilityLimits(
            max_longitudinal_accel_mps2=float(feasibility["max_longitudinal_accel_mps2"]),
            max_curvature_per_m=float(feasibility["max_curvature_per_m"]),
            max_longitudinal_jerk_mps3=float(feasibility["max_longitudinal_jerk_mps3"]),
            min_forward_progress_s_dot_mps=float(feasibility["min_forward_progress_s_dot_mps"]),
        ),
        collision_limits=CollisionLimits(
            ego_radius_m=float(collision["ego_radius_m"]),
            agent_radius_default_m=float(collision["agent_radius_default_m"]),
            collision_margin_m=float(collision["collision_margin_m"]),
        ),
    )


@dataclasses.dataclass(frozen=True)
class EgoKinematicState:
    """Current (never future/logged) ego Cartesian state -- the
    simplest representation sufficient for Stage 3-B's
    ``cartesian_to_frenet``."""

    x: float
    y: float
    yaw: float
    speed_mps: float


@dataclasses.dataclass(frozen=True)
class FollowInputs:
    """Causal gap/relative-speed inputs for FOLLOW/MERGE, sourced
    DIRECTLY from the already-causal 14D observation fields (this
    planner does not reselect a leader or re-derive gap/TTC itself --
    see module docstring). ``None`` fields mean "no lead", matching
    the observation's own ``*_present`` semantics; the caller is
    expected to have already applied ``BehaviorExecutor``'s no-lead
    fallback upstream (this planner does not special-case it, it just
    executes whatever ``BehaviorObjective.reference_speed_mps`` it is
    given)."""

    source_front_gap_m: Optional[float] = None
    source_front_relative_speed_mps: Optional[float] = None
    target_front_gap_m: Optional[float] = None
    target_front_relative_speed_mps: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class PlanRequest:
    behavior_action: BehaviorAction
    """Diagnostics only -- selects which per-action generation branch
    to run; carries no additional information beyond ``objective``
    (per the frozen ``BehaviorObjective`` contract, ``objective`` is
    the actual dispatch driver)."""

    objective: BehaviorObjective
    ego_state: EgoKinematicState
    source_reference: ReferenceLine
    target_reference: ReferenceLine
    follow_inputs: FollowInputs = dataclasses.field(default_factory=FollowInputs)
    surrounding_agents: List[CurrentAgentState] = dataclasses.field(default_factory=list)
    dt_s: Optional[float] = None
    """Overrides ``PlannerConfig.dt_s`` if provided (useful for tests
    running a shorter horizon); otherwise the config's own dt is
    used."""


@dataclasses.dataclass(frozen=True)
class PlanResult:
    status: str
    """One of ``PlannerStatus``'s four values."""

    cartesian_trajectory: Optional[CartesianTrajectory]
    frenet_path: Optional[FrenetPath]
    reference_used: Optional[ReferenceLine]
    """Which ReferenceLine the returned trajectory is expressed
    against -- source for KEEP/FOLLOW/STOP, target for MERGE."""

    diagnostics: dict


def plan(request: PlanRequest, config: PlannerConfig) -> PlanResult:
    """Generates and evaluates the ONE candidate trajectory for
    ``request.behavior_action``/``request.objective``, returning an
    explicit status (see ``PlannerStatus``) and, when ``OK``, the
    selected trajectory in both Frenet and Cartesian form.

    Per-action frame choice (see ``candidate_generator.py``'s module
    docstring for the "why"):
      - KEEP, FOLLOW, STOP: SOURCE reference frame (ego's current
        lane).
      - MERGE: TARGET reference frame directly (Stage 3-C's preferred
        approach (b): project ego's current Cartesian state into the
        target reference's own Frenet frame, rather than inventing a
        blended source/target reference).
    """

    diagnostics = {"timing_s": {}}
    dt_s = request.dt_s if request.dt_s is not None else config.dt_s
    action = request.behavior_action
    objective = request.objective
    ego = request.ego_state

    reference_lane = objective.reference_lane
    if reference_lane == "target":
        active_reference = request.target_reference
    elif reference_lane == "source":
        active_reference = request.source_reference
    else:
        return PlanResult(
            status=PlannerStatus.INVALID_REFERENCE,
            cartesian_trajectory=None, frenet_path=None, reference_used=None,
            diagnostics={
                **diagnostics,
                "error": f"Unknown objective.reference_lane: {reference_lane!r}",
            },
        )

    # --- Project current Cartesian state into the active reference's
    # Frenet frame. A failure here (e.g. a malformed/degenerate
    # reference whose construction never should have succeeded, or a
    # projection that raises) is INVALID_REFERENCE, not a crash.
    t_project_start = time.perf_counter()
    if not (
        _is_finite(ego.x) and _is_finite(ego.y)
        and _is_finite(ego.yaw) and _is_finite(ego.speed_mps)
    ):
        return PlanResult(
            status=PlannerStatus.INVALID_REFERENCE,
            cartesian_trajectory=None, frenet_path=None, reference_used=None,
            diagnostics={
                **diagnostics,
                "error": f"Non-finite ego_state input: {ego!r}",
            },
        )
    try:
        ego_frenet = project_cartesian_to_frame(
            ego.x, ego.y, ego.yaw, ego.speed_mps, active_reference,
        )
    except (InvalidReferenceGeometryError, Exception) as exc:  # noqa: BLE001
        return PlanResult(
            status=PlannerStatus.INVALID_REFERENCE,
            cartesian_trajectory=None, frenet_path=None, reference_used=None,
            diagnostics={
                **diagnostics,
                "error": f"Frenet projection failed: {type(exc).__name__}: {exc}",
            },
        )
    if not (
        _is_finite(ego_frenet.s) and _is_finite(ego_frenet.s_d)
        and _is_finite(ego_frenet.d) and _is_finite(ego_frenet.d_d)
    ):
        return PlanResult(
            status=PlannerStatus.INVALID_REFERENCE,
            cartesian_trajectory=None, frenet_path=None, reference_used=None,
            diagnostics={
                **diagnostics,
                "error": f"Frenet projection produced non-finite state: {ego_frenet!r}",
            },
        )
    diagnostics["timing_s"]["projection"] = time.perf_counter() - t_project_start
    diagnostics["reference_lane"] = reference_lane
    diagnostics["ego_frenet_state"] = {
        "s": ego_frenet.s, "s_d": ego_frenet.s_d, "s_dd": ego_frenet.s_dd,
        "d": ego_frenet.d, "d_d": ego_frenet.d_d, "d_dd": ego_frenet.d_dd,
    }
    diagnostics["reference_speed_mps"] = objective.reference_speed_mps
    diagnostics["delta_v_mps"] = objective.reference_speed_mps - ego_frenet.s_d

    # --- Generate the one candidate for this action.
    t_gen_start = time.perf_counter()
    stop_target_clamped = False
    terminal_speed_feasibility = None
    try:
        if action == BehaviorAction.KEEP:
            frenet_path, terminal_speed_feasibility = generate_keep_candidate(
                ego_frenet, objective.reference_speed_mps,
                config.trajectory_horizon_s, dt_s,
                config.feasibility_limits.max_longitudinal_accel_mps2,
            )
        elif action == BehaviorAction.FOLLOW:
            frenet_path, terminal_speed_feasibility = generate_follow_or_merge_candidate(
                ego_frenet, objective.reference_speed_mps,
                config.trajectory_horizon_s, dt_s,
                config.feasibility_limits.max_longitudinal_accel_mps2,
            )
        elif action == BehaviorAction.MERGE:
            frenet_path, terminal_speed_feasibility = generate_follow_or_merge_candidate(
                ego_frenet, objective.reference_speed_mps,
                config.trajectory_horizon_s, dt_s,
                config.feasibility_limits.max_longitudinal_accel_mps2,
            )
        elif action == BehaviorAction.STOP:
            stop_result = generate_stop_candidate(
                ego_frenet, config.comfortable_deceleration_mps2,
                config.trajectory_horizon_s, dt_s, active_reference,
            )
            frenet_path = stop_result.frenet_path
            stop_target_clamped = stop_result.stop_target_clamped
        else:
            return PlanResult(
                status=PlannerStatus.INVALID_REFERENCE,
                cartesian_trajectory=None, frenet_path=None, reference_used=None,
                diagnostics={**diagnostics, "error": f"Unknown BehaviorAction: {action!r}"},
            )
    except Exception as exc:  # noqa: BLE001 - surfaced, never a crash
        return PlanResult(
            status=PlannerStatus.INVALID_REFERENCE,
            cartesian_trajectory=None, frenet_path=None, reference_used=None,
            diagnostics={
                **diagnostics,
                "error": f"Candidate generation failed: {type(exc).__name__}: {exc}",
            },
        )
    diagnostics["timing_s"]["generation"] = time.perf_counter() - t_gen_start
    diagnostics["stop_target_clamped"] = stop_target_clamped
    if terminal_speed_feasibility is not None:
        diagnostics["terminal_speed_feasibility"] = dataclasses.asdict(terminal_speed_feasibility)

    # Diagnostic-only summary of the generated candidate's longitudinal
    # profile (never consulted by the feasibility decision below,
    # which reads frenet_path directly via evaluate_candidate).
    abs_s_dd = np.abs(frenet_path.s_dd)
    max_abs_index = int(np.argmax(abs_s_dd))
    diagnostics["generated_longitudinal_profile"] = {
        "min_s_dd_mps2": float(np.min(frenet_path.s_dd)),
        "max_s_dd_mps2": float(np.max(frenet_path.s_dd)),
        "max_abs_s_dd_mps2": float(abs_s_dd[max_abs_index]),
        "time_of_max_abs_s_dd_s": float(frenet_path.t[max_abs_index]),
        "min_s_d_mps": float(np.min(frenet_path.s_d)),
        "max_s_d_mps": float(np.max(frenet_path.s_d)),
        "terminal_s_d_mps": float(frenet_path.s_d[-1]),
    }

    # --- Evaluate: explicit feasibility + collision status.
    t_eval_start = time.perf_counter()
    evaluation: EvaluationResult = evaluate_candidate(
        frenet_path=frenet_path,
        reference=active_reference,
        limits=config.feasibility_limits,
        collision_limits=config.collision_limits,
        agents=request.surrounding_agents,
    )
    diagnostics["timing_s"]["evaluation"] = time.perf_counter() - t_eval_start
    diagnostics["checks_evaluated"] = evaluation.checks_evaluated
    diagnostics["checks_failed"] = evaluation.checks_failed
    diagnostics["feasibility_values"] = evaluation.feasibility_values
    if evaluation.reason is not None:
        diagnostics["reason"] = evaluation.reason
    if evaluation.colliding_agent_id is not None:
        diagnostics["colliding_agent_id"] = evaluation.colliding_agent_id
        diagnostics["collision_time_s"] = evaluation.collision_time_s

    if evaluation.status != PlannerStatus.OK:
        return PlanResult(
            status=evaluation.status,
            cartesian_trajectory=evaluation.cartesian_trajectory,
            frenet_path=frenet_path,
            reference_used=active_reference,
            diagnostics=diagnostics,
        )

    return PlanResult(
        status=PlannerStatus.OK,
        cartesian_trajectory=evaluation.cartesian_trajectory,
        frenet_path=frenet_path,
        reference_used=active_reference,
        diagnostics=diagnostics,
    )
