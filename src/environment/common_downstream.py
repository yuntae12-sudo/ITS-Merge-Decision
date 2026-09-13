"""Phase 3 Stage 3-E: common downstream / Waymax adapter.

Wires Stage 3-C's ``frenet_planner.plan`` and Stage 3-D's
``LtvMpcController.solve`` into a single entry point,
``CommonDownstream.step``, that both the FSM baseline and the PPO
policy will eventually call (Stage 3-F, NOT this stage, wires this
into ``MergeEnvironment``).

Fairness constraint (restated here because this is the module a
future caller will actually import -- see Stage 3-C's
``frenet_planner.py`` and Stage 3-0's audit for the same statement
applied one layer down): ``CommonDownstream`` has NO parameter,
attribute, or branch anywhere in its public API that carries or
inspects "which policy produced this ``BehaviorAction``". Its entire
input is ``(BehaviorAction, BehaviorObjective, ego/scene state)`` --
already-computed upstream by ``BehaviorExecutor``, not this module's
job to recompute -- and its output is a pure function of that input
(mod its own internal MPC warm-start state; see the module-level
determinism note below for how a test independently verifies this).

Conceptual flow (see this module's own docstring header comment in
the task brief for the full ASCII diagram, reproduced in
``docs/phase3/OVERNIGHT_PROGRESS.md``):

    BehaviorAction + BehaviorObjective + ego/scene state
        -> DownstreamRequest (this module's own request type)
        -> frenet_planner.plan(...)                [Stage 3-C]
        -> CartesianTrajectory, or a non-OK PlannerStatus
        -> LtvMpcController.solve(...)             [Stage 3-D]
           (only called if the planner returned OK)
        -> ControllerCommand, or a non-OK ControllerStatus
        -> DownstreamResult (this module's own result type)

Failure semantics (locked design decision for THIS stage; see task
brief and ``docs/phase3/OVERNIGHT_PROGRESS.md`` Stage 3-E entry for
the full rationale): on ANY non-OK status -- from the planner or from
the controller -- this module returns ``command=None``. It does NOT
invent a fallback command (e.g. a bounded-braking command) to keep a
downstream Waymax simulation numerically alive; that question is
explicitly deferred to Stage 3-F, which owns the decision of whether
``MergeEnvironment`` needs some physical command on a failure step to
keep ``waymax_env.step`` callable at all. Manufacturing a fallback
command here, before Stage 3-F has proven one is actually necessary,
would risk silently building exactly the kind of "quietly substituted
behavior" this repo's Stage 3-0 audit repeatedly flags as an unfair
policy-comparison risk -- so this stage deliberately stops at an
explicit, commandless failure status instead.

MPC reference-horizon construction: Stage 3-C's ``CartesianTrajectory``
places index 0 at the CURRENT ego state (``t[0] == 0.0``, verified
against ``ControllerState``'s own current-state convention), while
Stage 3-D's ``LtvMpcController.solve`` wants exactly
``mpc_config.horizon`` FUTURE reference points (one per predicted
step, not including the current state itself -- see
``ltv_mpc.py``'s own ``ReferencePoint`` docstring). This module
therefore builds the MPC reference list from trajectory indices
``[1, horizon]`` inclusive (skipping index 0), requiring the
planner's own ``trajectory_horizon_s`` to produce strictly more
samples than the MPC's ``horizon`` -- true for the shipped
``configs/phase3_downstream.yaml`` (31 planner samples at
trajectory_horizon_s=3.0s/dt=0.1s vs. an MPC horizon of 20 steps) and
checked explicitly (returns ``CONTROLLER_FAILURE`` rather than
crashing if a caller-supplied config ever violates it).
"""

import dataclasses
import enum
from typing import List, Optional

from src.control.ltv_mpc import LtvMpcController, MpcConfig
from src.control.mpc_types import ControllerCommand, ControllerState, ControllerStatus, ReferencePoint
from src.environment.behavior_action import BehaviorAction, BehaviorObjective
from src.planning.candidate_evaluator import CurrentAgentState, PlannerStatus
from src.planning.frenet_planner import (
    EgoKinematicState,
    FollowInputs,
    PlanRequest,
    PlannerConfig,
    plan,
)
from src.planning.reference import ReferenceLine


class DownstreamStatus(str, enum.Enum):
    """``CommonDownstream``'s own explicit outcome status. Wraps (does
    not duplicate the meaning of) Stage 3-C's ``PlannerStatus`` and
    Stage 3-D's ``ControllerStatus`` -- see each value's docstring for
    exactly which upstream status it corresponds to."""

    OK = "OK"

    INVALID_REFERENCE = "INVALID_REFERENCE"
    """Wraps ``PlannerStatus.INVALID_REFERENCE``: malformed reference
    geometry or a non-finite/failed Frenet projection. The MPC is
    never invoked."""

    PLANNER_INFEASIBLE = "PLANNER_INFEASIBLE"
    """Wraps ``PlannerStatus.PLANNER_INFEASIBLE``: the generated
    candidate trajectory violates a kinematic/comfort feasibility
    bound. The MPC is never invoked."""

    COLLISION_BLOCKED = "COLLISION_BLOCKED"
    """Wraps ``PlannerStatus.COLLISION_BLOCKED``: the generated
    candidate trajectory would collide with a surrounding agent. The
    MPC is never invoked."""

    CONTROLLER_FAILURE = "CONTROLLER_FAILURE"
    """Wraps EITHER of Stage 3-D's own failure statuses
    (``ControllerStatus.INVALID_INPUT`` or
    ``ControllerStatus.SOLVER_FAILURE``), or this module's own
    reference-horizon-length precondition failure (see module
    docstring). The specific upstream ``ControllerStatus`` (when
    applicable) is preserved in ``DownstreamResult.diagnostics``."""


@dataclasses.dataclass(frozen=True)
class DownstreamRequest:
    """Everything ``CommonDownstream.step`` needs for one step.
    Deliberately has NO field describing which policy (FSM/PPO)
    produced ``behavior_action``/``objective`` -- see module
    docstring's fairness constraint. Field shapes mirror
    ``frenet_planner.PlanRequest`` (planning side) plus
    ``ControllerState`` (control side); ego position/heading/speed is
    given once and reused for both, since both describe the same
    current ego state.
    """

    behavior_action: BehaviorAction
    objective: BehaviorObjective
    ego_state: EgoKinematicState
    source_reference: ReferenceLine
    target_reference: ReferenceLine
    follow_inputs: FollowInputs = dataclasses.field(default_factory=FollowInputs)
    surrounding_agents: List[CurrentAgentState] = dataclasses.field(default_factory=list)
    dt_s: Optional[float] = None
    """Overrides ``PlannerConfig.dt_s`` if provided, exactly mirroring
    ``PlanRequest.dt_s``'s own override semantics."""


@dataclasses.dataclass(frozen=True)
class DownstreamResult:
    status: DownstreamStatus
    command: Optional[ControllerCommand]
    """The physical command to apply this step. ``None`` whenever
    ``status != OK`` (see module docstring's failure-semantics
    section: no fallback command is invented at this stage)."""

    diagnostics: dict
    """Merged diagnostics: the planner's own ``PlanResult.diagnostics``
    under key ``"planner"``, plus (when the MPC was invoked) the MPC's
    own ``MpcResult`` fields (cost, solver_iterations, error) under
    key ``"controller"``."""


class CommonDownstream:
    """Stateful (owns one ``LtvMpcController`` for warm-start) common
    downstream adapter. Not thread-safe, mirroring
    ``LtvMpcController``'s own documented pattern. One instance per
    simulated agent/episode -- call ``reset()`` at episode boundaries
    (delegates directly to ``LtvMpcController.reset()``).
    """

    def __init__(self, planner_config: PlannerConfig, mpc_config: MpcConfig):
        self._planner_config = planner_config
        self._mpc_config = mpc_config
        self._controller = LtvMpcController(mpc_config)

    def reset(self) -> None:
        """Clears the MPC's warm-start state. Must be called at
        episode boundaries (see ``LtvMpcController.reset()``)."""

        self._controller.reset()

    def step(self, request: DownstreamRequest) -> DownstreamResult:
        """Executes ``request.behavior_action``/``request.objective``
        for one step: plan, then (only if planning succeeded) solve
        for a physical command.

        Same ``request`` (built from identical inputs) with a
        freshly-``reset()`` (or freshly-constructed) controller always
        produces byte-identical output -- see
        ``tests/environment/test_common_downstream.py``'s determinism
        test, which verifies this directly. The only internal state
        this method depends on is the MPC's own warm-start, which
        ``reset()``/a fresh instance clears.
        """

        plan_request = PlanRequest(
            behavior_action=request.behavior_action,
            objective=request.objective,
            ego_state=request.ego_state,
            source_reference=request.source_reference,
            target_reference=request.target_reference,
            follow_inputs=request.follow_inputs,
            surrounding_agents=request.surrounding_agents,
            dt_s=request.dt_s,
        )
        plan_result = plan(plan_request, self._planner_config)

        diagnostics = {"planner": plan_result.diagnostics}

        if plan_result.status != PlannerStatus.OK:
            status_map = {
                PlannerStatus.INVALID_REFERENCE: DownstreamStatus.INVALID_REFERENCE,
                PlannerStatus.PLANNER_INFEASIBLE: DownstreamStatus.PLANNER_INFEASIBLE,
                PlannerStatus.COLLISION_BLOCKED: DownstreamStatus.COLLISION_BLOCKED,
            }
            downstream_status = status_map.get(plan_result.status)
            if downstream_status is None:
                # Unreachable for any currently-defined PlannerStatus,
                # but never silently falls through to a fabricated OK.
                downstream_status = DownstreamStatus.INVALID_REFERENCE
                diagnostics["error"] = (
                    f"Unrecognized PlannerStatus from frenet_planner.plan: "
                    f"{plan_result.status!r}"
                )
            return DownstreamResult(status=downstream_status, command=None, diagnostics=diagnostics)

        trajectory = plan_result.cartesian_trajectory
        horizon = self._mpc_config.horizon
        # Index 0 is the CURRENT ego state (matches ControllerState
        # below); the MPC wants `horizon` FUTURE points, i.e. indices
        # 1..horizon inclusive (see module docstring).
        if len(trajectory.t) < horizon + 1:
            diagnostics["error"] = (
                f"Planner trajectory has {len(trajectory.t)} samples, "
                f"need at least {horizon + 1} (mpc_config.horizon + 1) "
                f"to build the MPC's reference horizon."
            )
            return DownstreamResult(
                status=DownstreamStatus.CONTROLLER_FAILURE, command=None, diagnostics=diagnostics,
            )

        reference = [
            ReferencePoint(
                x=float(trajectory.x[i]),
                y=float(trajectory.y[i]),
                yaw=float(trajectory.yaw[i]),
                speed=float(trajectory.velocity[i]),
            )
            for i in range(1, horizon + 1)
        ]

        controller_state = ControllerState(
            x=request.ego_state.x,
            y=request.ego_state.y,
            yaw=request.ego_state.yaw,
            speed=request.ego_state.speed_mps,
        )

        mpc_result = self._controller.solve(controller_state, reference)
        diagnostics["controller"] = {
            "status": mpc_result.status.value,
            "cost": mpc_result.cost,
            "solver_iterations": mpc_result.solver_iterations,
            "error": mpc_result.error,
        }

        if mpc_result.status != ControllerStatus.OK:
            return DownstreamResult(
                status=DownstreamStatus.CONTROLLER_FAILURE, command=None, diagnostics=diagnostics,
            )

        return DownstreamResult(status=DownstreamStatus.OK, command=mpc_result.command, diagnostics=diagnostics)
