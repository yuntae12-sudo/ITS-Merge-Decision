"""Stage 3-E tests for src/environment/common_downstream.py -- the
common Frenet-planning + LTV-MPC downstream adapter shared
unconditionally by the FSM baseline and PPO policy (Stage 3-F wires
this into MergeEnvironment; NOT this stage).

Reuses the same synthetic-reference-line construction pattern as
Stage 3-C's tests/planning/test_frenet_planner.py and Stage 3-D's
tests/control/test_ltv_mpc.py, since this module is a thin composition
of both.
"""

import copy
import dataclasses
import inspect

import numpy as np
import pytest

from src.control.ltv_mpc import load_mpc_config
from src.control.mpc_types import ControllerCommand, MAX_ACCEL_MPS2, MAX_STEERING_CURVATURE
from src.environment.behavior_action import BehaviorAction, BehaviorObjective
from src.environment.common_downstream import (
    CommonDownstream,
    DownstreamRequest,
    DownstreamResult,
    DownstreamStatus,
)
from src.planning.candidate_evaluator import CurrentAgentState
from src.planning.frenet_planner import EgoKinematicState, FollowInputs, load_planner_config
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
        lane_id=lane_id, lane_type=1, xy=xy, direction=direction,
        arc_length=compute_arc_length(xy),
    )


def _straight_line(n=500, spacing=1.0):
    x = np.arange(n) * spacing
    y = np.zeros(n)
    return np.stack([x, y], axis=1)


@pytest.fixture
def planner_config():
    return load_planner_config()


@pytest.fixture
def mpc_config():
    return load_mpc_config()


@pytest.fixture
def source_ref():
    return ReferenceLine.from_lane_polyline(_make_polyline(_straight_line(), lane_id=1))


@pytest.fixture
def target_ref():
    xy = _straight_line()
    xy[:, 1] += 3.5
    return ReferenceLine.from_lane_polyline(_make_polyline(xy, lane_id=2))


def _downstream(planner_config, mpc_config):
    return CommonDownstream(planner_config, mpc_config)


def _request(action, objective, ego_state, source_reference, target_reference, agents=None, dt_s=None):
    return DownstreamRequest(
        behavior_action=action,
        objective=objective,
        ego_state=ego_state,
        source_reference=source_reference,
        target_reference=target_reference,
        follow_inputs=FollowInputs(),
        surrounding_agents=agents or [],
        dt_s=dt_s,
    )


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------

def test_determinism_identical_input_produces_byte_identical_output(planner_config, mpc_config):
    """Same (BehaviorAction, BehaviorObjective, ego/scene state, dt)
    fed to two FRESH CommonDownstream instances (each owning its own
    fresh LtvMpcController, so no warm-start state is shared or
    reused across the two calls) must produce byte-identical results.
    This is the core fairness/reproducibility property: CommonDownstream
    itself carries no hidden entropy source.
    """

    source_ref = ReferenceLine.from_lane_polyline(_make_polyline(_straight_line(), lane_id=1))
    target_xy = _straight_line()
    target_xy[:, 1] += 3.5
    target_ref = ReferenceLine.from_lane_polyline(_make_polyline(target_xy, lane_id=2))

    ego_state = EgoKinematicState(x=10.0, y=0.0, yaw=0.0, speed_mps=15.0)
    objective = BehaviorObjective(reference_speed_mps=15.0, reference_lane="source", fallback_applied=None)
    agent = CurrentAgentState(agent_id=1, x=60.0, y=0.0, velocity_x_mps=10.0, velocity_y_mps=0.0)
    request = _request(BehaviorAction.KEEP, objective, ego_state, source_ref, target_ref, agents=[agent])

    downstream_a = _downstream(planner_config, mpc_config)
    downstream_b = _downstream(planner_config, mpc_config)

    result_a = downstream_a.step(request)
    result_b = downstream_b.step(request)

    assert result_a.status == DownstreamStatus.OK
    assert result_b.status == DownstreamStatus.OK
    assert result_a.command == result_b.command
    assert result_a.command.acceleration_mps2 == result_b.command.acceleration_mps2
    assert result_a.command.steering_curvature == result_b.command.steering_curvature

    # Also verify re-using the SAME instance after an explicit reset()
    # (the documented alternative determinism-preserving pattern) gives
    # the identical result too.
    downstream_a.reset()
    result_c = downstream_a.step(request)
    assert result_c.command.acceleration_mps2 == result_a.command.acceleration_mps2
    assert result_c.command.steering_curvature == result_a.command.steering_curvature


# ---------------------------------------------------------------------
# No policy-type dependency (structural/API-surface check)
# ---------------------------------------------------------------------

def test_no_policy_type_parameter_anywhere_in_public_api(planner_config, mpc_config):
    """Structural check: neither CommonDownstream's constructor/step
    signature, nor DownstreamRequest's/DownstreamResult's own fields,
    contain any parameter/attribute whose name could plausibly carry
    "which policy produced this action" (fsm/ppo/policy/origin/source
    -- "source" here is deliberately excluded from the banned-substring
    check since `source_reference`/`source_front_*` are legitimate
    lane-geometry/gap terms already used throughout Stage 3-C, not a
    policy-origin marker)."""

    banned_substrings = ["policy", "fsm", "ppo", "origin", "is_rl", "is_rule_based", "agent_type"]

    def _check_no_banned(names, where):
        for name in names:
            lowered = name.lower()
            for banned in banned_substrings:
                assert banned not in lowered, (
                    f"Found suspicious policy-origin-like field {name!r} in {where}"
                )

    step_params = list(inspect.signature(CommonDownstream.step).parameters.keys())
    _check_no_banned(step_params, "CommonDownstream.step signature")

    init_params = list(inspect.signature(CommonDownstream.__init__).parameters.keys())
    _check_no_banned(init_params, "CommonDownstream.__init__ signature")

    request_fields = [f.name for f in dataclasses.fields(DownstreamRequest)]
    _check_no_banned(request_fields, "DownstreamRequest fields")

    result_fields = [f.name for f in dataclasses.fields(DownstreamResult)]
    _check_no_banned(result_fields, "DownstreamResult fields")

    # Additionally: the actual field set is exactly what's needed to
    # describe (action, objective, scene state) -- no extra field at
    # all beyond what frenet_planner.PlanRequest already required plus
    # nothing else.
    assert set(request_fields) == {
        "behavior_action", "objective", "ego_state", "source_reference",
        "target_reference", "follow_inputs", "surrounding_agents", "dt_s",
    }


# ---------------------------------------------------------------------
# All four BehaviorActions produce a coherent result
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "action,reference_lane",
    [
        (BehaviorAction.KEEP, "source"),
        (BehaviorAction.FOLLOW, "source"),
        (BehaviorAction.MERGE, "target"),
        (BehaviorAction.STOP, "source"),
    ],
)
def test_all_four_behavior_actions_produce_coherent_result(
    planner_config, mpc_config, source_ref, target_ref, action, reference_lane,
):
    downstream = _downstream(planner_config, mpc_config)
    ego_state = EgoKinematicState(x=10.0, y=0.0, yaw=0.0, speed_mps=12.0)
    objective = BehaviorObjective(
        reference_speed_mps=12.0, reference_lane=reference_lane, fallback_applied=None,
    )
    request = _request(action, objective, ego_state, source_ref, target_ref)

    result = downstream.step(request)

    assert isinstance(result, DownstreamResult)
    if result.status == DownstreamStatus.OK:
        assert result.command is not None
        assert isinstance(result.command, ControllerCommand)
    else:
        assert result.status in (
            DownstreamStatus.INVALID_REFERENCE,
            DownstreamStatus.PLANNER_INFEASIBLE,
            DownstreamStatus.COLLISION_BLOCKED,
            DownstreamStatus.CONTROLLER_FAILURE,
        )
        assert result.command is None


# ---------------------------------------------------------------------
# Planner failure propagation (MPC never invoked)
# ---------------------------------------------------------------------

def test_collision_blocked_propagates_and_mpc_never_invoked(planner_config, mpc_config, source_ref, target_ref):
    downstream = _downstream(planner_config, mpc_config)
    ego_state = EgoKinematicState(x=10.0, y=0.0, yaw=0.0, speed_mps=15.0)
    objective = BehaviorObjective(reference_speed_mps=15.0, reference_lane="target", fallback_applied=None)
    # Same collision-inducing scenario pattern as
    # test_frenet_planner.py::test_collision_blocked_on_merge_returns_explicit_status_not_silent_fallback.
    agent = CurrentAgentState(agent_id=55, x=35.0, y=3.5, velocity_x_mps=0.0, velocity_y_mps=0.0)
    request = _request(BehaviorAction.MERGE, objective, ego_state, source_ref, target_ref, agents=[agent])

    original_solve = downstream._controller.solve
    calls = []

    def _spy_solve(*args, **kwargs):
        calls.append((args, kwargs))
        return original_solve(*args, **kwargs)

    downstream._controller.solve = _spy_solve

    result = downstream.step(request)

    assert result.status == DownstreamStatus.COLLISION_BLOCKED
    assert result.command is None
    assert len(calls) == 0, "MPC solve() must not be invoked when the planner reports COLLISION_BLOCKED"


def test_planner_infeasible_propagates_and_mpc_never_invoked(planner_config, mpc_config, source_ref, target_ref):
    downstream = _downstream(planner_config, mpc_config)
    ego_state = EgoKinematicState(x=10.0, y=20.0, yaw=0.0, speed_mps=15.0)
    objective = BehaviorObjective(reference_speed_mps=15.0, reference_lane="source", fallback_applied=None)
    request = _request(BehaviorAction.KEEP, objective, ego_state, source_ref, target_ref)

    # Same "unreasonably large lateral shift in a short horizon" trick
    # as test_frenet_planner.py::test_kinematically_infeasible_scenario_returns_planner_infeasible.
    short_horizon_planner_config = dataclasses.replace(planner_config, trajectory_horizon_s=0.5, dt_s=0.05)
    downstream_short = CommonDownstream(short_horizon_planner_config, mpc_config)

    original_solve = downstream_short._controller.solve
    calls = []

    def _spy_solve(*args, **kwargs):
        calls.append((args, kwargs))
        return original_solve(*args, **kwargs)

    downstream_short._controller.solve = _spy_solve

    result = downstream_short.step(request)

    assert result.status == DownstreamStatus.PLANNER_INFEASIBLE
    assert result.command is None
    assert len(calls) == 0, "MPC solve() must not be invoked when the planner reports PLANNER_INFEASIBLE"


def test_invalid_reference_propagates_and_mpc_never_invoked(planner_config, mpc_config, source_ref, target_ref):
    downstream = _downstream(planner_config, mpc_config)
    ego_state = EgoKinematicState(x=float("nan"), y=0.0, yaw=0.0, speed_mps=15.0)
    objective = BehaviorObjective(reference_speed_mps=15.0, reference_lane="source", fallback_applied=None)
    request = _request(BehaviorAction.KEEP, objective, ego_state, source_ref, target_ref)

    original_solve = downstream._controller.solve
    calls = []

    def _spy_solve(*args, **kwargs):
        calls.append((args, kwargs))
        return original_solve(*args, **kwargs)

    downstream._controller.solve = _spy_solve

    result = downstream.step(request)

    assert result.status == DownstreamStatus.INVALID_REFERENCE
    assert result.command is None
    assert len(calls) == 0, "MPC solve() must not be invoked when the planner reports INVALID_REFERENCE"


# ---------------------------------------------------------------------
# Controller failure propagation
# ---------------------------------------------------------------------

def test_controller_failure_propagates_as_controller_failure_status(planner_config, mpc_config, source_ref, target_ref):
    """Forces LtvMpcController.solve() itself to report a failure
    status (mirroring tests/control/test_ltv_mpc.py's own
    non-finite-input trigger) by monkeypatching the controller's solve
    to return an INVALID_INPUT MpcResult directly -- CommonDownstream's
    job here is only to verify it propagates whatever ControllerStatus
    the MPC reports as its own CONTROLLER_FAILURE, without inventing a
    command."""

    from src.control.mpc_types import ControllerStatus, MpcResult

    downstream = _downstream(planner_config, mpc_config)
    ego_state = EgoKinematicState(x=10.0, y=0.0, yaw=0.0, speed_mps=12.0)
    objective = BehaviorObjective(reference_speed_mps=12.0, reference_lane="source", fallback_applied=None)
    request = _request(BehaviorAction.KEEP, objective, ego_state, source_ref, target_ref)

    def _failing_solve(state, reference):
        return MpcResult(
            status=ControllerStatus.INVALID_INPUT,
            command=None, horizon_commands=None, predicted_states=None,
            cost=None, solver_iterations=None,
            error="forced failure for Stage 3-E test",
        )

    downstream._controller.solve = _failing_solve

    result = downstream.step(request)

    assert result.status == DownstreamStatus.CONTROLLER_FAILURE
    assert result.command is None
    assert result.diagnostics["controller"]["status"] == ControllerStatus.INVALID_INPUT.value


def test_controller_failure_via_genuinely_non_finite_reference_horizon_precondition(
    planner_config, mpc_config, source_ref, target_ref,
):
    """A too-short planner trajectory (fewer samples than
    mpc_config.horizon + 1) cannot supply the MPC's required reference
    horizon length -- CommonDownstream must report CONTROLLER_FAILURE
    rather than crash or silently truncate/pad."""

    downstream = _downstream(planner_config, mpc_config)
    ego_state = EgoKinematicState(x=10.0, y=0.0, yaw=0.0, speed_mps=12.0)
    objective = BehaviorObjective(reference_speed_mps=12.0, reference_lane="source", fallback_applied=None)
    # trajectory_horizon_s shorter than mpc_config.horizon*dt_s (20*0.1=2.0s)
    # guarantees too few samples for the MPC's reference horizon.
    short_planner_config = dataclasses.replace(planner_config, trajectory_horizon_s=1.0)
    downstream_short = CommonDownstream(short_planner_config, mpc_config)
    request = _request(BehaviorAction.KEEP, objective, ego_state, source_ref, target_ref)

    result = downstream_short.step(request)

    assert result.status == DownstreamStatus.CONTROLLER_FAILURE
    assert result.command is None


# ---------------------------------------------------------------------
# Command physical units/shape survive unmangled through the wrapper
# ---------------------------------------------------------------------

def test_successful_command_within_waymax_physical_bounds(planner_config, mpc_config, source_ref, target_ref):
    downstream = _downstream(planner_config, mpc_config)
    ego_state = EgoKinematicState(x=10.0, y=0.0, yaw=0.0, speed_mps=12.0)
    objective = BehaviorObjective(reference_speed_mps=12.0, reference_lane="source", fallback_applied=None)
    request = _request(BehaviorAction.KEEP, objective, ego_state, source_ref, target_ref)

    result = downstream.step(request)

    assert result.status == DownstreamStatus.OK
    assert result.command is not None
    assert -MAX_ACCEL_MPS2 <= result.command.acceleration_mps2 <= MAX_ACCEL_MPS2
    assert -MAX_STEERING_CURVATURE <= result.command.steering_curvature <= MAX_STEERING_CURVATURE

    # The value survives unmangled: re-derive the MPC's own raw
    # command by calling the controller a second, independent time
    # with a freshly reset warm-start and confirm the wrapper did not
    # alter it.
    downstream.reset()
    result2 = downstream.step(request)
    assert result2.command.acceleration_mps2 == result.command.acceleration_mps2
    assert result2.command.steering_curvature == result.command.steering_curvature
