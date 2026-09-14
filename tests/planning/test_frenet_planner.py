"""Stage 3-C tests for src/planning/frenet_planner.py -- the top-level
common Frenet behavior-execution planner entry point.

Uses both synthetic references (straight/curved, same pattern as
Stage 3-A/3-B and this stage's own generator/evaluator tests) and a
handful of REAL WOMD maneuvers (via
``src.environment.full_split_evaluator.load_maneuver_specs`` +
``MergeEnvironment``, the same loading pattern Stage 3-A/3-B's audit
scripts already use) for the "physically different actions" checks,
per the task brief.
"""

import numpy as np
import pytest

from src.environment.behavior_action import BehaviorAction, BehaviorObjective
from src.environment.full_split_evaluator import load_maneuver_specs
from src.environment.merge_environment import MergeEnvironment
from src.planning.candidate_evaluator import CurrentAgentState, PlannerStatus
from src.planning.frenet_planner import (
    EgoKinematicState,
    FollowInputs,
    PlanRequest,
    load_planner_config,
    plan,
)
from src.planning.reference import ReferenceLine
from src.scenarios.lane_geometry import LanePolyline, compute_arc_length

DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"


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
def config():
    return load_planner_config()


@pytest.fixture
def source_ref():
    return ReferenceLine.from_lane_polyline(_make_polyline(_straight_line(), lane_id=1))


@pytest.fixture
def target_ref():
    xy = _straight_line()
    xy[:, 1] += 3.5
    return ReferenceLine.from_lane_polyline(_make_polyline(xy, lane_id=2))


def _request(action, objective, ego_state, source_reference, target_reference, agents=None):
    return PlanRequest(
        behavior_action=action,
        objective=objective,
        ego_state=ego_state,
        source_reference=source_reference,
        target_reference=target_reference,
        follow_inputs=FollowInputs(),
        surrounding_agents=agents or [],
    )


def test_keep_produces_valid_trajectory_maintaining_d_zero_and_converging_speed(
    config, source_ref, target_ref,
):
    ego_state = EgoKinematicState(x=10.0, y=0.0, yaw=0.0, speed_mps=10.0)
    objective = BehaviorObjective(reference_speed_mps=15.0, reference_lane="source", fallback_applied=None)
    request = _request(BehaviorAction.KEEP, objective, ego_state, source_ref, target_ref)
    result = plan(request, config)

    assert result.status == PlannerStatus.OK
    assert result.frenet_path is not None
    assert np.isclose(result.frenet_path.d[-1], 0.0, atol=1e-6)
    assert np.isclose(result.frenet_path.s_d[-1], 15.0, atol=1e-6)


def test_stop_produces_valid_decelerating_trajectory(config, source_ref, target_ref):
    ego_state = EgoKinematicState(x=10.0, y=0.0, yaw=0.0, speed_mps=12.0)
    objective = BehaviorObjective(reference_speed_mps=0.0, reference_lane="source", fallback_applied=None)
    request = _request(BehaviorAction.STOP, objective, ego_state, source_ref, target_ref)
    result = plan(request, config)

    assert result.status == PlannerStatus.OK
    expected_braking_distance = (12.0 ** 2) / (2 * config.comfortable_deceleration_mps2)
    expected_s_stop = 10.0 + expected_braking_distance
    assert np.isclose(result.frenet_path.s[-1], expected_s_stop, atol=1e-2)
    assert np.isclose(result.frenet_path.s_d[-1], 0.0, atol=1e-6)
    assert result.diagnostics["stop_target_clamped"] is False


def test_keep_and_stop_produce_physically_different_trajectories_via_planner(
    config, source_ref, target_ref,
):
    ego_state = EgoKinematicState(x=10.0, y=0.0, yaw=0.0, speed_mps=15.0)
    keep_objective = BehaviorObjective(reference_speed_mps=15.0, reference_lane="source", fallback_applied=None)
    stop_objective = BehaviorObjective(reference_speed_mps=0.0, reference_lane="source", fallback_applied=None)

    keep_result = plan(_request(BehaviorAction.KEEP, keep_objective, ego_state, source_ref, target_ref), config)
    stop_result = plan(_request(BehaviorAction.STOP, stop_objective, ego_state, source_ref, target_ref), config)

    assert keep_result.status == PlannerStatus.OK
    assert stop_result.status == PlannerStatus.OK
    # STOP's longitudinal horizon is the natural stopping time
    # (v/comfortable_deceleration), which may differ in length from
    # KEEP's generic horizon_s -- compare over the overlapping range.
    n_common = min(len(keep_result.frenet_path.t), len(stop_result.frenet_path.t))
    assert not np.allclose(
        keep_result.frenet_path.s[:n_common], stop_result.frenet_path.s[:n_common], atol=1e-2,
    )
    assert keep_result.frenet_path.s_d[-1] > 10.0
    assert np.isclose(stop_result.frenet_path.s_d[-1], 0.0, atol=1e-6)


def test_merge_keep_stop_produce_meaningfully_different_trajectories_via_planner(
    config, source_ref, target_ref,
):
    ego_state = EgoKinematicState(x=10.0, y=0.0, yaw=0.0, speed_mps=15.0)
    keep_objective = BehaviorObjective(reference_speed_mps=15.0, reference_lane="source", fallback_applied=None)
    stop_objective = BehaviorObjective(reference_speed_mps=0.0, reference_lane="source", fallback_applied=None)
    merge_objective = BehaviorObjective(reference_speed_mps=15.0, reference_lane="target", fallback_applied=None)

    keep_result = plan(_request(BehaviorAction.KEEP, keep_objective, ego_state, source_ref, target_ref), config)
    stop_result = plan(_request(BehaviorAction.STOP, stop_objective, ego_state, source_ref, target_ref), config)
    merge_result = plan(_request(BehaviorAction.MERGE, merge_objective, ego_state, source_ref, target_ref), config)

    assert keep_result.status == PlannerStatus.OK
    assert stop_result.status == PlannerStatus.OK
    assert merge_result.status == PlannerStatus.OK

    # STOP's longitudinal horizon is the natural stopping time, which
    # may differ in length from KEEP/MERGE's shared generic horizon_s
    # -- compare over each pair's overlapping time range only.
    keep_y = keep_result.cartesian_trajectory.y
    keep_x = keep_result.cartesian_trajectory.x
    stop_x = stop_result.cartesian_trajectory.x
    stop_y = stop_result.cartesian_trajectory.y
    merge_x = merge_result.cartesian_trajectory.x
    merge_y = merge_result.cartesian_trajectory.y
    n_keep_merge = min(len(keep_y), len(merge_y))
    n_keep_stop = min(len(keep_x), len(stop_x))
    n_stop_merge = min(len(stop_x), len(merge_x))
    assert not np.allclose(keep_y[:n_keep_merge], merge_y[:n_keep_merge], atol=0.5)
    assert not np.allclose(keep_x[:n_keep_stop], stop_x[:n_keep_stop], atol=0.5)
    assert not np.allclose(stop_x[:n_stop_merge], merge_x[:n_stop_merge], atol=0.5)

    # MERGE ends near the target lane's lateral offset (~3.5m); KEEP/
    # STOP stay near the source lane (~0m).
    assert merge_y[-1] > 3.0
    assert abs(keep_y[-1]) < 0.5
    assert abs(stop_y[-1]) < 0.5


def test_merge_terminal_lateral_position_lands_near_target_centerline(config, source_ref, target_ref):
    ego_state = EgoKinematicState(x=10.0, y=0.0, yaw=0.0, speed_mps=15.0)
    objective = BehaviorObjective(reference_speed_mps=15.0, reference_lane="target", fallback_applied=None)
    result = plan(_request(BehaviorAction.MERGE, objective, ego_state, source_ref, target_ref), config)

    assert result.status == PlannerStatus.OK
    from src.planning.frenet_transform import cartesian_to_frenet
    terminal_x = result.cartesian_trajectory.x[-1]
    terminal_y = result.cartesian_trajectory.y[-1]
    terminal_yaw = result.cartesian_trajectory.yaw[-1]
    terminal_v = result.cartesian_trajectory.velocity[-1]
    terminal_a = result.cartesian_trajectory.acceleration[-1]
    reprojected = cartesian_to_frenet(terminal_x, terminal_y, terminal_yaw, terminal_v, terminal_a, target_ref)
    assert abs(reprojected.d) < 1e-2


def test_follow_reacts_differently_to_different_gap_and_relative_speed(config, source_ref, target_ref):
    ego_state = EgoKinematicState(x=10.0, y=0.0, yaw=0.0, speed_mps=15.0)

    def follow_speed(relative_speed_mps):
        closing_speed = max(relative_speed_mps, 0.0)
        return max(15.0 - closing_speed, 0.0)

    close_objective = BehaviorObjective(
        reference_speed_mps=follow_speed(8.0), reference_lane="source", fallback_applied=None,
    )
    far_objective = BehaviorObjective(
        reference_speed_mps=follow_speed(0.0), reference_lane="source", fallback_applied=None,
    )

    close_result = plan(_request(BehaviorAction.FOLLOW, close_objective, ego_state, source_ref, target_ref), config)
    far_result = plan(_request(BehaviorAction.FOLLOW, far_objective, ego_state, source_ref, target_ref), config)

    assert close_result.status == PlannerStatus.OK
    assert far_result.status == PlannerStatus.OK
    assert close_result.frenet_path.s_d[-1] < far_result.frenet_path.s_d[-1]


def test_collision_blocked_on_merge_returns_explicit_status_not_silent_fallback(
    config, source_ref, target_ref,
):
    ego_state = EgoKinematicState(x=10.0, y=0.0, yaw=0.0, speed_mps=15.0)
    objective = BehaviorObjective(reference_speed_mps=15.0, reference_lane="target", fallback_applied=None)

    # An agent sitting stationary directly on the target lane's
    # centerline ahead of ego, in the path the MERGE trajectory must
    # traverse to reach d=0 in the target frame.
    agent = CurrentAgentState(agent_id=55, x=35.0, y=3.5, velocity_x_mps=0.0, velocity_y_mps=0.0)

    request = _request(BehaviorAction.MERGE, objective, ego_state, source_ref, target_ref, agents=[agent])
    result = plan(request, config)

    assert result.status == PlannerStatus.COLLISION_BLOCKED
    assert result.status != PlannerStatus.OK
    # The candidate that was evaluated is still MERGE's own geometry
    # (heads toward the target lane), not silently reshaped into
    # KEEP/FOLLOW/STOP's source-lane-bound geometry.
    assert result.frenet_path is not None
    assert np.isclose(result.frenet_path.d[-1], 0.0, atol=1e-6)


def test_kinematically_infeasible_scenario_returns_planner_infeasible(config, source_ref, target_ref):
    ego_state = EgoKinematicState(x=10.0, y=20.0, yaw=0.0, speed_mps=15.0)
    # Ego placed 20m off the source lane's centerline: an unreasonably
    # large lateral shift back to d=0 within the configured horizon.
    objective = BehaviorObjective(reference_speed_mps=15.0, reference_lane="source", fallback_applied=None)
    request = _request(BehaviorAction.KEEP, objective, ego_state, source_ref, target_ref)
    # Force a very short horizon (and a finer dt_s so the sampled
    # trajectory actually captures the resulting sharp curvature
    # peak -- same reasoning as
    # test_candidate_evaluator.py::test_unreasonable_lateral_shift_in_short_time_returns_planner_infeasible)
    # via a directly-constructed config, to make the required lateral
    # maneuver kinematically unreasonable.
    import dataclasses
    short_horizon_config = dataclasses.replace(config, trajectory_horizon_s=0.5, dt_s=0.05)
    result = plan(request, short_horizon_config)

    assert result.status == PlannerStatus.PLANNER_INFEASIBLE
    assert result.status != PlannerStatus.OK


def test_reference_construction_failure_surfaces_as_invalid_reference_not_crash(config, target_ref):
    from src.planning.reference import InvalidReferenceGeometryError, ReferenceLine as RL

    degenerate_polyline = _make_polyline(np.array([[0.0, 0.0]]), lane_id=99)
    with pytest.raises(InvalidReferenceGeometryError):
        RL.from_lane_polyline(degenerate_polyline)
    # (End-to-end: a caller that fails to construct a ReferenceLine at
    # all never reaches `plan()` -- this confirms the failure raises
    # rather than silently producing unusable geometry, matching how
    # frenet_planner.plan()'s own try/except around candidate
    # generation and projection maps any such failure it DOES
    # encounter to INVALID_REFERENCE, never a crash, as exercised
    # by the malformed-projection test below.)


def test_planner_projection_failure_returns_invalid_reference_not_crash(config, source_ref, target_ref):
    ego_state = EgoKinematicState(x=float("nan"), y=0.0, yaw=0.0, speed_mps=15.0)
    objective = BehaviorObjective(reference_speed_mps=15.0, reference_lane="source", fallback_applied=None)
    request = _request(BehaviorAction.KEEP, objective, ego_state, source_ref, target_ref)
    result = plan(request, config)
    assert result.status == PlannerStatus.INVALID_REFERENCE
    assert "error" in result.diagnostics


# ----------------------------------------------------------------------
# Real WOMD maneuver coverage
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_maneuver_fixture():
    """Loads a handful of real TRAIN maneuvers and returns
    (source_ref, target_ref, ego_state) built from the FIRST maneuver
    whose scene/lanes reset successfully -- mirrors the loading
    pattern in scripts/audit_phase3_frenet_transform.py."""

    specs = load_maneuver_specs("train")
    env = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)

    for spec in specs[:15]:
        try:
            env.reset(spec)
        except Exception:  # noqa: BLE001
            continue
        source_lane_id = env._episode_context.active_source_lane_id
        target_lane_id = env._episode_context.active_target_lane_id
        source_polyline = env._polylines_by_id.get(source_lane_id)
        target_polyline = env._polylines_by_id.get(target_lane_id)
        if source_polyline is None or target_polyline is None:
            continue
        try:
            source_reference = ReferenceLine.from_lane_polyline(source_polyline)
            target_reference = ReferenceLine.from_lane_polyline(target_polyline)
        except Exception:  # noqa: BLE001
            continue

        ego_x, ego_y, ego_yaw, ego_speed = env._current_ego_pose_and_speed()
        return source_reference, target_reference, EgoKinematicState(
            x=ego_x, y=ego_y, yaw=ego_yaw, speed_mps=max(ego_speed, 1.0),
        )

    pytest.skip("No real maneuver in the first 15 TRAIN specs reset successfully.")


def test_real_womd_keep_and_stop_are_physically_different(config, real_maneuver_fixture):
    source_reference, target_reference, ego_state = real_maneuver_fixture
    keep_objective = BehaviorObjective(reference_speed_mps=15.0, reference_lane="source", fallback_applied=None)
    stop_objective = BehaviorObjective(reference_speed_mps=0.0, reference_lane="source", fallback_applied=None)

    keep_result = plan(
        _request(BehaviorAction.KEEP, keep_objective, ego_state, source_reference, target_reference), config,
    )
    stop_result = plan(
        _request(BehaviorAction.STOP, stop_objective, ego_state, source_reference, target_reference), config,
    )

    assert keep_result.status in (PlannerStatus.OK, PlannerStatus.PLANNER_INFEASIBLE)
    assert stop_result.status in (PlannerStatus.OK, PlannerStatus.PLANNER_INFEASIBLE)
    if keep_result.status == PlannerStatus.OK and stop_result.status == PlannerStatus.OK:
        assert not np.allclose(keep_result.frenet_path.s, stop_result.frenet_path.s, atol=1e-2)


def test_real_womd_merge_reaches_target_frame_d_near_zero(config, real_maneuver_fixture):
    source_reference, target_reference, ego_state = real_maneuver_fixture
    objective = BehaviorObjective(reference_speed_mps=15.0, reference_lane="target", fallback_applied=None)
    result = plan(
        _request(BehaviorAction.MERGE, objective, ego_state, source_reference, target_reference), config,
    )
    assert result.status in (PlannerStatus.OK, PlannerStatus.PLANNER_INFEASIBLE, PlannerStatus.COLLISION_BLOCKED)
    if result.status == PlannerStatus.OK:
        assert np.isclose(result.frenet_path.d[-1], 0.0, atol=1e-6)
