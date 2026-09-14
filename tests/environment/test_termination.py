"""Tests for src/environment/termination.py -- the online-causal
merge-success detector and SUCCESS/FAILURE/TRUNCATION termination
logic (Stage B-0 Section 6/8)."""

import numpy as np
import pytest

from src.environment.termination import (
    MAX_EPISODE_HORIZON_FRAMES,
    TerminationInputs,
    TerminationReason,
    check_online_causal_merge_success,
    check_termination,
)
from src.scenarios.lane_assignment import LaneAssignmentConfig
from src.scenarios.lane_geometry import LanePolyline, compute_arc_length

DEFAULT_CONFIG = LaneAssignmentConfig(
    max_lateral_distance_m=2.0,
    max_heading_difference_deg=45.0,
    persistence_frames=3,
)


def _make_straight_lane(lane_id: int, y: float, x_range=(0.0, 100.0)):
    xs = np.linspace(x_range[0], x_range[1], 50)
    xy = np.stack([xs, np.full_like(xs, y)], axis=1)
    direction = np.tile([1.0, 0.0], (xy.shape[0], 1))
    return LanePolyline(
        lane_id=lane_id,
        lane_type=2,
        xy=xy,
        direction=direction,
        arc_length=compute_arc_length(xy),
    )


# ======================================================================
# check_online_causal_merge_success
# ======================================================================


def test_success_when_stably_on_target_lane_with_progress():
    """Ego moves along the source lane then steps onto the target
    lane and holds there for >= persistence_frames with forward
    progress -- must report success."""

    source = _make_straight_lane(1, y=0.0)
    target = _make_straight_lane(2, y=3.5)  # > max_lateral_distance_m apart

    # Frames 0-2 on source lane (y=0), frames 3-6 on target lane
    # (y=3.5) moving forward -- 4 consecutive frames satisfies
    # persistence_frames=3.
    xs = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    ys = [0.0, 0.0, 0.0, 3.5, 3.5, 3.5, 3.5]

    ego_x = np.array(xs)
    ego_y = np.array(ys)
    ego_yaw = np.zeros_like(ego_x)
    ego_valid = np.ones_like(ego_x, dtype=bool)

    result = check_online_causal_merge_success(
        ego_x, ego_y, ego_yaw, ego_valid,
        target_lane_id=2,
        polylines=[source, target],
        lane_assignment_config=DEFAULT_CONFIG,
        episode_start_frame=0,
    )

    assert result is True


def test_no_success_before_persistence_frames_satisfied():
    """Only 2 consecutive frames on the target lane with
    persistence_frames=3 must NOT report success yet -- the stable
    sequence should not have switched."""

    source = _make_straight_lane(1, y=0.0)
    target = _make_straight_lane(2, y=3.5)

    ego_x = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    ego_y = np.array([0.0, 0.0, 0.0, 3.5, 3.5])  # only 2 frames on target
    ego_yaw = np.zeros_like(ego_x)
    ego_valid = np.ones_like(ego_x, dtype=bool)

    result = check_online_causal_merge_success(
        ego_x, ego_y, ego_yaw, ego_valid,
        target_lane_id=2,
        polylines=[source, target],
        lane_assignment_config=DEFAULT_CONFIG,
        episode_start_frame=0,
    )

    assert result is False


def test_no_success_while_still_on_source_lane():

    source = _make_straight_lane(1, y=0.0)
    target = _make_straight_lane(2, y=3.5)

    ego_x = np.array([10.0, 15.0, 20.0])
    ego_y = np.array([0.0, 0.0, 0.0])
    ego_yaw = np.zeros_like(ego_x)
    ego_valid = np.ones_like(ego_x, dtype=bool)

    result = check_online_causal_merge_success(
        ego_x, ego_y, ego_yaw, ego_valid,
        target_lane_id=2,
        polylines=[source, target],
        lane_assignment_config=DEFAULT_CONFIG,
        episode_start_frame=0,
    )

    assert result is False


def test_no_success_without_forward_progress_on_target():
    """A stable target-lane assignment with NO net longitudinal
    progress since first reaching it (e.g. momentary lateral wobble
    that snaps back without moving forward) must not count as
    success -- Stage A's explicit "no false completion from a wobble"
    requirement."""

    source = _make_straight_lane(1, y=0.0)
    target = _make_straight_lane(2, y=3.5)

    # Ego reaches the target lane and then does NOT move forward
    # (constant x) for the remaining frames -- current_arc_length ==
    # first_arc_length, not strictly greater.
    ego_x = np.array([10.0, 15.0, 20.0, 20.0, 20.0, 20.0])
    ego_y = np.array([0.0, 0.0, 3.5, 3.5, 3.5, 3.5])
    ego_yaw = np.zeros_like(ego_x)
    ego_valid = np.ones_like(ego_x, dtype=bool)

    result = check_online_causal_merge_success(
        ego_x, ego_y, ego_yaw, ego_valid,
        target_lane_id=2,
        polylines=[source, target],
        lane_assignment_config=DEFAULT_CONFIG,
        episode_start_frame=0,
    )

    assert result is False


def test_causal_function_only_uses_provided_frames():
    """Sanity/documentation check: calling the function with a
    prefix of frames must not raise or require any 'future' argument
    -- it is a pure function of the frames given, proving it can be
    called incrementally each simulation step."""

    source = _make_straight_lane(1, y=0.0)
    target = _make_straight_lane(2, y=3.5)

    full_x = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    full_y = np.array([0.0, 0.0, 0.0, 3.5, 3.5, 3.5, 3.5])
    full_yaw = np.zeros_like(full_x)
    full_valid = np.ones_like(full_x, dtype=bool)

    # Call incrementally on growing prefixes -- must never error, and
    # the final prefix's result must match calling on the full array.
    results = []
    for t in range(1, len(full_x) + 1):
        results.append(
            check_online_causal_merge_success(
                full_x[:t], full_y[:t], full_yaw[:t], full_valid[:t],
                target_lane_id=2,
                polylines=[source, target],
                lane_assignment_config=DEFAULT_CONFIG,
                episode_start_frame=0,
            )
        )

    assert results[-1] == check_online_causal_merge_success(
        full_x, full_y, full_yaw, full_valid,
        target_lane_id=2,
        polylines=[source, target],
        lane_assignment_config=DEFAULT_CONFIG,
        episode_start_frame=0,
    )


# ======================================================================
# check_termination
# ======================================================================


def test_termination_none_when_nothing_triggered():

    result = check_termination(
        TerminationInputs(
            success=False,
            collision=False,
            offroad=False,
            steps_elapsed=10,
            max_episode_horizon=MAX_EPISODE_HORIZON_FRAMES,
        )
    )

    assert result.reason == TerminationReason.NONE
    assert result.done is False


def test_termination_success():

    result = check_termination(
        TerminationInputs(
            success=True,
            collision=False,
            offroad=False,
            steps_elapsed=20,
            max_episode_horizon=MAX_EPISODE_HORIZON_FRAMES,
        )
    )

    assert result.reason == TerminationReason.SUCCESS
    assert result.done is True


def test_termination_collision_takes_priority_over_success():
    """A simultaneous success+collision frame must report collision,
    not a false clean success."""

    result = check_termination(
        TerminationInputs(
            success=True,
            collision=True,
            offroad=False,
            steps_elapsed=20,
            max_episode_horizon=MAX_EPISODE_HORIZON_FRAMES,
        )
    )

    assert result.reason == TerminationReason.FAILURE_COLLISION
    assert result.done is True


def test_termination_offroad():

    result = check_termination(
        TerminationInputs(
            success=False,
            collision=False,
            offroad=True,
            steps_elapsed=20,
            max_episode_horizon=MAX_EPISODE_HORIZON_FRAMES,
        )
    )

    assert result.reason == TerminationReason.FAILURE_OFFROAD
    assert result.done is True


def test_termination_truncation_at_horizon():

    result = check_termination(
        TerminationInputs(
            success=False,
            collision=False,
            offroad=False,
            steps_elapsed=MAX_EPISODE_HORIZON_FRAMES,
            max_episode_horizon=MAX_EPISODE_HORIZON_FRAMES,
        )
    )

    assert result.reason == TerminationReason.TRUNCATION_HORIZON
    assert result.done is True


def test_max_episode_horizon_derived_from_data_not_arbitrary():
    """MAX_EPISODE_HORIZON_FRAMES must exceed the observed 168
    -maneuver max span (52 frames, Stage B-0 Section 7) with a
    deliberate safety margin, not be an arbitrary round number picked
    without justification."""

    observed_max_span_frames = 52  # from the 168-maneuver distribution
    assert MAX_EPISODE_HORIZON_FRAMES > observed_max_span_frames
    assert MAX_EPISODE_HORIZON_FRAMES <= 3 * observed_max_span_frames
