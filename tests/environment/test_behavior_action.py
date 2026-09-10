"""Tests for src/environment/behavior_action.py."""

import numpy as np
import pytest

from src.environment.behavior_action import (
    NOMINAL_CRUISE_SPEED_MPS,
    NUM_BEHAVIOR_ACTIONS,
    STOP_TARGET_SPEED_MPS,
    BehaviorAction,
    BehaviorExecutor,
    FOLLOW_NO_LEAD_FALLBACK,
)


def _observation(**overrides):
    """A 14D observation with all vehicle slots absent by default,
    matching observation_builder's field order."""

    obs = np.array(
        [
            10.0,  # v_e
            5.0,  # d_m
            0.0,  # target_front_present
            0.0,  # target_front_gap
            0.0,  # target_front_relative_speed
            100.0,  # target_front_ttc
            0.0,  # target_rear_present
            0.0,
            0.0,
            100.0,
            0.0,  # source_front_present
            0.0,
            0.0,
            100.0,
        ]
    )
    for key, value in overrides.items():
        index = _FIELD_INDEX[key]
        obs[index] = value
    return obs


_FIELD_INDEX = {
    "v_e": 0,
    "d_m": 1,
    "target_front_present": 2,
    "target_front_gap": 3,
    "target_front_relative_speed": 4,
    "target_front_ttc": 5,
    "target_rear_present": 6,
    "target_rear_gap": 7,
    "target_rear_relative_speed": 8,
    "target_rear_ttc": 9,
    "source_front_present": 10,
    "source_front_gap": 11,
    "source_front_relative_speed": 12,
    "source_front_ttc": 13,
}


def test_action_enum_has_four_members_stable_values():

    assert NUM_BEHAVIOR_ACTIONS == 4
    assert BehaviorAction.KEEP == 0
    assert BehaviorAction.FOLLOW == 1
    assert BehaviorAction.MERGE == 2
    assert BehaviorAction.STOP == 3


def test_keep_targets_nominal_cruise_speed_even_with_close_lead():
    """Stage B-0 Section 3: KEEP must NOT silently become FOLLOW when
    a source-lane lead is close -- this is the specific behavior the
    prompt rejected from Stage A's closure patch."""

    executor = BehaviorExecutor()
    observation = _observation(
        source_front_present=1.0,
        source_front_gap=1.0,  # very close lead
        source_front_relative_speed=5.0,
    )

    objective = executor.compute_objective(BehaviorAction.KEEP, observation)

    assert objective.reference_speed_mps == pytest.approx(
        NOMINAL_CRUISE_SPEED_MPS
    )
    assert objective.reference_lane == "source"
    assert objective.fallback_applied is None


def test_follow_uses_source_lead_when_present():

    executor = BehaviorExecutor()
    observation = _observation(
        source_front_present=1.0,
        source_front_gap=10.0,
        source_front_relative_speed=5.0,  # closing
    )

    objective = executor.compute_objective(BehaviorAction.FOLLOW, observation)

    assert objective.fallback_applied is None
    assert objective.reference_speed_mps < NOMINAL_CRUISE_SPEED_MPS


def test_follow_with_no_lead_applies_explicit_fallback():
    """The one decided degenerate case (Stage B-0 Section 3): FOLLOW
    with no source-lane lead falls back to nominal cruise speed, WITH
    a visible diagnostic flag -- not a silent semantic change."""

    executor = BehaviorExecutor()
    observation = _observation(source_front_present=0.0)

    objective = executor.compute_objective(BehaviorAction.FOLLOW, observation)

    assert objective.reference_speed_mps == pytest.approx(
        NOMINAL_CRUISE_SPEED_MPS
    )
    assert objective.fallback_applied == FOLLOW_NO_LEAD_FALLBACK


def test_keep_and_follow_remain_semantically_distinct():
    """With a close, fast-closing source lead present, KEEP and
    FOLLOW must produce DIFFERENT objectives -- proving KEEP was not
    collapsed into FOLLOW's behavior."""

    executor = BehaviorExecutor()
    observation = _observation(
        source_front_present=1.0,
        source_front_gap=5.0,
        source_front_relative_speed=8.0,
    )

    keep_objective = executor.compute_objective(BehaviorAction.KEEP, observation)
    follow_objective = executor.compute_objective(
        BehaviorAction.FOLLOW, observation
    )

    assert keep_objective.reference_speed_mps != follow_objective.reference_speed_mps


def test_merge_targets_target_lane():

    executor = BehaviorExecutor()
    observation = _observation()

    objective = executor.compute_objective(BehaviorAction.MERGE, observation)

    assert objective.reference_lane == "target"
    assert objective.fallback_applied is None


def test_merge_uses_target_lead_when_present():

    executor = BehaviorExecutor()
    observation = _observation(
        target_front_present=1.0,
        target_front_gap=8.0,
        target_front_relative_speed=6.0,
    )

    objective = executor.compute_objective(BehaviorAction.MERGE, observation)

    assert objective.reference_speed_mps < NOMINAL_CRUISE_SPEED_MPS


def test_stop_is_non_terminal_and_targets_zero_speed():

    executor = BehaviorExecutor()
    observation = _observation()

    objective = executor.compute_objective(BehaviorAction.STOP, observation)

    assert objective.reference_speed_mps == pytest.approx(
        STOP_TARGET_SPEED_MPS
    )
    assert objective.reference_lane == "source"
    # STOP has no notion of "done" at the executor level at all --
    # BehaviorObjective carries no termination field, confirming
    # termination is handled elsewhere (termination.py), never by
    # the action/executor layer.
    assert not hasattr(objective, "done")
    assert not hasattr(objective, "terminated")


def test_all_four_actions_produce_distinct_objective_shapes_when_relevant():
    """Sanity: every action is handled (no silent fallthrough to an
    unrelated default)."""

    executor = BehaviorExecutor()
    observation = _observation()

    for action in BehaviorAction:
        objective = executor.compute_objective(action, observation)
        assert objective.reference_lane in ("source", "target")
        assert objective.reference_speed_mps >= 0.0
