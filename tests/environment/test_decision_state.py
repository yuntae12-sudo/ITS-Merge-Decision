"""Tests for src/environment/decision_state.py -- the MERGE
commitment / decision-phase state machine (Stage B-0 Section 4)."""

from src.environment.behavior_action import BehaviorAction
from src.environment.decision_state import DecisionPhase, DecisionState


def test_starts_in_decision_phase():

    state = DecisionState()

    assert state.phase == DecisionPhase.DECISION
    assert state.policy_choice_available
    assert not state.is_committed


def test_keep_follow_stop_remain_reselectable():

    state = DecisionState()

    for action in (
        BehaviorAction.KEEP,
        BehaviorAction.FOLLOW,
        BehaviorAction.STOP,
        BehaviorAction.KEEP,
    ):
        effective = state.advance(action)
        assert effective == action
        assert state.phase == DecisionPhase.DECISION
        assert state.policy_choice_available


def test_merge_commits_and_effective_action_is_merge_this_step():

    state = DecisionState()

    effective = state.advance(BehaviorAction.MERGE)

    assert effective == BehaviorAction.MERGE
    assert state.phase == DecisionPhase.MERGE_COMMITTED
    assert state.is_committed
    assert not state.policy_choice_available


def test_after_commitment_policy_action_is_ignored():
    """Once committed, whatever the policy 'selects' is ignored -- the
    effective action stays MERGE regardless (Stage B-0: the policy
    should not even be consulted once policy_choice_available is
    False, but advance() defensively ignores it either way)."""

    state = DecisionState()
    state.advance(BehaviorAction.MERGE)

    for attempted in (
        BehaviorAction.STOP,
        BehaviorAction.KEEP,
        BehaviorAction.FOLLOW,
    ):
        effective = state.advance(attempted)
        assert effective == BehaviorAction.MERGE
        assert state.is_committed


def test_stop_before_merge_does_not_commit():
    """STOP != MERGE: selecting STOP (even repeatedly) never
    transitions to MERGE_COMMITTED."""

    state = DecisionState()

    state.advance(BehaviorAction.STOP)
    state.advance(BehaviorAction.STOP)
    state.advance(BehaviorAction.KEEP)

    assert state.phase == DecisionPhase.DECISION
    assert not state.is_committed


def test_commitment_is_one_way():
    """Once MERGE_COMMITTED, no sequence of advance() calls returns to
    DECISION phase within the same DecisionState instance -- a fresh
    episode gets a fresh DecisionState instead."""

    state = DecisionState()
    state.advance(BehaviorAction.MERGE)

    for _ in range(5):
        state.advance(BehaviorAction.KEEP)

    assert state.is_committed
