"""Stage B-2 Rule-Based FSM baseline tests.

Synthetic-observation unit tests (boundary rule coverage) plus a
fairness-audit test confirming FsmPolicy reads nothing beyond the
shared 14D observation vector.
"""

import inspect

from src.environment.behavior_action import BehaviorAction
from src.environment.fsm_policy import (
    FOLLOW_GAP_M,
    FOLLOW_TTC_S,
    GAP_SAFE_M,
    STOP_MARGIN_M,
    TTC_SAFE_S,
    FsmPolicy,
)
from src.environment.observation_builder import OBSERVATION_DIM, TTC_CAP_S

FSM = FsmPolicy()


def make_observation(
    v_e=15.0,
    d_m=20.0,
    target_front_present=0.0,
    target_front_gap=0.0,
    target_front_relative_speed=0.0,
    target_front_ttc=TTC_CAP_S,
    target_rear_present=0.0,
    target_rear_gap=0.0,
    target_rear_relative_speed=0.0,
    target_rear_ttc=TTC_CAP_S,
    source_front_present=0.0,
    source_front_gap=0.0,
    source_front_relative_speed=0.0,
    source_front_ttc=TTC_CAP_S,
):
    obs = [
        v_e,
        d_m,
        target_front_present,
        target_front_gap,
        target_front_relative_speed,
        target_front_ttc,
        target_rear_present,
        target_rear_gap,
        target_rear_relative_speed,
        target_rear_ttc,
        source_front_present,
        source_front_gap,
        source_front_relative_speed,
        source_front_ttc,
    ]
    assert len(obs) == OBSERVATION_DIM
    return obs


def test_no_vehicles_present_merges():
    obs = make_observation()
    decision = FSM.decide(obs)
    assert decision.action == BehaviorAction.MERGE


def test_unsafe_target_front_prevents_merge():
    obs = make_observation(
        target_front_present=1.0,
        target_front_gap=GAP_SAFE_M - 1.0,
        target_front_ttc=TTC_SAFE_S - 1.0,
    )
    decision = FSM.decide(obs)
    assert decision.action != BehaviorAction.MERGE


def test_unsafe_target_rear_prevents_merge():
    obs = make_observation(
        target_rear_present=1.0,
        target_rear_gap=GAP_SAFE_M - 1.0,
        target_rear_ttc=TTC_SAFE_S - 1.0,
    )
    decision = FSM.decide(obs)
    assert decision.action != BehaviorAction.MERGE


def test_close_gap_prevents_merge_even_when_not_closing():
    """Stage B-2.6: gap alone below threshold is UNSAFE even if TTC
    clears (not closing) -- the two conditions are AND'd
    (`_target_gap_safe`). A present, non-closing/diverging vehicle is
    reported at the TTC_CAP_S sentinel regardless of true proximity
    (see ``observation_builder._encode_vehicle_slot``), so TTC alone
    cannot certify safety when the vehicle is physically close."""

    obs = make_observation(
        target_front_present=1.0,
        target_front_gap=GAP_SAFE_M - 1.0,
        target_front_ttc=TTC_CAP_S,
    )
    decision = FSM.decide(obs)
    assert decision.action != BehaviorAction.MERGE


def test_close_ttc_prevents_merge_even_with_ample_gap():
    """Stage B-2.6: TTC alone below threshold is UNSAFE even if the
    instantaneous gap clears -- a closing vehicle 11m away that will
    reach the gap in 1s is not safe to merge in front of/behind."""

    obs = make_observation(
        target_front_present=1.0,
        target_front_gap=GAP_SAFE_M + 1.0,
        target_front_ttc=TTC_SAFE_S - 3.0,
    )
    decision = FSM.decide(obs)
    assert decision.action != BehaviorAction.MERGE


def test_both_gap_and_ttc_clear_merges():
    """Stage B-2.6: both conditions clearing their thresholds is still
    sufficient for a present slot to be judged safe."""

    obs = make_observation(
        target_front_present=1.0,
        target_front_gap=GAP_SAFE_M + 1.0,
        target_front_ttc=TTC_SAFE_S + 1.0,
    )
    decision = FSM.decide(obs)
    assert decision.action == BehaviorAction.MERGE


def test_absent_slot_is_still_safe():
    obs = make_observation(target_front_present=0.0)
    decision = FSM.decide(obs)
    assert decision.action == BehaviorAction.MERGE


def test_safe_target_gap_merges():
    obs = make_observation(
        target_front_present=1.0,
        target_front_gap=GAP_SAFE_M + 5.0,
        target_front_ttc=TTC_SAFE_S + 1.0,
        target_rear_present=1.0,
        target_rear_gap=GAP_SAFE_M + 5.0,
        target_rear_ttc=TTC_SAFE_S + 1.0,
    )
    decision = FSM.decide(obs)
    assert decision.action == BehaviorAction.MERGE


def test_unsafe_merge_with_ample_margin_and_no_source_lead_keeps():
    obs = make_observation(
        d_m=STOP_MARGIN_M + 10.0,
        target_front_present=1.0,
        target_front_gap=GAP_SAFE_M - 1.0,
        target_front_ttc=TTC_SAFE_S - 1.0,
        source_front_present=0.0,
    )
    decision = FSM.decide(obs)
    assert decision.action == BehaviorAction.KEEP


def test_unsafe_merge_with_close_source_lead_follows():
    obs = make_observation(
        d_m=STOP_MARGIN_M + 10.0,
        target_front_present=1.0,
        target_front_gap=GAP_SAFE_M - 1.0,
        target_front_ttc=TTC_SAFE_S - 1.0,
        source_front_present=1.0,
        source_front_gap=FOLLOW_GAP_M - 1.0,
        source_front_ttc=FOLLOW_TTC_S - 1.0,
    )
    decision = FSM.decide(obs)
    assert decision.action == BehaviorAction.FOLLOW


def test_unsafe_merge_with_exhausted_margin_stops():
    obs = make_observation(
        d_m=STOP_MARGIN_M - 1.0,
        target_front_present=1.0,
        target_front_gap=GAP_SAFE_M - 1.0,
        target_front_ttc=TTC_SAFE_S - 1.0,
    )
    decision = FSM.decide(obs)
    assert decision.action == BehaviorAction.STOP


def test_missing_presence_flags_are_treated_as_safe():
    """A slot with present=0 (the Stage A/B-0 missing-value encoding)
    must never itself block a MERGE, regardless of its zero-filled
    gap/relative-speed/TTC-cap sentinel values."""

    obs = make_observation(
        target_front_present=0.0,
        target_front_gap=0.0,
        target_front_ttc=0.0,  # would look unsafe if present were 1.0
        target_rear_present=0.0,
        target_rear_gap=0.0,
        target_rear_ttc=0.0,
    )
    decision = FSM.decide(obs)
    assert decision.action == BehaviorAction.MERGE


def test_decision_is_deterministic():
    obs = make_observation(
        target_front_present=1.0,
        target_front_gap=GAP_SAFE_M - 1.0,
        target_front_ttc=TTC_SAFE_S - 1.0,
    )
    first = FSM.decide(obs)
    second = FSM.decide(obs)
    assert first.action == second.action
    assert first.internal_state == second.internal_state


def test_rejects_wrong_length_observation():
    import pytest

    with pytest.raises(ValueError):
        FSM.decide([0.0, 1.0, 2.0])


def test_fsm_policy_reads_only_the_observation_argument():
    """Fairness audit (Stage B-2 Section B11/B12): FsmPolicy.decide's
    only parameters are self and observation -- confirmed via
    signature inspection so this test fails loudly if a future edit
    adds any other input (e.g. environment/internal state) the FSM
    would read instead."""

    signature = inspect.signature(FsmPolicy.decide)
    param_names = list(signature.parameters.keys())
    assert param_names == ["self", "observation"]


def test_fsm_policy_has_no_stored_privileged_state():
    """FsmPolicy instances carry no attributes at all (stateless, pure
    function of observation) -- confirmed directly rather than assumed,
    so a future edit that sneaks in cached environment/internal state
    is caught here."""

    fresh = FsmPolicy()
    assert vars(fresh) == {}
