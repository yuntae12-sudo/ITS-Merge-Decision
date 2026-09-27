"""Final reward targeted tests (behavioral coverage for
src/rewards/merge_reward.py's Terminal+Safety+Progress+Decision reward
and its RewardDecisionCost.enabled on/off contract).

A minimal 14-length observation list is used throughout (mirrors
src.environment.observation_builder.OBSERVATION_FIELD_NAMES's order),
never importing src.environment (same constraint as merge_reward.py
itself).
"""

import ast
import dataclasses
import inspect
import math

import pytest

from src.environment.behavior_action import BehaviorAction
from src.environment.termination import TerminationReason
from src.rewards import merge_reward
from src.rewards.merge_reward import compute_reward, compute_reward_breakdown
from src.rewards.reward_wrapper import MergeRewardWrapper
from src.training.config import load_reward_config

REWARD_CONFIG_V1 = load_reward_config("configs/reward.yaml")
# RewardDecisionCost.enabled=True variant (the flag's "on" state) --
# exercises the legacy flat per-decision-step cost path, which the
# final config's enabled=false leaves otherwise untested.
REWARD_CONFIG_V0 = dataclasses.replace(
    REWARD_CONFIG_V1,
    decision_cost=dataclasses.replace(REWARD_CONFIG_V1.decision_cost, enabled=True),
)

TTC_CAP_S = 100.0  # mirrors observation_builder.TTC_CAP_S


def make_observation(
    d_m=10.0,
    target_front_present=1.0,
    target_front_gap=10.0,
    target_front_ttc=5.0,
):
    """Minimal 14D observation vector (only the fields Reward V1 reads
    are parameterized; the rest are filled with harmless defaults)."""

    return [
        5.0,  # v_e
        d_m,
        target_front_present,
        target_front_gap,
        0.0,  # target_front_relative_speed
        target_front_ttc,
        0.0,  # target_rear_present
        0.0,  # target_rear_gap
        0.0,  # target_rear_relative_speed
        TTC_CAP_S,  # target_rear_ttc
        0.0,  # source_front_present
        0.0,  # source_front_gap
        0.0,  # source_front_relative_speed
        TTC_CAP_S,  # source_front_ttc
    ]


# ------------------------------------------------------------------
# Terminal (Section 24 / 15) -- re-verify against V1 config specifically,
# since V0's own regression suite only loads the V0 config.
# ------------------------------------------------------------------

TERMINAL_TABLE = {
    TerminationReason.SUCCESS.value: 1.0,
    TerminationReason.FAILURE_COLLISION.value: -1.0,
    TerminationReason.FAILURE_OFFROAD.value: -1.0,
    TerminationReason.TRUNCATION_HORIZON.value: -0.5,
    TerminationReason.NONE.value: 0.0,
}


@pytest.mark.parametrize("reason,expected", list(TERMINAL_TABLE.items()))
def test_v1_terminal_outcome_table_unchanged(reason, expected):
    value = compute_reward(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=reason,
        is_policy_step=False,
    )
    assert value == pytest.approx(expected)


# ------------------------------------------------------------------
# Legacy V0 decision_cost exclusion from V1 total (this session's
# correctness patch, brief Section 15 items 1-8).
# ------------------------------------------------------------------


def test_v0_policy_decision_cost_still_applies():
    """V0 must be completely unaffected by the V1 correctness patch."""

    assert REWARD_CONFIG_V0.decision_cost.enabled is True
    value = compute_reward(
        reward_config=REWARD_CONFIG_V0,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=True,
    )
    assert value == pytest.approx(-0.01)


def test_v1_legacy_decision_cost_disabled_in_config():
    assert REWARD_CONFIG_V1.decision_cost.enabled is False


def test_v1_policy_decision_legacy_cost_is_zero():
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=True,
    )
    assert breakdown["decision_cost"] == pytest.approx(0.0)


def test_v1_auto_execution_legacy_cost_is_zero():
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=False,
    )
    assert breakdown["decision_cost"] == pytest.approx(0.0)


def test_v1_total_excludes_legacy_decision_cost_on_switch():
    """Action switch: V1 total = terminal + 0(legacy) + safety + progress
    + weighted -1.0 decision regularizer -- never terminal + legacy
    decision_cost + ... (the pre-patch bug)."""

    obs = make_observation(d_m=10.0, target_front_present=0.0)
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.SUCCESS.value,
        is_policy_step=True,
        observation=obs,
        next_observation=obs,
        action=BehaviorAction.MERGE,
        previous_policy_action=BehaviorAction.FOLLOW,
        d_m_initial=10.0,
    )
    assert breakdown["decision_cost"] == pytest.approx(0.0)
    expected_total = (
        breakdown["terminal"] + breakdown["safety_weighted"]
        + breakdown["progress_weighted"] + breakdown["decision_weighted"]
    )
    assert breakdown["total"] == pytest.approx(expected_total)


def test_v0_total_is_terminal_plus_legacy_decision_cost():
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V0,
        termination_reason=TerminationReason.SUCCESS.value,
        is_policy_step=True,
    )
    assert breakdown["total"] == pytest.approx(breakdown["terminal"] + breakdown["decision_cost"])
    assert breakdown["total"] == pytest.approx(1.0 - 0.01)


def test_v1_total_is_exactly_four_components():
    obs = make_observation(d_m=10.0, target_front_gap=5.0, target_front_ttc=4.0)
    next_obs = make_observation(d_m=8.0)
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.SUCCESS.value,
        is_policy_step=True,
        observation=obs,
        next_observation=next_obs,
        action=BehaviorAction.MERGE,
        previous_policy_action=BehaviorAction.FOLLOW,
        d_m_initial=10.0,
    )
    four_component_sum = (
        breakdown["terminal"] + breakdown["safety_weighted"]
        + breakdown["progress_weighted"] + breakdown["decision_weighted"]
    )
    assert breakdown["total"] == pytest.approx(four_component_sum)
    assert breakdown["decision_cost"] == pytest.approx(0.0)


# ------------------------------------------------------------------
# TTC (Section 25)
# ------------------------------------------------------------------


def _r_ttc_only(ttc, gap=10.0, present=1.0):
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=False,
        observation=make_observation(target_front_present=present, target_front_gap=gap, target_front_ttc=ttc),
    )
    return breakdown["ttc_penalty_raw"]


def test_ttc_plus_inf_is_zero():
    assert _r_ttc_only(TTC_CAP_S) == pytest.approx(0.0)


def test_ttc_zero_sentinel_masked():
    """gap<=0 -> TTC==0.0 sentinel must be masked to 0, never -1."""
    assert _r_ttc_only(ttc=0.0, gap=0.0) == pytest.approx(0.0)


def test_ttc_negative_masked():
    assert _r_ttc_only(ttc=0.0, gap=-3.0) == pytest.approx(0.0)


def test_safety_mask_semantics_audit_against_upstream_ttc_source():
    """CASE A verification (this session's correctness patch): asserts
    the actual upstream fact this module's masking rule depends on --
    that src.scenarios.scenario_features._compute_ttc has NO branch
    that can produce a genuine (non-sentinel) finite TTC when
    gap_m<=0.0. If this ever changes upstream, this test must fail
    before the reward's masking rule silently becomes stale."""

    from src.scenarios.scenario_features import _compute_ttc

    for closing_speed in (-5.0, 0.0, 0.001, 1.0, 100.0):
        for gap in (-10.0, -0.001, 0.0):
            assert _compute_ttc(gap, closing_speed) == 0.0, (
                f"_compute_ttc({gap}, {closing_speed}) did not return the "
                "expected 0.0 sentinel -- merge_reward's safety-mask "
                "correctness assumption (gap<=0 implies TTC is "
                "structurally the sentinel, never an independent "
                "reading) no longer holds; re-audit CASE A/B before "
                "trusting r_safety's current masking rule."
            )

    # And the reverse: for gap>0, _compute_ttc CAN produce a genuine
    # non-sentinel value (either a finite closing time or +Inf for
    # non-closing) -- confirming TTC validity is NOT always tied to gap
    # sign, only specifically at gap<=0.
    assert _compute_ttc(5.0, 2.0) == pytest.approx(2.5)
    assert _compute_ttc(5.0, -1.0) == float("inf")


def test_ttc_one_second_is_danger():
    assert _r_ttc_only(ttc=1.0) == pytest.approx(-1.0)


def test_ttc_at_danger_threshold():
    assert _r_ttc_only(ttc=2.5) == pytest.approx(-1.0)


def test_ttc_between_danger_and_safe_is_linear():
    # midpoint of [2.5, 9.0] = 5.75 -> exactly -0.5
    value = _r_ttc_only(ttc=5.75)
    assert value == pytest.approx(-0.5)


def test_ttc_at_safe_threshold():
    assert _r_ttc_only(ttc=9.0) == pytest.approx(0.0)


def test_ttc_above_safe_threshold():
    assert _r_ttc_only(ttc=50.0) == pytest.approx(0.0)


def test_ttc_missing_target_front_is_zero():
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=False,
        observation=make_observation(target_front_present=0.0),
    )
    assert breakdown["ttc_penalty_raw"] == pytest.approx(0.0)
    assert breakdown["gap_penalty_raw"] == pytest.approx(0.0)


# ------------------------------------------------------------------
# Gap (Section 26)
# ------------------------------------------------------------------


def _r_gap_only(gap, ttc=50.0, present=1.0):
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=False,
        observation=make_observation(target_front_present=present, target_front_gap=gap, target_front_ttc=ttc),
    )
    return breakdown["gap_penalty_raw"]


def test_gap_invalid_is_zero():
    assert _r_gap_only(gap=10.0, present=0.0) == pytest.approx(0.0)


def test_gap_le_zero_masked():
    assert _r_gap_only(gap=0.0) == pytest.approx(0.0)
    assert _r_gap_only(gap=-5.0) == pytest.approx(0.0)


def test_gap_10m_is_half_penalty():
    assert _r_gap_only(gap=10.0) == pytest.approx(-0.5)


def test_gap_20m_is_zero():
    assert _r_gap_only(gap=20.0) == pytest.approx(0.0)


def test_gap_above_20m_is_zero():
    assert _r_gap_only(gap=45.0) == pytest.approx(0.0)


def test_rear_and_source_gap_ttc_never_enter_safety():
    """Even with dangerous rear/source values, r_safety must be
    unaffected -- final approved scope is target-front only."""

    obs = make_observation(target_front_present=0.0)
    obs[6] = 1.0  # target_rear_present
    obs[7] = 0.5  # target_rear_gap (dangerously small)
    obs[9] = 0.5  # target_rear_ttc (dangerously small)
    obs[10] = 1.0  # source_front_present
    obs[11] = 0.5  # source_front_gap
    obs[13] = 0.5  # source_front_ttc

    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=False,
        observation=obs,
    )
    assert breakdown["safety_weighted"] == pytest.approx(0.0)
    assert breakdown["ttc_penalty_raw"] == pytest.approx(0.0)
    assert breakdown["gap_penalty_raw"] == pytest.approx(0.0)


# ------------------------------------------------------------------
# Safety aggregation (Section 27)
# ------------------------------------------------------------------


def test_safety_both_zero():
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=False,
        observation=make_observation(target_front_gap=20.0, target_front_ttc=9.0),
    )
    assert breakdown["ttc_penalty_raw"] == pytest.approx(0.0)
    assert breakdown["gap_penalty_raw"] == pytest.approx(0.0)


def test_safety_min_picks_ttc():
    # ttc=1.0 -> r_ttc=-1.0; gap=20.0 -> r_gap=0.0; min = -1.0
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=False,
        observation=make_observation(target_front_gap=20.0, target_front_ttc=1.0),
    )
    assert breakdown["ttc_penalty_raw"] == pytest.approx(-1.0)
    assert breakdown["gap_penalty_raw"] == pytest.approx(0.0)
    assert breakdown["safety_weighted"] == pytest.approx(REWARD_CONFIG_V1.safety.weight * -1.0)


def test_safety_min_picks_gap():
    # ttc=50 -> r_ttc=0.0; gap=0.1 (>0, not sentinel) -> r_gap close to -1
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=False,
        observation=make_observation(target_front_gap=0.1, target_front_ttc=50.0),
    )
    expected_gap = -(20.0 - 0.1) / 20.0
    assert breakdown["gap_penalty_raw"] == pytest.approx(expected_gap)
    assert breakdown["safety_weighted"] == pytest.approx(REWARD_CONFIG_V1.safety.weight * expected_gap)


def test_safety_range_always_bounded():
    for ttc in (0.0, 1.0, 2.5, 5.75, 9.0, 50.0, TTC_CAP_S):
        for gap in (-5.0, 0.0, 0.1, 10.0, 20.0, 45.0):
            breakdown = compute_reward_breakdown(
                reward_config=REWARD_CONFIG_V1,
                termination_reason=TerminationReason.NONE.value,
                is_policy_step=False,
                observation=make_observation(target_front_gap=gap, target_front_ttc=ttc),
            )
            raw_safety = min(breakdown["ttc_penalty_raw"], breakdown["gap_penalty_raw"])
            assert -1.0 <= raw_safety <= 0.0


def test_weighted_safety_range():
    w = REWARD_CONFIG_V1.safety.weight
    for ttc in (1.0, 5.75, 50.0):
        breakdown = compute_reward_breakdown(
            reward_config=REWARD_CONFIG_V1,
            termination_reason=TerminationReason.NONE.value,
            is_policy_step=False,
            observation=make_observation(target_front_ttc=ttc),
        )
        assert -w <= breakdown["safety_weighted"] <= 0.0


# ------------------------------------------------------------------
# Progress (Section 28)
# ------------------------------------------------------------------


def test_progress_stationary_is_zero():
    obs = make_observation(d_m=10.0)
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=True,
        observation=obs,
        next_observation=obs,
        d_m_initial=10.0,
    )
    assert breakdown["progress_weighted"] == pytest.approx(0.0)


def test_progress_forward_is_positive():
    obs = make_observation(d_m=10.0)
    next_obs = make_observation(d_m=8.0)
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=True,
        observation=obs,
        next_observation=next_obs,
        d_m_initial=10.0,
    )
    assert breakdown["progress_weighted"] > 0.0


def test_progress_backward_is_nonpositive():
    obs = make_observation(d_m=8.0)
    next_obs = make_observation(d_m=10.0)
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=True,
        observation=obs,
        next_observation=next_obs,
        d_m_initial=10.0,
    )
    assert breakdown["progress_weighted"] <= 0.0


def test_phi_range_bounded():
    for d_m in (-5.0, 0.0, 5.0, 10.0, 15.0):
        phi = merge_reward.compute_progress_potential(d_m, d_m_initial=10.0, epsilon_m=0.5)
        assert 0.0 <= phi <= 1.0


def test_progress_exact_example_from_brief():
    """d_m_initial=10, d_m_current=10, d_m_next=8 ->
    Phi_current=0, Phi_next=0.2, raw progress=0.2, weighted=0.1."""

    obs = make_observation(d_m=10.0)
    next_obs = make_observation(d_m=8.0)
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=True,
        observation=obs,
        next_observation=next_obs,
        d_m_initial=10.0,
    )
    raw_progress = breakdown["progress_weighted"] / REWARD_CONFIG_V1.progress.weight
    assert raw_progress == pytest.approx(0.2)
    assert breakdown["progress_weighted"] == pytest.approx(0.1)


def test_degenerate_d_m_initial_is_safe():
    obs = make_observation(d_m=0.1)
    next_obs = make_observation(d_m=0.05)
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=True,
        observation=obs,
        next_observation=next_obs,
        d_m_initial=0.1,  # <= epsilon_m (0.5) -> Phi := 1.0 always
    )
    assert breakdown["progress_weighted"] == pytest.approx(0.0)
    assert math.isfinite(breakdown["progress_weighted"])


def test_no_d_m_initial_yields_zero_progress():
    obs = make_observation(d_m=10.0)
    next_obs = make_observation(d_m=5.0)
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=True,
        observation=obs,
        next_observation=next_obs,
        d_m_initial=None,
    )
    assert breakdown["progress_weighted"] == pytest.approx(0.0)


# ------------------------------------------------------------------
# Progress via the stateful wrapper -- d_m_initial lifecycle, reset (Section 11/28)
# ------------------------------------------------------------------


def test_wrapper_d_m_initial_fixed_during_episode():
    wrapper = MergeRewardWrapper(REWARD_CONFIG_V1)
    obs0 = make_observation(d_m=10.0)
    obs1 = make_observation(d_m=8.0)
    obs2 = make_observation(d_m=6.0)

    wrapper.compute(TerminationReason.NONE.value, True, observation=obs0, next_observation=obs1, action=BehaviorAction.KEEP)
    assert wrapper._d_m_initial == pytest.approx(10.0)

    wrapper.compute(TerminationReason.NONE.value, True, observation=obs1, next_observation=obs2, action=BehaviorAction.KEEP)
    assert wrapper._d_m_initial == pytest.approx(10.0)  # unchanged mid-episode


def test_wrapper_reset_clears_d_m_initial():
    wrapper = MergeRewardWrapper(REWARD_CONFIG_V1)
    obs0 = make_observation(d_m=10.0)
    wrapper.compute(TerminationReason.NONE.value, True, observation=obs0, next_observation=obs0, action=BehaviorAction.KEEP)
    assert wrapper._d_m_initial is not None

    wrapper.reset()
    assert wrapper._d_m_initial is None


def test_no_cross_episode_progress_leak():
    """After reset(), the next episode's d_m_initial must be captured
    fresh from its own first observation, not the previous episode's."""

    wrapper = MergeRewardWrapper(REWARD_CONFIG_V1)
    wrapper.compute(TerminationReason.NONE.value, True, observation=make_observation(d_m=50.0), next_observation=make_observation(d_m=50.0), action=BehaviorAction.KEEP)
    assert wrapper._d_m_initial == pytest.approx(50.0)

    wrapper.reset()
    wrapper.compute(TerminationReason.NONE.value, True, observation=make_observation(d_m=3.0), next_observation=make_observation(d_m=3.0), action=BehaviorAction.KEEP)
    assert wrapper._d_m_initial == pytest.approx(3.0)


def test_future_logged_information_not_used_static_check():
    """Static check: _r_progress's signature takes only observation/
    next_observation/d_m_initial -- no trajectory/future-outcome
    parameter exists for it to (mis)use."""

    import inspect as _inspect

    sig = _inspect.signature(merge_reward._r_progress)
    assert set(sig.parameters) == {"reward_config", "observation", "next_observation", "d_m_initial"}


# ------------------------------------------------------------------
# Decision (Section 29)
# ------------------------------------------------------------------


def test_decision_first_action_is_zero():
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=True,
        action=BehaviorAction.KEEP,
        previous_policy_action=None,
    )
    assert breakdown["decision_weighted"] == pytest.approx(0.0)


def test_decision_same_action_is_zero():
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=True,
        action=BehaviorAction.FOLLOW,
        previous_policy_action=BehaviorAction.FOLLOW,
    )
    assert breakdown["decision_weighted"] == pytest.approx(0.0)


def test_decision_switch_is_negative():
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=True,
        action=BehaviorAction.MERGE,
        previous_policy_action=BehaviorAction.FOLLOW,
    )
    assert breakdown["decision_weighted"] == pytest.approx(REWARD_CONFIG_V1.decision.weight * -1.0)


def test_decision_weighted_switch_value():
    assert REWARD_CONFIG_V1.decision.weight == pytest.approx(0.02)
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=True,
        action=BehaviorAction.STOP,
        previous_policy_action=BehaviorAction.KEEP,
    )
    assert breakdown["decision_weighted"] == pytest.approx(-0.02)


def test_decision_policy_mask_zero_is_zero():
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=False,
        action=BehaviorAction.MERGE,
        previous_policy_action=BehaviorAction.FOLLOW,
    )
    assert breakdown["decision_weighted"] == pytest.approx(0.0)


def test_wrapper_auto_execution_does_not_overwrite_previous_action():
    wrapper = MergeRewardWrapper(REWARD_CONFIG_V1)
    wrapper.compute(TerminationReason.NONE.value, True, action=BehaviorAction.FOLLOW)
    assert wrapper._previous_policy_action == BehaviorAction.FOLLOW

    # Auto-executed MERGE-commitment step: policy_mask=0, must not
    # overwrite previous_policy_action even though action=MERGE differs.
    wrapper.compute(TerminationReason.NONE.value, False, action=BehaviorAction.MERGE)
    assert wrapper._previous_policy_action == BehaviorAction.FOLLOW


def test_wrapper_reset_clears_previous_policy_action():
    wrapper = MergeRewardWrapper(REWARD_CONFIG_V1)
    wrapper.compute(TerminationReason.NONE.value, True, action=BehaviorAction.KEEP)
    assert wrapper._previous_policy_action == BehaviorAction.KEEP

    wrapper.reset()
    assert wrapper._previous_policy_action is None


# ------------------------------------------------------------------
# Accounting (Section 30 / 41-44)
# ------------------------------------------------------------------


def test_component_sum_equals_total():
    obs = make_observation(d_m=10.0, target_front_gap=5.0, target_front_ttc=4.0)
    next_obs = make_observation(d_m=8.0)
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.SUCCESS.value,
        is_policy_step=True,
        observation=obs,
        next_observation=next_obs,
        action=BehaviorAction.MERGE,
        previous_policy_action=BehaviorAction.FOLLOW,
        d_m_initial=10.0,
    )
    component_sum = (
        breakdown["terminal"]
        + breakdown["decision_cost"]
        + breakdown["safety_weighted"]
        + breakdown["progress_weighted"]
        + breakdown["decision_weighted"]
    )
    assert component_sum == pytest.approx(breakdown["total"], rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("reason", list(TERMINAL_TABLE.keys()))
@pytest.mark.parametrize("is_policy_step", [True, False])
def test_no_nan_or_unexpected_inf_v1(reason, is_policy_step):
    obs = make_observation(d_m=10.0, target_front_gap=5.0, target_front_ttc=4.0)
    next_obs = make_observation(d_m=9.0)
    value = compute_reward(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=reason,
        is_policy_step=is_policy_step,
        observation=obs,
        next_observation=next_obs,
        action=BehaviorAction.KEEP,
        previous_policy_action=BehaviorAction.FOLLOW,
        d_m_initial=10.0,
    )
    assert math.isfinite(value)


def test_diagnostic_ttc_gap_not_double_counted_in_total():
    """reward_diag/ttc_penalty and reward_diag/gap_penalty are raw
    (unweighted) diagnostics -- they must never be separately summed
    into reward/total (only the aggregated, weighted r_safety is)."""

    obs = make_observation(target_front_gap=0.1, target_front_ttc=1.0)
    breakdown = compute_reward_breakdown(
        reward_config=REWARD_CONFIG_V1,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=False,
        observation=obs,
    )
    # raw ttc/gap penalties are both near -1 / -0.995 -- if they were
    # BOTH separately added to total (instead of only the min() picked
    # once, weighted), total would be far more negative than this.
    assert breakdown["ttc_penalty_raw"] == pytest.approx(-1.0)
    assert breakdown["gap_penalty_raw"] == pytest.approx(-0.995, abs=1e-3)
    reconstructed_total = (
        breakdown["terminal"] + breakdown["decision_cost"] + breakdown["safety_weighted"]
        + breakdown["progress_weighted"] + breakdown["decision_weighted"]
    )
    assert reconstructed_total == pytest.approx(breakdown["total"])
    # safety_weighted alone reflects only min(ttc, gap) * weight, not (ttc+gap)*weight
    w = REWARD_CONFIG_V1.safety.weight
    assert breakdown["safety_weighted"] == pytest.approx(w * min(breakdown["ttc_penalty_raw"], breakdown["gap_penalty_raw"]))


# ------------------------------------------------------------------
# Reserved-field guards (use_target_rear / use_source_front / use_gamma / switching_only=False)
# ------------------------------------------------------------------


def test_reserved_use_target_rear_raises():
    import dataclasses

    bad_safety = dataclasses.replace(REWARD_CONFIG_V1.safety, use_target_rear=True)
    bad_config = dataclasses.replace(REWARD_CONFIG_V1, safety=bad_safety)
    with pytest.raises(NotImplementedError):
        compute_reward_breakdown(
            reward_config=bad_config,
            termination_reason=TerminationReason.NONE.value,
            is_policy_step=False,
            observation=make_observation(),
        )


def test_reserved_use_gamma_raises():
    bad_progress = dataclasses.replace(REWARD_CONFIG_V1.progress, use_gamma=True)
    bad_config = dataclasses.replace(REWARD_CONFIG_V1, progress=bad_progress)
    with pytest.raises(NotImplementedError):
        compute_reward_breakdown(
            reward_config=bad_config,
            termination_reason=TerminationReason.NONE.value,
            is_policy_step=True,
            observation=make_observation(),
            next_observation=make_observation(),
            d_m_initial=10.0,
        )


def test_unrecognized_termination_reason_rejected():
    """This module must only ever map the fixed 5-value enum table --
    it must not silently accept (and thus implicitly invent semantics
    for) some other string."""

    with pytest.raises(ValueError):
        merge_reward.compute_reward(
            reward_config=REWARD_CONFIG_V1,
            termination_reason="not_a_real_termination_reason",
            is_policy_step=True,
        )


def test_reward_module_does_not_reimplement_termination_detection():
    """Code-level check: merge_reward.py must not import anything from
    src.environment (it must consume only the already-computed
    termination_reason string/enum handed to it, not re-derive
    success/collision/offroad/timeout by importing the environment's
    own detection machinery).

    Checks actual import statements (not every substring occurrence) --
    the module docstring/comments legitimately mention
    ``src.environment.observation_builder`` by name (to document which
    observation indices its local mirrored constants correspond to)
    without importing it; a plain substring check would false-positive
    on that prose."""

    tree = ast.parse(inspect.getsource(merge_reward))
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not any(m == "src.environment" or m.startswith("src.environment.") for m in imported_modules)
    assert not any(m == "waymax" or m.startswith("waymax.") for m in imported_modules)

    # Signature-level check: compute_reward takes termination_reason
    # and is_policy_step as explicit inputs (not, e.g., a raw
    # trajectory/state it could inspect itself).
    sig = inspect.signature(merge_reward.compute_reward)
    assert "termination_reason" in sig.parameters
    assert "is_policy_step" in sig.parameters
