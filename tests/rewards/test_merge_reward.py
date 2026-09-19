"""P2 behavioral tests for Reward V0 (docs/ppo/PPO_PLAN.md SS0.1/P2,
SS5, SS5.1).

Covers the required-tests list from PPO_PLAN.md SS0.1/P2:
  - SUCCESS -> +1.0, COLLISION -> -1.0, OFFROAD -> -1.0,
    TIMEOUT/TRUNCATION_HORIZON -> -0.5, NONE -> 0.0
  - real decision step -> -0.01 decision cost, auto-committed step ->
    0.0 decision cost
  - total reward = terminal_component + decision_cost_component,
    correctly summed for a few example combinations
  - no NaN/inf in any reward path
  - reward module does not re-derive termination -- verified at the
    code level: it only consumes the caller-supplied
    ``termination_reason``/``is_policy_step``, never any independent
    collision/offroad/success detection code.
"""

import inspect
import math

import pytest

from src.environment.termination import TerminationReason
from src.rewards import merge_reward
from src.rewards.reward_wrapper import MergeRewardWrapper
from src.training.config import load_reward_config

REWARD_CONFIG = load_reward_config("configs/reward/merge_reward_v0.yaml")

# TerminationReason.value strings, per src/environment/termination.py:48-53.
TERMINAL_TABLE = {
    TerminationReason.SUCCESS.value: 1.0,
    TerminationReason.FAILURE_COLLISION.value: -1.0,
    TerminationReason.FAILURE_OFFROAD.value: -1.0,
    TerminationReason.TRUNCATION_HORIZON.value: -0.5,
    TerminationReason.NONE.value: 0.0,
}


@pytest.mark.parametrize("reason,expected", list(TERMINAL_TABLE.items()))
def test_terminal_outcome_table(reason, expected):
    """Terminal component alone (decision cost held at 0 via
    is_policy_step=False on an auto-execution step) matches the fixed
    V0 table exactly."""

    value = merge_reward.compute_reward(
        reward_config=REWARD_CONFIG,
        termination_reason=reason,
        is_policy_step=False,
    )
    assert value == pytest.approx(expected)


def test_termination_reason_none_maps_to_zero():
    # Mirrors MergeEnvironment's info["termination_reason"] == None
    # case (before any TerminationResult exists, e.g. at reset()).
    value = merge_reward.compute_reward(
        reward_config=REWARD_CONFIG,
        termination_reason=None,
        is_policy_step=False,
    )
    assert value == pytest.approx(0.0)


def test_real_decision_step_cost():
    value = merge_reward.compute_reward(
        reward_config=REWARD_CONFIG,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=True,
    )
    assert value == pytest.approx(-0.01)


def test_auto_committed_step_cost():
    value = merge_reward.compute_reward(
        reward_config=REWARD_CONFIG,
        termination_reason=TerminationReason.NONE.value,
        is_policy_step=False,
    )
    assert value == pytest.approx(0.0)


@pytest.mark.parametrize(
    "reason,is_policy_step,expected_total",
    [
        (TerminationReason.SUCCESS.value, True, 1.0 - 0.01),
        (TerminationReason.SUCCESS.value, False, 1.0 + 0.0),
        (TerminationReason.FAILURE_COLLISION.value, True, -1.0 - 0.01),
        (TerminationReason.FAILURE_COLLISION.value, False, -1.0 + 0.0),
        (TerminationReason.FAILURE_OFFROAD.value, True, -1.0 - 0.01),
        (TerminationReason.TRUNCATION_HORIZON.value, False, -0.5 + 0.0),
        (TerminationReason.NONE.value, True, 0.0 - 0.01),
        (TerminationReason.NONE.value, False, 0.0 + 0.0),
    ],
)
def test_total_reward_is_terminal_plus_decision_cost(reason, is_policy_step, expected_total):
    value = merge_reward.compute_reward(
        reward_config=REWARD_CONFIG,
        termination_reason=reason,
        is_policy_step=is_policy_step,
    )
    assert value == pytest.approx(expected_total)


@pytest.mark.parametrize("reason", list(TERMINAL_TABLE.keys()) + [None])
@pytest.mark.parametrize("is_policy_step", [True, False])
def test_no_nan_or_inf(reason, is_policy_step):
    value = merge_reward.compute_reward(
        reward_config=REWARD_CONFIG,
        termination_reason=reason,
        is_policy_step=is_policy_step,
    )
    assert math.isfinite(value)
    assert not math.isnan(value)


def test_unrecognized_termination_reason_rejected():
    """Guards SS5.1: this module must only ever map the fixed 5-value
    enum table -- it must not silently accept (and thus implicitly
    invent semantics for) some other string."""

    with pytest.raises(ValueError):
        merge_reward.compute_reward(
            reward_config=REWARD_CONFIG,
            termination_reason="not_a_real_termination_reason",
            is_policy_step=True,
        )


def test_reward_module_does_not_reimplement_termination_detection():
    """Code-level check (SS5.1): merge_reward.py must not import
    anything from src.environment (it must consume only the
    already-computed termination_reason string/enum handed to it, not
    re-derive success/collision/offroad/timeout by importing the
    environment's own detection machinery)."""

    source = inspect.getsource(merge_reward)
    assert "src.environment" not in source
    assert "waymax" not in source

    # Signature-level check: compute_reward takes termination_reason
    # and is_policy_step as explicit inputs (not, e.g., a raw
    # trajectory/state it could inspect itself).
    sig = inspect.signature(merge_reward.compute_reward)
    assert "termination_reason" in sig.parameters
    assert "is_policy_step" in sig.parameters


def test_reward_wrapper_matches_compute_reward():
    """The wrapper's per-step total must agree exactly with the
    underlying compute_reward function it delegates to (no second,
    divergent reward path)."""

    wrapper = MergeRewardWrapper(REWARD_CONFIG)
    for reason, is_policy_step in [
        (TerminationReason.SUCCESS.value, True),
        (TerminationReason.FAILURE_OFFROAD.value, False),
        (TerminationReason.NONE.value, True),
    ]:
        direct = merge_reward.compute_reward(
            reward_config=REWARD_CONFIG,
            termination_reason=reason,
            is_policy_step=is_policy_step,
        )
        via_wrapper = wrapper.compute(termination_reason=reason, is_policy_step=is_policy_step)
        assert via_wrapper == pytest.approx(direct)


def test_reward_wrapper_episode_sums():
    wrapper = MergeRewardWrapper(REWARD_CONFIG)
    wrapper.compute(termination_reason=TerminationReason.NONE.value, is_policy_step=True)
    wrapper.compute(termination_reason=TerminationReason.NONE.value, is_policy_step=False)
    wrapper.compute(termination_reason=TerminationReason.SUCCESS.value, is_policy_step=True)

    sums = wrapper.episode_sums()
    assert sums["reward/decision_cost"] == pytest.approx(-0.01 + 0.0 + -0.01)
    assert sums["reward/terminal"] == pytest.approx(0.0 + 0.0 + 1.0)
    assert sums["reward/total"] == pytest.approx(sums["reward/terminal"] + sums["reward/decision_cost"])

    wrapper.reset()
    assert wrapper.episode_sums() == {
        "reward/terminal": 0.0,
        "reward/decision_cost": 0.0,
        "reward/total": 0.0,
    }
