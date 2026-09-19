"""Reward V0 computation (docs/ppo/PPO_PLAN.md SS5, SS5.1, SS0.1 P2).

P2 implementation. This module maps the frozen ``MergeEnvironment``'s
own ``terminated``/``truncated``/``info["termination_reason"]`` to the
fixed Reward V0 terminal-outcome table, and adds a pre-step decision
cost. It never re-derives SUCCESS/COLLISION/OFFROAD/TIMEOUT itself --
that detection is exclusively the frozen ``MergeEnvironment``'s job
(``src/environment/termination.py``). This module only consumes the
already-computed ``info["termination_reason"]`` string value (see
``MergeEnvironment._build_info``, which sets it to
``termination_reason.value if termination_reason else None`` -- i.e.
one of ``"success"``, ``"failure_collision"``, ``"failure_offroad"``,
``"truncation_horizon"``, ``"none"``, or ``None``).

Reward V0::

    R = terminal_outcome + decision_cost

Terminal-outcome table (``TerminationReason`` values, from
``src/environment/termination.py:48-53``, the only five values it
defines)::

    TerminationReason.SUCCESS            -> +1.0
    TerminationReason.FAILURE_COLLISION  -> -1.0
    TerminationReason.FAILURE_OFFROAD    -> -1.0
    TerminationReason.TRUNCATION_HORIZON -> -0.5
    TerminationReason.NONE               -> 0.0

Decision cost: ``-0.01`` on a real PPO decision step
(pre-step ``is_policy_step is True``), ``0.0`` on an auto-executed
MERGE-commitment step (``is_policy_step is False``). This flag MUST be
computed by the caller BEFORE calling ``env.step()`` (PPO_PLAN.md
SS7.1) and passed in here -- this module never infers it from
post-step state (e.g. it never looks at ``info["merge_committed"]``
itself to decide the cost).

Explicitly forbidden inside this module (SS5.1): a merge-success
detector, an overlap/collision detector, an offroad detector, an
episode-timeout detector. This module only performs a fixed dict
lookup against the ``termination_reason`` string the environment
already computed.
"""

import math
from typing import Any, Dict, Optional

from src.training.config import RewardConfig

# String keys are exactly the ``TerminationReason.value`` strings
# defined in src/environment/termination.py:48-53. No other values are
# ever produced by the frozen MergeEnvironment.
_TERMINATION_REASON_VALUES = (
    "success",
    "failure_collision",
    "failure_offroad",
    "truncation_horizon",
    "none",
)


def _terminal_outcome(reward_config: RewardConfig, termination_reason: Optional[str]) -> float:
    """Looks up the fixed terminal-outcome value for one termination
    reason string. Pure table lookup -- no detection logic.

    ``termination_reason`` may be ``None`` (mirrors
    ``MergeEnvironment``'s own ``info["termination_reason"]`` before
    any ``TerminationResult`` has ever been computed for this episode,
    e.g. at ``reset()`` time) -- treated the same as
    ``TerminationReason.NONE`` ("no terminal outcome yet").
    """

    if termination_reason is None:
        termination_reason = "none"

    if termination_reason not in _TERMINATION_REASON_VALUES:
        raise ValueError(
            "merge_reward: unrecognized termination_reason "
            f"{termination_reason!r}. This module only maps the fixed "
            "TerminationReason enum values from "
            "src/environment/termination.py:48-53 "
            f"({_TERMINATION_REASON_VALUES}); it must never receive or "
            "invent any other value (PPO_PLAN.md SS5.1)."
        )

    terminal = reward_config.terminal
    return {
        "success": terminal.success,
        "failure_collision": terminal.failure_collision,
        "failure_offroad": terminal.failure_offroad,
        "truncation_horizon": terminal.truncation_horizon,
        "none": terminal.none,
    }[termination_reason]


def _decision_cost(reward_config: RewardConfig, is_policy_step: bool) -> float:
    """Looks up the fixed decision-cost value for one step, selected
    purely by the caller-supplied pre-step ``is_policy_step`` flag
    (PPO_PLAN.md SS7.1/SS5.1) -- never derived from post-step state.
    """

    cost = reward_config.decision_cost
    return cost.real_decision_step if is_policy_step else cost.auto_execution_step


def compute_reward(
    reward_config: RewardConfig,
    termination_reason: str,
    is_policy_step: bool,
    info: Dict[str, Any] = None,
) -> float:
    """Computes one step's Reward V0 value.

    Args:
        reward_config: loaded via
            ``src.training.config.load_reward_config`` -- the sole
            source of the fixed V0 numeric values (never hardcoded a
            second time in this function body).
        termination_reason: the frozen ``MergeEnvironment``'s own
            ``info["termination_reason"]`` value for this step (a
            ``TerminationReason.value`` string, or ``None``). This
            function performs a fixed lookup against this value; it
            never recomputes it.
        is_policy_step: pre-step decision flag (PPO_PLAN.md SS7.1) --
            ``True`` on a real PPO decision step, ``False`` on an
            auto-executed MERGE-commitment step. Must be computed by
            the caller BEFORE ``env.step()`` from
            ``info_before["merge_committed"]`` (``is_policy_step = not
            info_before["merge_committed"]``); this function does not
            infer it itself.
        info: accepted for call-signature symmetry with
            ``MergeRewardWrapper.compute`` and future callers that may
            want to pass the full step ``info`` dict through for
            diagnostics; NOT read for reward computation itself in V0
            (V0 needs only ``termination_reason`` + ``is_policy_step``,
            both already passed explicitly) -- kept unused here rather
            than silently reading extra fields out of it, since doing
            so would risk exactly the kind of independent
            re-derivation SS5.1 forbids.

    Returns:
        ``terminal_outcome + decision_cost`` as a finite Python float.
    """

    del info  # unused in V0 -- see docstring; kept for signature symmetry.

    terminal_outcome = _terminal_outcome(reward_config, termination_reason)
    decision_cost = _decision_cost(reward_config, is_policy_step)

    total = float(terminal_outcome) + float(decision_cost)

    if not math.isfinite(total):
        raise ValueError(
            f"merge_reward.compute_reward produced a non-finite reward "
            f"({total!r}) for termination_reason={termination_reason!r}, "
            f"is_policy_step={is_policy_step!r}. This should be "
            "unreachable given the fixed V0 config values."
        )

    return total
