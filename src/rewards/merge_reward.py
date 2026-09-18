"""Reward V0 computation (docs/ppo/PPO_PLAN.md SS5, SS5.1, SS0.1 P2).

P1 scope: structural skeleton only. The function signature below is
fixed now so P4's rollout loop and P2's real implementation target the
same call contract from the start; the body is NOT implemented in P1
(raises ``NotImplementedError``) -- real reward computation is P2's
job.

When implemented (P2), this module MUST:

- consume ``terminated``, ``truncated``, and
  ``info["termination_reason"]`` as the sole source of truth for the
  terminal outcome (never re-derive SUCCESS/COLLISION/OFFROAD/TIMEOUT
  itself -- SS5.1)
- map outcomes using the frozen enum table:
  ``TerminationReason.SUCCESS -> +1.0``,
  ``TerminationReason.FAILURE_COLLISION -> -1.0``,
  ``TerminationReason.FAILURE_OFFROAD -> -1.0``,
  ``TerminationReason.TRUNCATION_HORIZON -> -0.5``,
  ``TerminationReason.NONE -> 0.0``
- add a decision cost of ``-0.01`` on a real PPO decision step
  (pre-step ``policy_mask == 1``) or ``0.0`` on an auto-executed
  MERGE-commitment step (``policy_mask == 0``), decided PRE-step, never
  derived from post-step state (SS5.1, SS7.1)
- return ``terminal_outcome + decision_cost``, and never NaN/inf

It must NOT implement a merge-success detector, an overlap/collision
detector, an offroad detector, or an episode-timeout detector -- those
are exclusively the frozen ``MergeEnvironment``'s job.
"""

from typing import Any, Dict

from src.training.config import RewardConfig


def compute_reward(
    reward_config: RewardConfig,
    termination_reason: str,
    is_policy_step: bool,
    info: Dict[str, Any] = None,
) -> float:
    """Computes one step's Reward V0 value.

    P1 skeleton: not yet implemented. Real body lands in P2 per
    PPO_PLAN.md SS0.1/P2 and SS5.1. Kept as a raising stub (rather than
    silently returning ``0.0``) so P1's import/structure tests can
    confirm the module and signature exist without a caller ever
    mistaking a stub 0.0 for a real reward value.
    """

    raise NotImplementedError(
        "Reward V0 computation lands in P2 (docs/ppo/PPO_PLAN.md "
        "SS0.1/P2). P1 only defines this module's structure/signature."
    )
