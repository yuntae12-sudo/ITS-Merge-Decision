"""Environment-facing reward wrapper (docs/ppo/PPO_PLAN.md SS0.1 P2).

P1 scope: structural skeleton only. This module will wrap a
``MergeEnvironment`` step's outputs, feeding ``terminated``,
``truncated``, ``info["termination_reason"]``, and the pre-step
``policy_mask``/``is_policy_step`` decision into
``src.rewards.merge_reward.compute_reward`` -- never re-deriving
termination itself (SS5.1). Real implementation lands in P2.
"""

from typing import Any, Dict

from src.training.config import RewardConfig


class MergeRewardWrapper:
    """Wraps step outputs from ``MergeEnvironment`` and computes Reward
    V0 for that step.

    P1 skeleton: stores the reward config; ``compute`` is not yet
    implemented (lands in P2). Kept as a thin class here (rather than a
    bare function) since P4's rollout loop needs a stateful place to
    accumulate reward diagnostics (``reward/terminal``,
    ``reward/decision_cost``, ``reward/total`` per SS8) across a
    rollout -- that accumulation logic also lands in P2/P4, not P1.
    """

    def __init__(self, reward_config: RewardConfig):
        self.reward_config = reward_config

    def compute(
        self,
        termination_reason: str,
        is_policy_step: bool,
        info: Dict[str, Any] = None,
    ) -> float:
        raise NotImplementedError(
            "Reward wrapper computation lands in P2 (docs/ppo/PPO_PLAN.md "
            "SS0.1/P2). P1 only defines this module's structure/signature."
        )
