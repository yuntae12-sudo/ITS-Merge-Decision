"""Environment-facing reward wrapper (docs/ppo/PPO_PLAN.md SS0.1 P2).

P2 implementation. Thin call-site glue that a rollout loop (P4) uses
each step to turn one ``MergeEnvironment`` step's outputs into a
Reward V0 value, via ``src.rewards.merge_reward.compute_reward`` --
never re-deriving termination itself (SS5.1). Also accumulates the
``reward/terminal``, ``reward/decision_cost``, ``reward/total``
diagnostic components (PPO_PLAN.md SS8) for the *last* computed step,
and a running per-episode sum of each, so a rollout loop / W&B logger
can read them back without recomputing anything.
"""

import math
from typing import Any, Dict, List, Optional

from src.rewards.merge_reward import compute_reward
from src.training.config import RewardConfig


class MergeRewardWrapper:
    """Wraps step outputs from ``MergeEnvironment`` and computes Reward
    V0 for that step.

    Stateful only for convenience (rollout-loop diagnostics
    accumulation, PPO_PLAN.md SS8) -- ``compute`` itself is a pure
    function of its arguments plus ``self.reward_config`` and performs
    no environment-state inspection of its own.
    """

    def __init__(self, reward_config: RewardConfig):
        self.reward_config = reward_config
        self.last_terminal_component: Optional[float] = None
        self.last_decision_cost_component: Optional[float] = None
        self.last_total: Optional[float] = None
        self._terminal_history: List[float] = []
        self._decision_cost_history: List[float] = []
        self._total_history: List[float] = []

    def compute(
        self,
        termination_reason: str,
        is_policy_step: bool,
        info: Dict[str, Any] = None,
    ) -> float:
        """Computes and returns this step's Reward V0 total, and
        records the terminal/decision-cost components for later
        retrieval (``reward/terminal``, ``reward/decision_cost``,
        ``reward/total`` per PPO_PLAN.md SS8).
        """

        # Component breakdown mirrors merge_reward.compute_reward's
        # internal lookups exactly -- recomputed here (not re-derived
        # independently) purely so the two components can be reported
        # separately for W&B logging, per SS8's reward/terminal and
        # reward/decision_cost metric names.
        terminal_table = self.reward_config.terminal
        terminal_by_reason = {
            "success": terminal_table.success,
            "failure_collision": terminal_table.failure_collision,
            "failure_offroad": terminal_table.failure_offroad,
            "truncation_horizon": terminal_table.truncation_horizon,
            "none": terminal_table.none,
        }
        reason_key = termination_reason if termination_reason is not None else "none"
        terminal_component = float(terminal_by_reason[reason_key])

        decision_cost_table = self.reward_config.decision_cost
        decision_cost_component = float(
            decision_cost_table.real_decision_step
            if is_policy_step
            else decision_cost_table.auto_execution_step
        )

        total = compute_reward(
            reward_config=self.reward_config,
            termination_reason=termination_reason,
            is_policy_step=is_policy_step,
            info=info,
        )

        # The wrapper's own component sum must agree with
        # merge_reward.compute_reward's total -- both derive from the
        # same fixed config table, so any mismatch would indicate a
        # bug in this wrapper, not a legitimate second reward path.
        if not math.isclose(terminal_component + decision_cost_component, total, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                "MergeRewardWrapper: component sum "
                f"({terminal_component} + {decision_cost_component}) does "
                f"not match compute_reward's total ({total})."
            )

        self.last_terminal_component = terminal_component
        self.last_decision_cost_component = decision_cost_component
        self.last_total = total
        self._terminal_history.append(terminal_component)
        self._decision_cost_history.append(decision_cost_component)
        self._total_history.append(total)

        return total

    def episode_sums(self) -> Dict[str, float]:
        """Returns the running per-episode sums of each reward
        component accumulated so far via ``compute`` calls, keyed by
        the PPO_PLAN.md SS8 metric names."""

        return {
            "reward/terminal": float(sum(self._terminal_history)),
            "reward/decision_cost": float(sum(self._decision_cost_history)),
            "reward/total": float(sum(self._total_history)),
        }

    def reset(self) -> None:
        """Clears per-episode accumulation (call at each episode
        reset)."""

        self.last_terminal_component = None
        self.last_decision_cost_component = None
        self.last_total = None
        self._terminal_history.clear()
        self._decision_cost_history.clear()
        self._total_history.clear()
