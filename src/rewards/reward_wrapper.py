"""Environment-facing reward wrapper (docs/ppo/PPO_PLAN.md SS0.1 P2;
Reward V1 extension per outputs/reward_v1_spec/REWARD_V1_SPEC_FINAL.md).

P2 implementation. Thin call-site glue that a rollout loop (P4) uses
each step to turn one ``MergeEnvironment`` step's outputs into a Reward
value, via ``src.rewards.merge_reward.compute_reward_breakdown`` --
never re-deriving termination itself (SS5.1), and never recomputing any
component a second, divergent way. Also accumulates the
``reward/terminal``, ``reward/decision_cost``, ``reward/safety``,
``reward/progress``, ``reward/decision``, ``reward/total`` diagnostic
components (PPO_PLAN.md SS8) for the *last* computed step, and a running
per-episode sum of each, so a rollout loop / W&B logger can read them
back without recomputing anything.

Reward V1 stateful lifecycle this wrapper owns (a V0 ``RewardConfig``
never touches either of these -- both stay ``None``/unused):

- ``d_m_initial``: this episode's ``d_m`` (observation index 1) value
  captured the FIRST time ``compute`` is called this episode (i.e. at
  the first step after reset, using that step's pre-step
  ``observation``) -- fixed for the rest of the episode, cleared only by
  ``reset()``. Never recomputed mid-episode, so
  ``compute_progress_potential``'s normalization never drifts.
- ``previous_policy_action``: the most recent REAL policy decision's
  action (only updated on an ``is_policy_step=True`` call) -- an
  auto-executed MERGE-commitment step's action is never written here, so
  Reward V1's Decision component's "did the policy just switch its
  choice" question is answered only by comparing two genuine policy
  decisions, never contaminated by an automatic action. Cleared only by
  ``reset()``.
"""

import math
from typing import Any, Dict, List, Optional

from src.rewards.merge_reward import compute_reward_breakdown
from src.training.config import RewardConfig


class MergeRewardWrapper:
    """Wraps step outputs from ``MergeEnvironment`` and computes the
    configured Reward (V0 or V1) for that step.

    Stateful for two reasons: (1) rollout-loop diagnostics accumulation
    (PPO_PLAN.md SS8, unchanged from V0), and (2) Reward V1's
    ``d_m_initial``/``previous_policy_action`` per-episode lifecycle (see
    module docstring). ``compute`` itself still delegates all actual
    reward arithmetic to ``merge_reward.compute_reward_breakdown`` -- it
    performs no independent reward computation of its own.
    """

    def __init__(self, reward_config: RewardConfig):
        self.reward_config = reward_config
        self.last_terminal_component: Optional[float] = None
        self.last_decision_cost_component: Optional[float] = None
        self.last_safety_component: Optional[float] = None
        self.last_progress_component: Optional[float] = None
        self.last_decision_component: Optional[float] = None
        self.last_ttc_penalty_raw: Optional[float] = None
        self.last_gap_penalty_raw: Optional[float] = None
        self.last_total: Optional[float] = None

        self._terminal_history: List[float] = []
        self._decision_cost_history: List[float] = []
        self._safety_history: List[float] = []
        self._progress_history: List[float] = []
        self._decision_history: List[float] = []
        self._total_history: List[float] = []

        self._d_m_initial: Optional[float] = None
        self._previous_policy_action: Any = None

    def compute(
        self,
        termination_reason: str,
        is_policy_step: bool,
        info: Dict[str, Any] = None,
        observation: Any = None,
        next_observation: Any = None,
        action: Any = None,
    ) -> float:
        """Computes and returns this step's Reward total, and records
        every component for later retrieval (``reward/terminal``,
        ``reward/decision_cost``, ``reward/safety``, ``reward/progress``,
        ``reward/decision``, ``reward/total`` per PPO_PLAN.md SS8 +
        REWARD_V1_SPEC_FINAL.md Section 13). ``observation``/
        ``next_observation``/``action`` are ignored entirely for a V0
        ``reward_config`` (kept optional so existing V0 call sites need
        no change).
        """

        if self.reward_config.progress is not None and observation is not None:
            if self._d_m_initial is None:
                # First compute() call this episode (post-reset) --
                # capture d_m_initial from THIS step's pre-step
                # observation, per the module docstring's lifecycle.
                progress_cfg = self.reward_config.progress
                d_m_index = 1  # mirrors observation_builder.OBSERVATION_FIELD_NAMES[1] == "d_m"
                self._d_m_initial = float(observation[d_m_index])

        breakdown = compute_reward_breakdown(
            reward_config=self.reward_config,
            termination_reason=termination_reason,
            is_policy_step=is_policy_step,
            observation=observation,
            next_observation=next_observation,
            action=action,
            previous_policy_action=self._previous_policy_action,
            d_m_initial=self._d_m_initial,
        )

        self.last_terminal_component = breakdown["terminal"]
        self.last_decision_cost_component = breakdown["decision_cost"]
        self.last_safety_component = breakdown["safety_weighted"]
        self.last_progress_component = breakdown["progress_weighted"]
        self.last_decision_component = breakdown["decision_weighted"]
        self.last_ttc_penalty_raw = breakdown["ttc_penalty_raw"]
        self.last_gap_penalty_raw = breakdown["gap_penalty_raw"]
        self.last_total = breakdown["total"]

        self._terminal_history.append(breakdown["terminal"])
        self._decision_cost_history.append(breakdown["decision_cost"])
        self._safety_history.append(breakdown["safety_weighted"])
        self._progress_history.append(breakdown["progress_weighted"])
        self._decision_history.append(breakdown["decision_weighted"])
        self._total_history.append(breakdown["total"])

        # Reward Breakdown Accounting Contract: this wrapper's own
        # component sum must agree with compute_reward_breakdown's total
        # -- both derive from the same computation, so any mismatch
        # would indicate a bug in this wrapper, not a legitimate second
        # reward path.
        component_sum = (
            breakdown["terminal"]
            + breakdown["decision_cost"]
            + breakdown["safety_weighted"]
            + breakdown["progress_weighted"]
            + breakdown["decision_weighted"]
        )
        if not math.isclose(component_sum, breakdown["total"], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                "MergeRewardWrapper: component sum "
                f"({component_sum}) does not match compute_reward_"
                f"breakdown's total ({breakdown['total']})."
            )

        # Only a REAL policy decision updates previous_policy_action --
        # an auto-executed MERGE-commitment step (is_policy_step=False)
        # must never overwrite it (see module docstring).
        if is_policy_step:
            self._previous_policy_action = action

        return breakdown["total"]

    def episode_sums(self) -> Dict[str, float]:
        """Returns the running per-episode sums of each reward
        component accumulated so far via ``compute`` calls, keyed by
        the PPO_PLAN.md SS8 / REWARD_V1_SPEC_FINAL.md Section 13 metric
        names."""

        return {
            "reward/terminal": float(sum(self._terminal_history)),
            "reward/decision_cost": float(sum(self._decision_cost_history)),
            "reward/safety": float(sum(self._safety_history)),
            "reward/progress": float(sum(self._progress_history)),
            "reward/decision": float(sum(self._decision_history)),
            "reward/total": float(sum(self._total_history)),
        }

    def reset(self) -> None:
        """Clears per-episode accumulation AND the Reward V1
        ``d_m_initial``/``previous_policy_action`` lifecycle state (call
        at each episode reset)."""

        self.last_terminal_component = None
        self.last_decision_cost_component = None
        self.last_safety_component = None
        self.last_progress_component = None
        self.last_decision_component = None
        self.last_ttc_penalty_raw = None
        self.last_gap_penalty_raw = None
        self.last_total = None

        self._terminal_history.clear()
        self._decision_cost_history.clear()
        self._safety_history.clear()
        self._progress_history.clear()
        self._decision_history.clear()
        self._total_history.clear()

        self._d_m_initial = None
        self._previous_policy_action = None
