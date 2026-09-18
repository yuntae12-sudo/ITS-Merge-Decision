"""Rollout collection against the real ``MergeEnvironment`` (docs/ppo/
PPO_PLAN.md SS0.1 P4, SS7.1).

P1 scope: structural skeleton only. Fixes the minimum rollout
transition field set (SS0.1/P4) so P4's real implementation and any
earlier test scaffolding agree on shape from the start:

    observation, action, reward, next_observation, terminated,
    truncated, value, next_value, log_prob, policy_mask

Optional diagnostics: ``episode_id``, ``maneuver_id``, ``step_index``.

``policy_mask`` MUST be decided PRE-step (SS7.1) -- never from the
post-step ``info`` dict. Real rollout-loop implementation (which will
mirror the pre-step ``info_before["merge_committed"]`` pattern already
used by ``src.environment.full_split_evaluator.run_episode``, confirmed
during P0) lands in P4.
"""

import dataclasses
from typing import Any, List, Optional

# Minimum required transition fields (PPO_PLAN.md SS0.1/P4).
REQUIRED_TRANSITION_FIELDS = (
    "observation",
    "action",
    "reward",
    "next_observation",
    "terminated",
    "truncated",
    "value",
    "next_value",
    "log_prob",
    "policy_mask",
)


@dataclasses.dataclass(frozen=True)
class Transition:
    """One rollout transition. Optional diagnostic fields default to
    ``None`` so P4 can populate them without breaking this contract."""

    observation: Any
    action: Any
    reward: float
    next_observation: Any
    terminated: bool
    truncated: bool
    value: float
    next_value: float
    log_prob: float
    policy_mask: int
    episode_id: Optional[str] = None
    maneuver_id: Optional[str] = None
    step_index: Optional[int] = None


def collect_rollout(*args, **kwargs) -> List[Transition]:
    """Runs the PPO policy against ``MergeEnvironment`` and collects a
    list of ``Transition``s.

    P1 skeleton: not yet implemented. Real implementation (including
    the pre-step ``policy_mask`` decision pattern per SS7.1) lands in
    P4.
    """

    raise NotImplementedError(
        "Rollout collection against MergeEnvironment lands in P4 "
        "(docs/ppo/PPO_PLAN.md SS0.1/P4). P1 only fixes the Transition "
        "field contract."
    )
