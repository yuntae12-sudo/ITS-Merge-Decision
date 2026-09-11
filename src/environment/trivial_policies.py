"""Phase 2 Stage B-2.5: trivial sanity-check policies.

These are NOT proposed as research baselines to compare PPO against
in the final paper -- they exist solely to answer Stage B-2.5's
validity question: does the current MergeEnvironment actually require
a meaningful KEEP/FOLLOW/MERGE/STOP decision, or does a policy that
ignores the observation entirely perform just as well as the
interpretable Rule-FSM?

Each trivial policy shares the identical downstream infrastructure
(MergeEnvironment, BehaviorAction, BehaviorExecutor, LowLevelController,
termination) as the FSM and any future PPO policy -- no privileged
information, no shortcuts.
"""

import dataclasses

from src.environment.behavior_action import BehaviorAction


@dataclasses.dataclass(frozen=True)
class TrivialDecision:
    action: BehaviorAction


class AlwaysMergePolicy:
    """Ignores the observation entirely; always proposes MERGE. If
    this performs comparably to the Rule-FSM, the FSM's conditional
    logic is not actually earning its keep -- the task may be
    degenerate (immediate MERGE is always safe/available)."""

    def decide(self, observation) -> TrivialDecision:
        del observation
        return TrivialDecision(action=BehaviorAction.MERGE)


class AlwaysKeepPolicy:
    """Ignores the observation entirely; always proposes KEEP (never
    attempts to merge). A sanity floor -- if this does nearly as well
    as MERGE-seeking policies on some metric, that metric alone isn't
    informative about merge-decision quality."""

    def decide(self, observation) -> TrivialDecision:
        del observation
        return TrivialDecision(action=BehaviorAction.KEEP)


class AlwaysStopPolicy:
    """Ignores the observation entirely; always proposes STOP.
    Diagnostic only -- checks that STOP is genuinely non-terminal and
    that an always-decelerating policy behaves distinctly from
    always-KEEP (e.g. never reaches MERGE commitment at all)."""

    def decide(self, observation) -> TrivialDecision:
        del observation
        return TrivialDecision(action=BehaviorAction.STOP)
