"""Phase 2 decision-phase / merge-committed-phase state machine.

Stage B-0 Section 4 resolution: this research compares HIGH-LEVEL
BEHAVIOR DECISION-MAKING (Rule-Based FSM vs PPO), not low-level
steering control. Re-selecting among KEEP/FOLLOW/MERGE/STOP every
0.1 s simulation step -- as Stage A's Final Closure Patch originally
proposed -- would blur that high-level semantic: a policy could
"select" MERGE for one step, immediately reconsider, and the
resulting behavior would look more like a low-level steering
oscillation than a committed decision. Instead:

    DECISION phase
        KEEP / FOLLOW / STOP remain freely reselectable every step.
        Selecting MERGE transitions to MERGE_COMMITTED.
    MERGE_COMMITTED phase
        The maneuver is locked in. No further high-level action is
        solicited from the policy -- the shared Behavior Executor
        continues issuing MERGE's objective (BehaviorExecutor
        already computes this identically regardless of who selected
        it) every step until the environment resolves the maneuver to
        SUCCESS or FAILURE (``decision_state.py`` does not itself
        decide success/failure -- see ``termination.py``).

This means the POLICY (FSM or PPO) only makes a fresh choice among
the 4 actions while in DECISION phase; the environment silently keeps
re-issuing the last commitment while in MERGE_COMMITTED. This is
symmetric for FSM and PPO -- neither can re-litigate MERGE once
committed, so this is a constraint on both equally, not a fairness
issue.

STOP remains non-terminal in BOTH phases: choosing STOP while still
in DECISION phase does not transition to MERGE_COMMITTED (STOP != MERGE
-- selecting a different action just means MERGE was never chosen);
STOP has no effect on an already-MERGE_COMMITTED episode, since the
policy is not consulted for a new action there at all.
"""

import dataclasses
import enum

from src.environment.behavior_action import BehaviorAction


class DecisionPhase(enum.Enum):
    """Which phase a maneuver-level episode (Stage A: episode unit =
    ``maneuver_id``) is currently in."""

    DECISION = "decision"
    MERGE_COMMITTED = "merge_committed"


@dataclasses.dataclass
class DecisionState:
    """Mutable per-episode decision-phase state. One instance per
    running episode; the environment owns it and calls
    ``advance`` once per step with whatever action the policy
    selected (only meaningful while ``phase == DECISION``)."""

    phase: DecisionPhase = DecisionPhase.DECISION
    committed_action: BehaviorAction | None = None

    def advance(self, policy_action: BehaviorAction) -> BehaviorAction:
        """Advances the decision state by one step and returns the
        EFFECTIVE action the shared ``BehaviorExecutor`` should use
        this step.

        While in DECISION phase: the effective action is whatever the
        policy just selected. If that selection is MERGE, this call
        transitions ``phase`` to MERGE_COMMITTED and records
        ``committed_action = BehaviorAction.MERGE`` as a side effect,
        but the EFFECTIVE action returned for THIS step is still
        MERGE (the transition takes effect starting next step, not
        retroactively).

        While in MERGE_COMMITTED phase: ``policy_action`` is ignored
        entirely (the policy is not actually consulted once committed
        -- callers should not even present the action choice to the
        policy in this phase, this method's tolerance of an ignored
        argument is only a defensive convenience). The effective
        action returned is always the recorded ``committed_action``
        (always ``BehaviorAction.MERGE`` in the current design, since
        MERGE is the only action that commits).
        """

        if self.phase == DecisionPhase.MERGE_COMMITTED:
            assert self.committed_action is not None
            return self.committed_action

        if policy_action == BehaviorAction.MERGE:
            self.phase = DecisionPhase.MERGE_COMMITTED
            self.committed_action = BehaviorAction.MERGE

        return policy_action

    @property
    def is_committed(self) -> bool:
        return self.phase == DecisionPhase.MERGE_COMMITTED

    @property
    def policy_choice_available(self) -> bool:
        """Whether the environment should solicit a fresh action from
        the policy this step. False once MERGE_COMMITTED -- callers
        should skip the policy call entirely in that case rather than
        calling it and discarding the result, to keep FSM and PPO
        evaluation logs free of meaningless "reselections" during a
        committed maneuver."""

        return self.phase == DecisionPhase.DECISION
