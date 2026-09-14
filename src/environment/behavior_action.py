"""Phase 2 high-level behavior action enum and semantics.

Fixed by Stage A's Final Closure Patch (Section 4) and revised by
Stage B-0 (Section 3) after the prompt correctly flagged that
collapsing KEEP into FOLLOW via an automatic safety fallback hides
behavior-decision quality behind a downstream safety net -- exactly
the kind of asymmetry that would make an FSM-vs-PPO comparison unfair
if only one side's mistakes were silently corrected.

Four actions, all NON-TERMINAL (episode termination is an
environment-level event -- see ``decision_state.py`` -- never the
result of choosing a particular action):

    KEEP
        Stay in the source lane. Nominal/current cruise-speed
        objective. Does NOT automatically apply source-lane
        lead-following logic, even if a close lead exists -- if KEEP
        is chosen while a lead is close, the executor still targets
        the nominal cruise speed (see ``BehaviorExecutor``), and
        whatever consequence that has (a closing gap, a safety-metric
        penalty) is a genuine, visible outcome of the choice, not
        something masked by silently substituting FOLLOW's behavior.
    FOLLOW
        Stay in the source lane. Explicit lead-following objective
        using ``source_front_gap``/``source_front_relative_speed``
        from the observation. If no source-lane lead exists
        (``source_front_present == 0``), see
        ``FOLLOW_NO_LEAD_FALLBACK`` below for the one explicitly
        decided degenerate-input behavior.
    MERGE
        Commit to the source->target lane transition. Latched/committed
        once selected (Stage B-0 Section 4) -- see ``decision_state.py``
        for the decision-phase / merge-committed-phase state machine
        this actually requires.
    STOP
        Non-terminal hold/strong-deceleration objective in the current
        (source) lane. Selectable repeatedly; does not by itself end
        the episode.

FOLLOW-with-no-lead is the ONE degenerate case Stage B-0 explicitly
decided (Stage A's KEEP-with-close-lead fallback was rejected instead,
per the prompt): FOLLOW_NO_LEAD_FALLBACK = maintain the nominal
cruise speed (the same objective as KEEP), plus a diagnostic flag so
this is visible to evaluation code -- not a silent, semantics-altering
substitution, since "there is nothing to follow" is a simple
factual gap in FOLLOW's own required input, not a safety override of
a different action's choice.

No action masking is added (Stage B-0 explicit instruction: only add
masking if clearly necessary). Both degenerate cases above are handled
by deterministic fallback inside the shared ``BehaviorExecutor``, not
by preventing the policy from selecting an action in the first place.
"""

import dataclasses
import enum


class BehaviorAction(enum.IntEnum):
    """High-level behavior action. Values are stable across
    serialization boundaries (e.g. a PPO discrete-action network
    output or a logged action trace) -- do not renumber."""

    KEEP = 0
    FOLLOW = 1
    MERGE = 2
    STOP = 3


NUM_BEHAVIOR_ACTIONS = 4

# The one explicitly decided degenerate-input behavior (Stage B-0
# Section 3): FOLLOW selected but no source-lane lead exists.
FOLLOW_NO_LEAD_FALLBACK = "maintain_nominal_cruise_speed"


@dataclasses.dataclass(frozen=True)
class BehaviorObjective:
    """The Common Behavior Executor's output for one step: a
    reference speed/lane pair for the shared low-level controller
    (Stage B-0 Section 5) to track. This is the ONLY interface between
    the (FSM- or PPO-produced) discrete action and the shared
    downstream control code -- both policies' chosen action passes
    through the identical ``BehaviorExecutor.compute_objective`` to
    reach this same structure.
    """

    reference_speed_mps: float
    reference_lane: str  # "source" or "target"
    fallback_applied: str | None
    """Non-None when a degenerate-input fallback (see module
    docstring) was applied instead of the action's literal semantic --
    e.g. ``FOLLOW_NO_LEAD_FALLBACK``. None for KEEP/STOP/MERGE, and
    for FOLLOW when a real lead exists. Exposed so evaluation code can
    report fallback frequency as a diagnostic (Stage B-0 Section 3),
    without altering the action taken or applying any reward penalty
    for it here -- reward policy is a separate, later decision.
    """


# Nominal cruise speed used by KEEP and by FOLLOW's no-lead fallback.
# Stage B-0 scope: NOT tuned/derived from data in this task (that is a
# reward/controller-tuning concern for Stage B-1+); a single shared
# module-level constant keeps both actions' "no constraint" behavior
# consistent without inventing a second, slightly different default.
NOMINAL_CRUISE_SPEED_MPS = 15.0

# STOP's deceleration target. Zero (full stop), not a small crawl
# speed -- STOP is described in Stage A/B-0 as "strong deceleration",
# and a literal 0.0 target keeps the low-level controller's behavior
# unambiguous; Stage B-1 may revisit if a crawl speed proves more
# realistic once the controller exists.
STOP_TARGET_SPEED_MPS = 0.0


class BehaviorExecutor:
    """Converts one ``BehaviorAction`` + the current 14D observation
    into one ``BehaviorObjective``. Shared, unconditionally, by both
    the FSM baseline and the PPO policy (Stage A/B-0 fairness
    constraint) -- this class contains no policy logic of its own, it
    only implements the four actions' already-fixed semantics.
    """

    def compute_objective(
        self, action: BehaviorAction, observation
    ) -> BehaviorObjective:
        """Args:
            action: the selected ``BehaviorAction``.
            observation: a 14D vector in
                ``observation_builder.OBSERVATION_FIELD_NAMES`` order
                (accepts anything indexable in that order, e.g. a
                numpy array or a list).

        Returns:
            BehaviorObjective for the shared low-level controller.
        """

        if action == BehaviorAction.KEEP:
            return BehaviorObjective(
                reference_speed_mps=NOMINAL_CRUISE_SPEED_MPS,
                reference_lane="source",
                fallback_applied=None,
            )

        if action == BehaviorAction.FOLLOW:
            source_front_present = observation[10]
            if source_front_present == 0.0:
                return BehaviorObjective(
                    reference_speed_mps=NOMINAL_CRUISE_SPEED_MPS,
                    reference_lane="source",
                    fallback_applied=FOLLOW_NO_LEAD_FALLBACK,
                )

            source_front_gap = observation[11]
            source_front_relative_speed = observation[12]
            follow_speed = self._compute_follow_speed(
                source_front_gap, source_front_relative_speed
            )
            return BehaviorObjective(
                reference_speed_mps=follow_speed,
                reference_lane="source",
                fallback_applied=None,
            )

        if action == BehaviorAction.MERGE:
            target_front_present = observation[2]
            if target_front_present == 0.0:
                merge_speed = NOMINAL_CRUISE_SPEED_MPS
            else:
                target_front_gap = observation[3]
                target_front_relative_speed = observation[4]
                merge_speed = self._compute_follow_speed(
                    target_front_gap, target_front_relative_speed
                )
            return BehaviorObjective(
                reference_speed_mps=merge_speed,
                reference_lane="target",
                fallback_applied=None,
            )

        if action == BehaviorAction.STOP:
            return BehaviorObjective(
                reference_speed_mps=STOP_TARGET_SPEED_MPS,
                reference_lane="source",
                fallback_applied=None,
            )

        raise ValueError(f"Unknown BehaviorAction: {action!r}")

    @staticmethod
    def _compute_follow_speed(
        lead_gap_m: float, lead_relative_speed_mps: float
    ) -> float:
        """Minimal lead-following reference-speed rule (Stage B-0
        scope: NOT the final controller -- Stage B-1's shared
        low-level controller is what actually tracks this reference
        speed; this only decides WHAT speed to target).

        Placeholder-simple by design: targets the nominal cruise speed
        reduced in proportion to closing risk, floored at 0. This is
        deliberately not IDM or any other named car-following model --
        Stage B-1 is expected to refine this once the shared controller
        exists; the important fixed decision here is only that FOLLOW
        and MERGE compute their reference speed the SAME way (both
        call this), not what the exact formula is.
        """

        del lead_gap_m  # reserved for a future gap-aware refinement
        closing_speed = max(lead_relative_speed_mps, 0.0)
        return max(NOMINAL_CRUISE_SPEED_MPS - closing_speed, 0.0)
