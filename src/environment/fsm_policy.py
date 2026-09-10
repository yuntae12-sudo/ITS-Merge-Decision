"""Phase 2 Stage B-2: Rule-Based FSM behavior-decision baseline.

Reads ONLY the shared 14D observation vector
(``observation_builder.OBSERVATION_FIELD_NAMES``) and outputs ONLY a
``BehaviorAction`` (KEEP/FOLLOW/MERGE/STOP) via the exact same
``BehaviorExecutor``/``LowLevelController``/``MergeEnvironment`` a
future PPO policy will also use. No privileged information: this
module never reads ``MergeEnvironment`` internals, Waymax state, or
anything beyond the 14 observation fields -- verified by
``tests/environment/test_fsm_policy.py``'s fairness-audit test.

Design goal: a SMALL number of interpretable, physically-motivated
thresholds, not an exhaustive rule table -- an FSM with dozens of
handcrafted special cases would make any later FSM-vs-PPO comparison
unfair (the FSM would encode far more domain knowledge than a learned
policy could plausibly discover) and would not be defensible as
"interpretable" in the paper. Every threshold below is either a
standard, literature-grounded driving-safety value (TTC/gap) or a
direct consequence of the environment's own physical parameters
(nominal cruise speed, controller horizon) -- documented at its
definition, not tuned by observing TRAIN or VALIDATION performance.

FSM decision order per step (deterministic, memoryless except for
MERGE commitment which the ENVIRONMENT's own DecisionState already
handles -- this FSM does not re-implement commitment, it only decides
what to propose before commitment, exactly the same task PPO faces):

  1. If the target lane gap is unsafe in front AND in the rear ->
     cannot merge safely right now.
       a. If there is still comfortable stopping margin (d_m large
          relative to how far ego would travel before the merge point
          closes) -> KEEP or FOLLOW (whichever the source-lane lead
          situation calls for) to wait for a better gap.
       b. If stopping margin is nearly exhausted (d_m small) -> STOP
          (the environment's own non-terminal hold/decelerate action)
          rather than run out of source lane while still unsafe.
  2. Otherwise (both target gaps are safe, or no vehicle at all
     occupies them) -> MERGE.
  3. Whenever a close source-lane lead exists (below the FOLLOW gap/TTC
     threshold) and the FSM has not just decided to MERGE or STOP ->
     FOLLOW instead of KEEP, exactly mirroring the physical reason a
     rule-based ACC system would follow rather than cruise blindly.

This intentionally does NOT special-case chained maneuvers: the FSM
only ever sees the CURRENT active (source, target) pair through the
shared observation vector (Stage B-1's EpisodeContext already resolves
which pair is active) -- the same "no special-casing" principle used
throughout Phase 2's environment code.
"""

import dataclasses
import enum

from src.environment.behavior_action import BehaviorAction
from src.environment.observation_builder import (
    OBSERVATION_FIELD_NAMES,
    TTC_CAP_S,
)

# Index constants into the 14D observation vector (mirrors
# OBSERVATION_FIELD_NAMES order; kept explicit here rather than
# re-deriving via .index() at call time for readability at each rule).
_V_E = 0
_D_M = 1
_TARGET_FRONT_PRESENT = 2
_TARGET_FRONT_GAP = 3
_TARGET_FRONT_TTC = 5
_TARGET_REAR_PRESENT = 6
_TARGET_REAR_GAP = 7
_TARGET_REAR_TTC = 9
_SOURCE_FRONT_PRESENT = 10
_SOURCE_FRONT_GAP = 11
_SOURCE_FRONT_TTC = 13

assert len(OBSERVATION_FIELD_NAMES) == 14  # guards the indices above

# --- Frozen thresholds (Stage B-1.5/B-2 Section B4: TRAIN-only, no
# validation-driven tuning) -------------------------------------------
#
# TTC_SAFE_S: minimum time-to-collision against a target-lane vehicle
# (front or rear) for MERGE to be considered safe. 4.0s is a standard,
# widely-cited conservative TTC threshold in car-following/lane-change
# safety literature (comfortably above typical driver reaction time of
# ~1.5s plus margin) -- not derived from this project's own (very
# sparse, Stage A found n=1 finite sample in the 39-row manual-label
# pool) empirical TTC distribution, since that pool is too small to
# support a data-driven threshold.
TTC_SAFE_S = 4.0

# GAP_SAFE_M: minimum absolute gap (independent of TTC) for MERGE to be
# considered safe, guarding against the TTC_CAP_S sentinel (a
# present-but-not-closing vehicle reports the TTC cap, which would
# otherwise always look "safe" by the TTC rule alone even if physically
# adjacent). A conservative single-lane-change minimum gap at highway
# speed (~2 car lengths front and rear).
GAP_SAFE_M = 10.0

# FOLLOW_GAP_M / FOLLOW_TTC_S: below either, a source-lane lead is
# considered "close enough to warrant following" rather than cruising
# at the nominal speed regardless of it. Deliberately the SAME
# thresholds as the merge-safety ones above (one physical notion of
# "too close", reused instead of inventing a second unrelated pair) --
# keeping the total distinct threshold count small per Stage B-2's
# interpretability goal.
FOLLOW_GAP_M = GAP_SAFE_M
FOLLOW_TTC_S = TTC_SAFE_S

# STOP_MARGIN_M: below this remaining merge distance (d_m), if the
# target lane is still unsafe, the FSM chooses STOP instead of
# KEEP/FOLLOW -- physically, "the source lane is about to run out
# while a safe gap still hasn't appeared". Matches
# MIN_SOURCE_BLEND_WINDOW_M's order of magnitude (both describe "not
# much source lane left"), but is defined independently here since it
# governs a high-level decision, not a low-level reference-construction
# detail.
STOP_MARGIN_M = 5.0


class FsmInternalState(enum.Enum):
    """Optional internal bookkeeping state, distinct from the
    BehaviorAction the FSM outputs (Stage B-2 Section B2: the FSM may
    have internal state, but must always emit exactly one of
    KEEP/FOLLOW/MERGE/STOP to the environment). Exposed only for
    diagnostics/tests -- the environment never reads it."""

    CRUISE = "cruise"
    FOLLOWING = "following"
    WAIT_GAP = "wait_gap"
    MERGE_READY = "merge_ready"
    STOPPING = "stopping"


@dataclasses.dataclass(frozen=True)
class FsmDecision:
    action: BehaviorAction
    internal_state: FsmInternalState


def _target_gap_safe(observation) -> bool:
    """True if EITHER the target-lane front slot or the target-lane
    rear slot is absent, or -- when present -- both TTC and gap clear
    their safety thresholds. Both front and rear must independently be
    safe for MERGE to be proposed."""

    front_present = observation[_TARGET_FRONT_PRESENT] == 1.0
    front_safe = (not front_present) or (
        observation[_TARGET_FRONT_TTC] >= TTC_SAFE_S
        or observation[_TARGET_FRONT_GAP] >= GAP_SAFE_M
    )

    rear_present = observation[_TARGET_REAR_PRESENT] == 1.0
    rear_safe = (not rear_present) or (
        observation[_TARGET_REAR_TTC] >= TTC_SAFE_S
        or observation[_TARGET_REAR_GAP] >= GAP_SAFE_M
    )

    return front_safe and rear_safe


def _source_lead_is_close(observation) -> bool:
    source_front_present = observation[_SOURCE_FRONT_PRESENT] == 1.0
    if not source_front_present:
        return False
    return (
        observation[_SOURCE_FRONT_GAP] < FOLLOW_GAP_M
        and observation[_SOURCE_FRONT_TTC] < FOLLOW_TTC_S
    )


class FsmPolicy:
    """Deterministic, memoryless (aside from the environment's own
    MERGE commitment) rule-based baseline. One instance is stateless
    and reusable across many episodes/steps -- exactly like
    ``BehaviorExecutor``, it is a pure function of the current 14D
    observation."""

    def decide(self, observation) -> FsmDecision:
        """Args:
            observation: a 14D vector in OBSERVATION_FIELD_NAMES order
                (the SAME vector a future PPO policy would receive --
                no other input is read).

        Returns:
            FsmDecision (BehaviorAction + diagnostic internal state).
        """

        if len(observation) != 14:
            raise ValueError(
                f"FsmPolicy expects a 14D observation, got length {len(observation)}"
            )

        target_safe = _target_gap_safe(observation)

        if target_safe:
            return FsmDecision(
                action=BehaviorAction.MERGE,
                internal_state=FsmInternalState.MERGE_READY,
            )

        d_m = observation[_D_M]
        if d_m < STOP_MARGIN_M:
            return FsmDecision(
                action=BehaviorAction.STOP,
                internal_state=FsmInternalState.STOPPING,
            )

        if _source_lead_is_close(observation):
            return FsmDecision(
                action=BehaviorAction.FOLLOW,
                internal_state=FsmInternalState.FOLLOWING,
            )

        return FsmDecision(
            action=BehaviorAction.KEEP,
            internal_state=FsmInternalState.WAIT_GAP,
        )
