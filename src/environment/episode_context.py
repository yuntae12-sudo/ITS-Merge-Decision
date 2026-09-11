"""Phase 2 chained-maneuver runtime semantics (Stage B-1, Section 1 --
the first gate this stage must clear before any environment code).

Stage A's hierarchy audit found 8/168 maneuvers are CHAINED: one
``maneuver_id`` spans >=2 candidate transitions whose lane_chain reads
e.g. ``607->595->523`` (candidate 1: source=607/target=595, candidate
2: source=595/target=523 -- the first candidate's target IS the
second's source). The observation/action interface needs exactly one
(source_lane, target_lane) pair active at any simulated frame, so this
module defines how that pair advances as ego moves through the chain.

Principle (fixed by the Stage B-1 prompt, non-negotiable): reaching an
INTERMEDIATE target lane in the chain is NOT maneuver success --
success only occurs once ego stably enters the FINAL target lane
(``lane_chain[-1]``). A single-candidate maneuver is just the
degenerate case of this same model with a chain of length 2
(``[source, target]``) and zero intermediate transitions -- so both
160 single-candidate maneuvers and 8 chained maneuvers use the
IDENTICAL EpisodeContext/advancement logic; no special-casing by
transition count anywhere else in the environment.
"""

import dataclasses
from typing import List


@dataclasses.dataclass
class EpisodeContext:
    """Runtime state for one running episode (one maneuver_id).

    Immutable fields describe the maneuver itself (from Phase 1's
    manifest/maneuver tables); ``active_transition_index`` is the only
    mutable field, advanced by ``advance_if_intermediate_reached``.
    """

    maneuver_id: str
    scene_key: str
    candidate_ids: List[str]
    lane_chain: List[int]
    """[source_lane_0, target_lane_0(=source_lane_1), target_lane_1, ...,
    target_lane_final]. Length == len(candidate_ids) + 1. For a
    single-candidate maneuver this is exactly [source, target]."""

    merge_start_frame: int
    """The FIRST candidate's merge_start_frame -- Phase 1's canonical
    physical merge-region reference. NOT the simulation/episode start
    (see ``decision_start_frame`` below, Stage B-2.8); kept unchanged
    for merge-region geometry and as the causal-success progress
    reference maneuver tables already store it as."""

    decision_start_frame: int
    """Phase 2 Stage B-2.8: the actual simulation/policy episode start
    frame (``src.environment.decision_window.compute_decision_start_
    frame``'s result) -- the earliest causal frame at which the raw
    lane assignment already matches the source lane. Always
    <= merge_start_frame for a resolvable maneuver."""

    active_transition_index: int = 0
    """Which (source, target) pair in the chain is currently active.
    0 for every maneuver at episode start, including single-candidate
    ones. Advances by exactly 1 each time ego stably enters an
    INTERMEDIATE target lane; never advances past the final index."""

    def __post_init__(self):
        if len(self.lane_chain) != len(self.candidate_ids) + 1:
            raise ValueError(
                "lane_chain length must be len(candidate_ids) + 1: "
                f"got lane_chain={self.lane_chain}, "
                f"candidate_ids={self.candidate_ids}"
            )
        if not self.candidate_ids:
            raise ValueError("EpisodeContext requires >=1 candidate_id")

    @property
    def num_transitions(self) -> int:
        return len(self.candidate_ids)

    @property
    def is_chained(self) -> bool:
        return self.num_transitions > 1

    @property
    def active_source_lane_id(self) -> int:
        return self.lane_chain[self.active_transition_index]

    @property
    def active_target_lane_id(self) -> int:
        return self.lane_chain[self.active_transition_index + 1]

    @property
    def final_target_lane_id(self) -> int:
        return self.lane_chain[-1]

    @property
    def is_on_final_transition(self) -> bool:
        """True once the active pair's target IS the maneuver's final
        target -- i.e. there is no further intermediate lane to pass
        through. Always True for a single-candidate maneuver from the
        start (active_transition_index=0, num_transitions=1)."""

        return self.active_transition_index == self.num_transitions - 1

    def advance_if_intermediate_reached(self, current_stable_lane_id) -> bool:
        """Called once per step with the CURRENT (causal, this-frame)
        stable lane assignment. If ego has stably entered the CURRENT
        active target lane AND that target is an INTERMEDIATE one (not
        the final target), advances ``active_transition_index`` by 1
        and returns True. Otherwise leaves state unchanged and returns
        False.

        Deliberately does NOT itself decide "stably entered" (that is
        ``termination.check_online_causal_merge_success``'s job,
        reused here with the ACTIVE target lane, not the final one --
        see ``merge_environment.py`` for the exact call). This method
        only implements the chain-advancement RULE once that
        per-transition success has already been determined by the
        caller for the currently-active target.
        """

        if self.is_on_final_transition:
            return False  # no intermediate lane left to advance past

        if current_stable_lane_id != self.active_target_lane_id:
            return False

        self.active_transition_index += 1
        return True


def parse_lane_chain(lane_chain_str: str) -> List[int]:
    """Parses a maneuver table's ``lane_chain`` field, e.g.
    ``"607->595->523"``, into ``[607, 595, 523]``."""

    return [int(part) for part in lane_chain_str.split("->")]


def parse_candidate_ids(candidate_ids_str: str) -> List[str]:
    """Parses a maneuver table's semicolon-separated ``candidate_ids``
    field into a list, in chain order (matching ``lane_chain``)."""

    return [part.strip() for part in candidate_ids_str.split(";") if part.strip()]
