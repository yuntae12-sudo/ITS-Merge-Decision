"""Phase 2 episode termination: SUCCESS / FAILURE / TRUNCATION.

Stage B-0 Section 6/8 resolution. Confirms (does not reimplement) that
``src.scenarios.lane_assignment.assign_ego_lane_sequence`` and
``compute_stable_lane_sequence`` are CAUSAL: both iterate strictly
forward over the frame index using only already-seen frames (a single
running `current_stable`/`pending_lane`/`pending_run_length`/
`ambiguous_gap_length` state, updated in frame order, with no second
pass or lookahead). This was verified by reading the actual loop body
in ``lane_assignment.py`` (lines ~340-409), not assumed. Because they
are causal, this module reuses them AS-IS for a live simulated-ego
success check, called incrementally each simulation step on the
trajectory accumulated so far -- never on future simulated frames.

``find_lane_transitions`` is likewise a simple forward scan over an
already-causal stable sequence; calling it on a growing prefix each
step is equally causal.

Stage A's original proposal to reuse this logic is therefore
CONFIRMED, not replaced -- the only thing Stage B-0 adds is (a) this
explicit causality verification, on record, and (b) wiring it into an
incremental, step-by-step SUCCESS check rather than the offline
whole-trajectory call Phase 1's own CLIs use.

Role of logged ``transition_frame``/``merge_complete_frame`` (Stage
A/B-0, unchanged): REFERENCE ONLY. Used only to pick the maneuver's
``episode_start`` (Stage A, unchanged) and as an after-the-fact
diagnostic comparison ("did the simulated rollout merge faster/slower
than the logged human driver") -- never as the live termination
trigger itself, since a PPO- or FSM-controlled ego can legitimately
merge earlier or later than the logged trajectory did.
"""

import dataclasses
import enum
from typing import List, Optional

import numpy as np

from src.scenarios.lane_assignment import (
    LaneAssignmentConfig,
    assign_ego_lane_sequence,
    compute_stable_lane_sequence,
)
from src.scenarios.lane_geometry import LanePolyline


class TerminationReason(enum.Enum):
    NONE = "none"
    SUCCESS = "success"
    FAILURE_COLLISION = "failure_collision"
    FAILURE_OFFROAD = "failure_offroad"
    TRUNCATION_HORIZON = "truncation_horizon"


@dataclasses.dataclass(frozen=True)
class TerminationResult:
    reason: TerminationReason
    done: bool
    """True for SUCCESS, both FAILURE reasons, and TRUNCATION_HORIZON.
    False for NONE (episode continues)."""


def check_online_causal_merge_success(
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_yaw: np.ndarray,
    ego_valid: np.ndarray,
    target_lane_id: int,
    polylines: List[LanePolyline],
    lane_assignment_config: LaneAssignmentConfig,
    episode_start_frame: int,
) -> bool:
    """Online-causal simulated merge-success check (Stage B-0 Section
    6/8), evaluated at the CURRENT last frame of ``ego_x``/... (i.e.
    callers pass arrays sliced up to and including "now" -- never
    frames beyond the current simulation step).

    Reuses ``assign_ego_lane_sequence``/``compute_stable_lane_sequence``
    exactly as Phase 1 already validated them (same config, same
    persistence/ambiguous-gap semantics) -- confirmed causal, see
    module docstring.

    Success criterion (Stage A's completion-criterion candidates,
    Stage B-0 finalized):
        - the stable lane assignment at the CURRENT (last) frame
          equals ``target_lane_id`` (lateral distance and heading
          -difference gates are already enforced inside
          ``assign_ego_lane_sequence`` via ``lane_assignment_config``,
          reusing Phase 1's own validated thresholds -- not
          re-derived here), AND
        - that stable assignment has held for at least
          ``lane_assignment_config.persistence_frames`` consecutive
          frames (enforced by ``compute_stable_lane_sequence`` itself
          -- a fresh switch is not trusted immediately), AND
        - net longitudinal progress since ``episode_start_frame``: the
          ego's arc-length position on ``target_lane_id`` at the
          current frame is greater at the current frame than it was
          at the first frame where the stable sequence first equalled
          target_lane_id (guards against a momentary lateral wobble
          counting as completion -- Stage A's stated concern).

    Args:
        ego_x, ego_y, ego_yaw, ego_valid: (T,) arrays of the SIMULATED
            ego trajectory from episode start through the current
            frame only (T = current frame index - episode_start_frame
            + 1, or any causal-consistent slice -- callers must never
            pass frames beyond "now").
        target_lane_id: the maneuver's target lane.
        polylines: the scene's lane polylines (static per scene, not
            per frame).
        lane_assignment_config: reused unmodified from Phase 1.
        episode_start_frame: the ABSOLUTE frame index ``ego_x[0]``
            corresponds to (for progress-since-start bookkeeping only;
            does not affect causality).

    Returns:
        True iff the success criterion is met at the current (last)
        frame of the provided arrays.
    """

    del episode_start_frame  # reserved: absolute-frame bookkeeping by
    # callers that track progress across a rolling buffer; this
    # function only needs the relative array, since progress-since
    # -start is evaluated by comparing to the FIRST frame index where
    # target_lane_id was seen (below), which is already relative.

    raw_assignments = assign_ego_lane_sequence(
        ego_x, ego_y, ego_yaw, ego_valid, polylines, lane_assignment_config
    )
    stable_sequence = compute_stable_lane_sequence(
        raw_assignments,
        persistence_frames=lane_assignment_config.persistence_frames,
        max_ambiguous_gap_frames=(
            lane_assignment_config.max_ambiguous_gap_frames
        ),
    )

    current_frame = len(stable_sequence) - 1
    if current_frame < 0:
        return False

    if stable_sequence[current_frame] != target_lane_id:
        return False

    first_target_frame = next(
        (
            i
            for i, lane_id in enumerate(stable_sequence)
            if lane_id == target_lane_id
        ),
        None,
    )
    if first_target_frame is None:
        return False  # unreachable given the check above, defensive only

    first_arc_length = raw_assignments[first_target_frame].arc_length_m
    current_arc_length = raw_assignments[current_frame].arc_length_m

    if first_arc_length is None or current_arc_length is None:
        return False

    return current_arc_length > first_arc_length


@dataclasses.dataclass(frozen=True)
class TerminationInputs:
    """Everything ``check_termination`` needs for one step."""

    success: bool
    """From ``check_online_causal_merge_success`` (only meaningful
    once the episode is in MERGE_COMMITTED phase -- see
    ``decision_state.py``; a non-committed episode cannot succeed)."""

    collision: bool
    """From ``waymax.metrics.overlap`` (Stage B-0 Section 5, confirmed
    reusable -- not reimplemented here)."""

    offroad: bool
    """From ``waymax.metrics.roadgraph`` (same)."""

    steps_elapsed: int
    max_episode_horizon: int


def check_termination(inputs: TerminationInputs) -> TerminationResult:
    """Single termination decision point. Evaluated every step;
    checks in this fixed priority order so a simultaneous
    success+failure frame (e.g. ego reaches the target lane exactly as
    it collides) is not misreported as a clean success."""

    if inputs.collision:
        return TerminationResult(TerminationReason.FAILURE_COLLISION, True)

    if inputs.offroad:
        return TerminationResult(TerminationReason.FAILURE_OFFROAD, True)

    if inputs.success:
        return TerminationResult(TerminationReason.SUCCESS, True)

    if inputs.steps_elapsed >= inputs.max_episode_horizon:
        return TerminationResult(TerminationReason.TRUNCATION_HORIZON, True)

    return TerminationResult(TerminationReason.NONE, False)


# Stage B-0 Section 7: derived from the 168-maneuver
# (transition_frame - merge_start_frame) span distribution --
# min=7, median=17, p90=33, p95=39, max=52 frames (see
# outputs/phase1/training_10shard_pilot/ manifest + maneuver tables).
# Set with a generous safety margin above the observed max (roughly
# 2x) so a controlled (PPO/FSM) rollout -- which may legitimately take
# longer than the logged human driver -- is not truncated
# prematurely, while still bounding a runaway episode. NOT the
# logged transition_frame itself (that remains reference-only, see
# module docstring).
MAX_EPISODE_HORIZON_FRAMES = 100
