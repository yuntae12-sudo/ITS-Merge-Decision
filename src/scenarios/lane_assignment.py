"""Ego lane assignment and lane transition detection over a trajectory.

This module maps ego's logged trajectory to a sequence of lane ids
using the static geometry in ``lane_geometry.py``, then temporally
filters that raw sequence into a stable sequence and extracts A -> B
lane transition candidates.

Scope boundary (see project plan, Commit C vs. Commit D): this module
only decides "ego moved from lane A to lane B in a stable way". It
does NOT decide whether that transition is a merge, a lane-ending, or
a normal parallel-lane change -- that topology judgment belongs to
Commit D's merge detector and is out of scope here.
"""

import dataclasses
from typing import List, Optional

import numpy as np
import yaml

from src.scenarios.lane_geometry import (
    LanePolyline,
    nearest_lane_candidates,
    project_point_to_polyline,
)


@dataclasses.dataclass(frozen=True)
class LaneAssignmentConfig:
    """Thresholds and weights for per-frame lane assignment.

    See configs/phase1_merge.yaml `lane_assignment` for the source of
    these values and the real-scene observations behind them.
    """

    max_lateral_distance_m: float
    max_heading_difference_deg: float
    persistence_frames: int
    max_ambiguous_gap_frames: Optional[int] = None
    candidate_count: int = 8
    lateral_weight: float = 1.0
    heading_weight: float = 1.0
    # Heading-difference values are normalized against this scale (deg)
    # before being weighted, so lateral (meters) and heading (degrees)
    # contribute on a comparable footing in the combined score.
    heading_normalization_deg: float = 45.0


@dataclasses.dataclass(frozen=True)
class LaneAssignment:
    """Result of assigning ego to a lane at a single frame."""

    frame_index: int
    lane_id: Optional[int]

    lateral_distance_m: Optional[float]
    heading_difference_rad: Optional[float]
    arc_length_m: Optional[float]

    score: Optional[float]
    valid: bool


@dataclasses.dataclass(frozen=True)
class LaneTransition:
    """A single stable source-lane -> target-lane transition candidate.

    This is a purely geometric/temporal observation: "ego was reliably
    on source_lane_id, then reliably on target_lane_id". Whether this
    constitutes a merge is decided in Commit D, not here.
    """

    source_lane_id: int
    target_lane_id: int

    transition_frame: int

    source_start_frame: int
    source_end_frame: int

    target_start_frame: int
    target_end_frame: int


def _wrap_angle(angle_rad: float) -> float:
    """Wraps an angle to [-pi, pi]."""

    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


def load_lane_assignment_config(config_path: str) -> LaneAssignmentConfig:
    """Loads the `lane_assignment` section of a Phase 1 merge config YAML.

    See configs/phase1_merge.yaml for the field descriptions and the
    real-scene observations behind the current values.
    """

    with open(config_path, "r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file) or {}

    lane_assignment_raw = raw.get("lane_assignment", {})

    return LaneAssignmentConfig(**lane_assignment_raw)


def assign_lane(
    x: float,
    y: float,
    yaw: float,
    polylines: List[LanePolyline],
    config: LaneAssignmentConfig,
    frame_index: int = -1,
) -> LaneAssignment:
    """Assigns a single (x, y, yaw) pose to the best-matching lane.

    Candidate generation uses ``nearest_lane_candidates`` (cheap sampled
    -vertex distance) only to shortlist ``config.candidate_count`` lanes
    -- it is a preselection step, not the final decision. Final scoring
    for each shortlisted candidate uses the exact segment-projected
    geometry from ``project_point_to_polyline``: absolute lateral
    distance and the wrapped heading difference between ego yaw and the
    lane's local heading at the projection.

    A candidate is rejected outright (not scored) if it exceeds either
    ``config.max_lateral_distance_m`` or ``config.max_heading_difference_deg``
    -- distance alone or heading alone is never sufficient, because at
    an intersection the physically nearest lane can be a crossing or
    opposing lane with a large heading mismatch (observed directly in
    real WOMD scenes during Commit C's threshold selection).

    Among surviving candidates, the one with the lowest combined score
    (normalized lateral distance + normalized heading difference) wins.
    If no candidate survives rejection, the frame is assigned
    ``lane_id=None`` rather than forcing a bad match.

    Args:
        x: ego x position (m).
        y: ego y position (m).
        yaw: ego heading (rad).
        polylines: candidate lanes, e.g. from ``extract_lane_polylines``.
        config: thresholds and score weights.
        frame_index: caller-supplied frame index, carried through into
            the result for traceability (not used in the computation).

    Returns:
        LaneAssignment. ``valid=False`` only when the input pose itself
        is not usable (handled by callers before invoking this
        function); an unmatched pose with no acceptable lane still has
        ``valid=True`` but ``lane_id=None``.
    """

    if not polylines:
        return LaneAssignment(
            frame_index=frame_index,
            lane_id=None,
            lateral_distance_m=None,
            heading_difference_rad=None,
            arc_length_m=None,
            score=None,
            valid=True,
        )

    candidates = nearest_lane_candidates(
        polylines, x, y, k=config.candidate_count
    )

    lane_by_id = {polyline.lane_id: polyline for polyline in polylines}

    best_lane_id = None
    best_lateral = None
    best_heading_diff = None
    best_arc_length = None
    best_score = None

    for lane_id, _ in candidates:

        polyline = lane_by_id[lane_id]
        projection = project_point_to_polyline(polyline, x, y)

        lateral_distance = abs(projection["lateral_distance_m"])
        heading_diff = _wrap_angle(projection["heading_rad"] - yaw)
        heading_diff_deg = abs(np.degrees(heading_diff))

        if lateral_distance > config.max_lateral_distance_m:
            continue

        if heading_diff_deg > config.max_heading_difference_deg:
            continue

        normalized_lateral = (
            lateral_distance / config.max_lateral_distance_m
        )
        normalized_heading = (
            heading_diff_deg / config.heading_normalization_deg
        )

        score = (
            config.lateral_weight * normalized_lateral
            + config.heading_weight * normalized_heading
        )

        if best_score is None or score < best_score:
            best_lane_id = lane_id
            best_lateral = projection["lateral_distance_m"]
            best_heading_diff = heading_diff
            best_arc_length = projection["arc_length_m"]
            best_score = score

    return LaneAssignment(
        frame_index=frame_index,
        lane_id=best_lane_id,
        lateral_distance_m=best_lateral,
        heading_difference_rad=best_heading_diff,
        arc_length_m=best_arc_length,
        score=best_score,
        valid=True,
    )


def assign_ego_lane_sequence(
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
    valid: np.ndarray,
    polylines: List[LanePolyline],
    config: LaneAssignmentConfig,
) -> List[LaneAssignment]:
    """Assigns a lane at every frame of an ego trajectory.

    Invalid frames (``valid[t] is False``) are not passed through
    ``assign_lane`` -- they are recorded as
    ``LaneAssignment(valid=False, lane_id=None, ...)`` so the returned
    list stays index-aligned with the input arrays.

    Args:
        x, y, yaw: (T,) ego pose arrays (e.g. from
            ``SimulatorState.log_trajectory``).
        valid: (T,) boolean validity mask, same length as x/y/yaw.
        polylines: candidate lanes for the scene.
        config: thresholds and score weights.

    Returns:
        List of length T of raw (pre-temporal-filtering) LaneAssignment.
    """

    num_frames = x.shape[0]
    assignments = []

    for frame_index in range(num_frames):

        if not valid[frame_index]:
            assignments.append(
                LaneAssignment(
                    frame_index=frame_index,
                    lane_id=None,
                    lateral_distance_m=None,
                    heading_difference_rad=None,
                    arc_length_m=None,
                    score=None,
                    valid=False,
                )
            )
            continue

        assignments.append(
            assign_lane(
                float(x[frame_index]),
                float(y[frame_index]),
                float(yaw[frame_index]),
                polylines,
                config,
                frame_index=frame_index,
            )
        )

    return assignments


def compute_stable_lane_sequence(
    assignments: List[LaneAssignment],
    persistence_frames: int,
    max_ambiguous_gap_frames: Optional[int] = None,
) -> List[Optional[int]]:
    """Temporally filters a raw per-frame lane sequence into a stable one.

    Raw per-frame assignment can jitter near lane boundaries or across
    ambiguous/intersection frames (e.g. ``A A A B A A``). This applies a
    straightforward hysteresis rule: a candidate new lane only replaces
    an *already-established* stable lane once it has been the raw
    assignment for at least ``persistence_frames`` consecutive (valid)
    frames; until that threshold is met, frames keep the previously
    stable lane id. Hysteresis only applies once a stable lane exists,
    though: at a cold start (no stable lane yet, e.g. the first valid
    frame(s) of a trajectory, or right after a stable lane was dropped
    by an over-long ambiguous gap) there is no existing assignment for
    the rule to protect, so the first real candidate is accepted
    immediately rather than waiting out persistence_frames again.

    Frames with ``lane_id=None`` (invalid pose, or no lane passed
    rejection) do not themselves force a stable-lane change: they are
    treated as "no new candidate observed this frame", so a short
    ambiguous gap does not flip the stable sequence -- the current
    stable lane carries forward across it. However, this carry-forward
    is only justified for a *short* gap: without a limit, an arbitrarily
    long run of ambiguous frames would silently bridge two lanes that
    may have nothing to do with each other. If
    ``max_ambiguous_gap_frames`` is set and a run of consecutive
    ``None`` raw assignments exceeds it, the stable lane is dropped to
    ``None`` for the remainder of that run (and must be re-established
    by a fresh persistent run once real candidates resume) rather than
    bridged. If left as the default ``None``, gaps are bridged
    indefinitely (equivalent to the previous, unbounded behavior).

    Args:
        assignments: raw per-frame assignments, e.g. from
            ``assign_ego_lane_sequence``.
        persistence_frames: number of consecutive raw-assignment frames
            required before accepting a stable lane change.
        max_ambiguous_gap_frames: maximum consecutive ``None`` raw
            frames the current stable lane is allowed to bridge over.

    Returns:
        List of the same length as `assignments`: the stable lane id
        (or None) at each frame.
    """

    num_frames = len(assignments)
    stable_sequence: List[Optional[int]] = [None] * num_frames

    current_stable: Optional[int] = None
    pending_lane: Optional[int] = None
    pending_run_length = 0
    ambiguous_gap_length = 0

    for frame_index, assignment in enumerate(assignments):

        raw_lane_id = assignment.lane_id

        if raw_lane_id is None:
            # No candidate observed this frame: does not break an
            # in-progress persistence run, but does not advance it
            # either. The current stable lane carries forward, unless
            # the ambiguous run has gone on too long to justify that.
            ambiguous_gap_length += 1

            if (
                max_ambiguous_gap_frames is not None
                and ambiguous_gap_length > max_ambiguous_gap_frames
            ):
                current_stable = None

            stable_sequence[frame_index] = current_stable
            continue

        ambiguous_gap_length = 0

        if raw_lane_id == current_stable:
            pending_lane = None
            pending_run_length = 0
            stable_sequence[frame_index] = current_stable
            continue

        if current_stable is None:
            # Cold start (no stable lane has ever been established, or
            # one was dropped by an over-long ambiguous gap): there is
            # no existing stable lane for hysteresis to protect, so
            # accept the very first real candidate immediately rather
            # than waiting out persistence_frames again.
            current_stable = raw_lane_id
            pending_lane = None
            pending_run_length = 0
            stable_sequence[frame_index] = current_stable
            continue

        if raw_lane_id == pending_lane:
            pending_run_length += 1
        else:
            pending_lane = raw_lane_id
            pending_run_length = 1

        if pending_run_length >= persistence_frames:
            current_stable = pending_lane
            pending_lane = None
            pending_run_length = 0

        stable_sequence[frame_index] = current_stable

    return stable_sequence


def find_lane_transitions(
    stable_sequence: List[Optional[int]],
    max_bridge_gap_frames: Optional[int] = None,
) -> List[LaneTransition]:
    """Extracts source -> target transitions from a stable lane sequence.

    A transition is recorded each time the stable lane id changes from
    one non-None value to a different non-None value, without an
    intervening ``None`` gap wider than ``max_bridge_gap_frames``.
    ``None`` runs (frames with no stable lane) do not themselves
    produce a transition and do not extend either lane's frame range,
    but a gap that is too wide is treated as insufficient evidence to
    link the two lanes at all -- "A -> None -> B" is not automatically
    confirmed as an A -> B transition just because B happens to follow
    eventually. With the default ``None``, any gap length is bridged
    (no rejection), matching an unbounded connection.

    This function does not evaluate whether a transition is a merge --
    see the module docstring.

    Args:
        stable_sequence: e.g. from ``compute_stable_lane_sequence``.
        max_bridge_gap_frames: maximum number of intervening ``None``
            frames allowed between two lane runs for them to still
            count as a source->target transition. Typically passed the
            same value as ``compute_stable_lane_sequence``'s
            ``max_ambiguous_gap_frames``, since a stable sequence
            should not itself contain a bridged gap wider than what
            was already allowed when it was computed; this parameter
            guards against a stable_sequence built without that limit
            (or with a different one) still producing an
            unsubstantiated long-gap transition here.

    Returns:
        List of ``LaneTransition``, one per source->target change,
        in frame order.
    """

    transitions = []

    # Collect contiguous runs of a single non-None lane id, e.g.
    # [(lane_id, start_frame, end_frame), ...], skipping None runs.
    # The number of frames separating two runs (all necessarily None,
    # since consecutive runs always differ in lane id) is then simply
    # next_run.start_frame - previous_run.end_frame - 1.
    runs = []
    run_lane_id = None
    run_start = None

    for frame_index, lane_id in enumerate(stable_sequence):

        if lane_id != run_lane_id:

            if run_lane_id is not None:
                runs.append((run_lane_id, run_start, frame_index - 1))

            run_lane_id = lane_id
            run_start = frame_index

    if run_lane_id is not None:
        runs.append(
            (run_lane_id, run_start, len(stable_sequence) - 1)
        )

    for previous_run, next_run in zip(runs, runs[1:]):

        source_lane_id, source_start, source_end = previous_run
        target_lane_id, target_start, target_end = next_run

        gap_frames = target_start - source_end - 1

        if (
            max_bridge_gap_frames is not None
            and gap_frames > max_bridge_gap_frames
        ):
            continue

        transitions.append(
            LaneTransition(
                source_lane_id=source_lane_id,
                target_lane_id=target_lane_id,
                transition_frame=target_start,
                source_start_frame=source_start,
                source_end_frame=source_end,
                target_start_frame=target_start,
                target_end_frame=target_end,
            )
        )

    return transitions
