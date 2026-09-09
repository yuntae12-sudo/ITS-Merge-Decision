"""Merge topology classification for stable lane transitions.

Commit C's ``find_lane_transitions`` only observes "ego moved from
lane A to lane B in a stable way" -- it makes no claim about *why*.
This module classifies each such ``LaneTransition`` against a series
of independent geometric/temporal gates and decides whether it is a
plausible Merge candidate, a normal parallel lane change, a serial
map-segment continuation (the same physical lane split across
multiple WOMD feature ids), a crossing/unrelated lane, or ambiguous.

Important constraint -- no invented lane topology
--------------------------------------------------
Waymax's ``roadgraph_points`` (as reconstructed into ``LanePolyline``
by ``lane_geometry.py``) exposes only sampled point geometry: position,
per-point direction, and an id grouping. There is no
predecessor/successor/neighbor-lane field anywhere in this pipeline.
Every judgment below is therefore either:

  (a) directly available geometry -- polyline endpoints, arc length,
      per-point heading, exact segment projection (all already
      provided by ``lane_geometry.py``), or
  (b) geometry-INFERRED evidence -- e.g. "these two lanes' endpoints
      are close and their separation shrinks approaching that point"
      as a proxy for merge topology, since no explicit
      "target_lane.entry_lanes" field exists to consult directly.

Every gate below is commented as to which category it falls into. If
raw WOMD scenario-proto topology (which does encode lane
predecessors/successors, unlike the roadgraph_points sample cloud used
here) becomes necessary for higher precision, that is a known
limitation of this Commit, not something silently assumed to already
be present.

Offline vs. online (event-time) distinction
--------------------------------------------
This module is offline dataset-construction code: it may use ego's
logged trajectory *after* the transition frame (e.g. to confirm target
-lane persistence, or to define merge completion) because Phase 1 is
building a labeled dataset, not making a live driving decision. This
is explicitly NOT allowed once Phase 3's online policy observation is
built from this same geometry -- that code path must never depend on
future frames. See the Gate 6 docstring below for where this applies.
"""

import dataclasses
from typing import List, Optional

import numpy as np
import yaml

from src.scenarios.lane_assignment import LaneTransition
from src.scenarios.lane_geometry import LanePolyline, project_point_to_polyline


@dataclasses.dataclass(frozen=True)
class MergeTopologyConfig:
    """Thresholds for merge topology classification.

    See configs/phase1_merge.yaml `merge_topology` and `event` for the
    field descriptions and the real-scene observations behind them.
    """

    max_source_end_distance_m: float
    max_endpoint_target_distance_m: float
    max_heading_difference_deg: float
    convergence_window_m: float
    convergence_sample_count: int
    min_separation_reduction_m: float
    min_decreasing_fraction: float
    serial_continuation_max_lateral_m: float
    min_pre_merge_frames: int
    min_target_lane_frames: int


REJECT_MISSING_GEOMETRY = "missing_geometry"
REJECT_SOURCE_NOT_NEAR_END = "source_not_near_end"
REJECT_TARGET_TOO_FAR = "target_too_far"
REJECT_HEADING_MISMATCH = "heading_mismatch"
REJECT_SERIAL_CONTINUATION = "serial_lane_continuation"
REJECT_PARALLEL_LANE_CHANGE = "parallel_lane_change"
REJECT_INSUFFICIENT_CONVERGENCE = "insufficient_convergence"
REJECT_INSUFFICIENT_PRE_MERGE = "insufficient_pre_merge_history"
REJECT_INSUFFICIENT_TARGET_PERSISTENCE = "insufficient_target_persistence"


@dataclasses.dataclass(frozen=True)
class MergeDiagnostic:
    """Structured result of classifying one LaneTransition.

    Every candidate -- accepted or rejected -- keeps enough
    diagnostic information to answer "why". ``reject_reason`` is one
    of the ``REJECT_*`` constants in this module, or ``None`` if
    ``is_merge_candidate`` is True.
    """

    source_lane_id: int
    target_lane_id: int
    transition_frame: int

    # Gate 1 (stable transition) is implicit: this diagnostic only
    # exists for transitions that already passed Commit C's stable
    # transition detection.

    # Gate 2/7: source-lane-ending evidence (geometry-inferred: no
    # source lane termination flag exists in roadgraph_points, only
    # the ego-observed remaining arc length toward the polyline's own
    # travel-direction endpoint).
    source_lane_ends: bool
    source_remaining_distance_m: Optional[float]

    # Gate 3: endpoint-to-target proximity (directly available exact
    # segment projection).
    endpoint_target_distance_m: Optional[float]
    endpoint_target_arc_length_m: Optional[float]

    # Gate 4: heading compatibility at the convergence point (directly
    # available geometry).
    heading_difference_deg: Optional[float]

    # Gate 5 / serial-continuation vs. parallel-change: convergence
    # evidence (geometry-inferred from sampled separation trend; see
    # module docstring).
    lanes_converge: bool
    parallel_continuation: bool
    serial_continuation: bool
    separation_reduction_m: Optional[float]
    decreasing_fraction: Optional[float]

    # Gate 6: pre-merge source history and post-transition target
    # persistence (uses Commit C's stable-sequence frame ranges;
    # target persistence relies on frames AFTER the transition --
    # offline-only, see module docstring).
    pre_merge_frames: int
    target_lane_persistent: bool

    is_merge_candidate: bool
    reject_reason: Optional[str]


def load_merge_topology_config(config_path: str) -> MergeTopologyConfig:
    """Loads the `merge_topology` + `event` sections of a Phase 1 merge
    config YAML into one ``MergeTopologyConfig``.
    """

    with open(config_path, "r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file) or {}

    merge_topology_raw = dict(raw.get("merge_topology", {}))
    event_raw = raw.get("event", {})

    merge_topology_raw["min_pre_merge_frames"] = event_raw.get(
        "min_pre_merge_frames"
    )
    merge_topology_raw["min_target_lane_frames"] = event_raw.get(
        "min_target_lane_frames"
    )

    return MergeTopologyConfig(**merge_topology_raw)


def _wrap_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def _interpolate_xy(polyline: LanePolyline, arc_length_m: float):
    """Interpolates an (x, y) position at a given arc length along a
    polyline (clamped to the polyline's own arc-length domain).
    """

    arc_length_m = float(
        np.clip(arc_length_m, polyline.arc_length[0], polyline.arc_length[-1])
    )

    segment_index = int(
        np.clip(
            np.searchsorted(polyline.arc_length, arc_length_m),
            1,
            len(polyline.arc_length) - 1,
        )
    )

    s0 = polyline.arc_length[segment_index - 1]
    s1 = polyline.arc_length[segment_index]
    t = (arc_length_m - s0) / (s1 - s0) if s1 > s0 else 0.0

    xy0 = polyline.xy[segment_index - 1]
    xy1 = polyline.xy[segment_index]

    return xy0 + t * (xy1 - xy0)


def _measure_convergence(
    source_polyline: LanePolyline,
    target_polyline: LanePolyline,
    config: MergeTopologyConfig,
):
    """Samples source-target separation over a window before the
    source lane's endpoint and summarizes the trend.

    Geometry-inferred evidence (see module docstring): there is no
    explicit "these lanes converge" field, so convergence is inferred
    from whether the exact segment-projected separation between a
    series of source-lane points and the target polyline decreases
    approaching the source lane's endpoint.

    Returns:
        (separations, reduction_m, decreasing_fraction) where
        `separations` is the sampled separation list from farthest to
        nearest the source endpoint, `reduction_m` is
        separations[0] - separations[-1], and `decreasing_fraction` is
        the fraction of consecutive sample-to-sample steps that
        decreased.
    """

    source_end_s = source_polyline.arc_length[-1]
    sample_offsets = np.linspace(
        config.convergence_window_m, 0.0, config.convergence_sample_count
    )

    separations = []

    for offset in sample_offsets:
        sample_s = max(source_end_s - offset, source_polyline.arc_length[0])
        xy = _interpolate_xy(source_polyline, sample_s)
        projection = project_point_to_polyline(
            target_polyline, float(xy[0]), float(xy[1])
        )
        separations.append(abs(projection["lateral_distance_m"]))

    separations = np.asarray(separations)
    reduction_m = float(separations[0] - separations[-1])

    steps = np.diff(separations)
    decreasing_fraction = (
        float(np.mean(steps < 0.0)) if len(steps) > 0 else 0.0
    )

    return separations, reduction_m, decreasing_fraction


def detect_merge(
    transition: LaneTransition,
    source_polyline: Optional[LanePolyline],
    target_polyline: Optional[LanePolyline],
    ego_source_arc_length_m: Optional[float],
    config: MergeTopologyConfig,
) -> MergeDiagnostic:
    """Classifies one LaneTransition against the merge topology gates.

    Args:
        transition: a stable A->B transition from
            ``find_lane_transitions``.
        source_polyline: the source lane's geometry, or None if it
            could not be reconstructed for this scene (e.g. filtered
            out by lane type).
        target_polyline: the target lane's geometry, or None likewise.
        ego_source_arc_length_m: ego's source-lane arc-length position
            (from ``project_point_to_polyline`` on the source lane) at
            the transition frame, or None if unavailable.
        config: thresholds (see configs/phase1_merge.yaml).

    Returns:
        MergeDiagnostic with every gate's evidence populated as far as
        it was reached, and ``is_merge_candidate`` / ``reject_reason``
        set based on the first gate that failed (later gates are still
        computed where the necessary geometry is available, so
        rejected candidates keep useful diagnostics, but a gate that
        depends on an earlier failed gate's output may be left as
        ``None``/``False`` defaults).
    """

    def _reject(reason, **evidence):
        defaults = dict(
            source_lane_id=transition.source_lane_id,
            target_lane_id=transition.target_lane_id,
            transition_frame=transition.transition_frame,
            source_lane_ends=False,
            source_remaining_distance_m=None,
            endpoint_target_distance_m=None,
            endpoint_target_arc_length_m=None,
            heading_difference_deg=None,
            lanes_converge=False,
            parallel_continuation=False,
            serial_continuation=False,
            separation_reduction_m=None,
            decreasing_fraction=None,
            pre_merge_frames=(
                transition.source_end_frame
                - transition.source_start_frame
                + 1
            ),
            target_lane_persistent=False,
            is_merge_candidate=False,
            reject_reason=reason,
        )
        defaults.update(evidence)
        return MergeDiagnostic(**defaults)

    pre_merge_frames = (
        transition.source_end_frame - transition.source_start_frame + 1
    )
    target_frames = (
        transition.target_end_frame - transition.target_start_frame + 1
    )
    target_lane_persistent = target_frames >= config.min_target_lane_frames

    if source_polyline is None or target_polyline is None:
        return _reject(REJECT_MISSING_GEOMETRY)

    if pre_merge_frames < config.min_pre_merge_frames:
        return _reject(
            REJECT_INSUFFICIENT_PRE_MERGE,
            pre_merge_frames=pre_merge_frames,
            target_lane_persistent=target_lane_persistent,
        )

    # Gate 2: source-lane-ending evidence. There is no explicit "this
    # lane terminates" flag (geometry-inferred): we only know how far
    # ego's own transition-frame position was from the source
    # polyline's travel-direction endpoint.
    if ego_source_arc_length_m is None:
        source_remaining = None
        source_lane_ends = False
    else:
        source_remaining = float(
            source_polyline.arc_length[-1] - ego_source_arc_length_m
        )
        source_lane_ends = (
            0.0 <= source_remaining <= config.max_source_end_distance_m
        )

    if not source_lane_ends:
        return _reject(
            REJECT_SOURCE_NOT_NEAR_END,
            source_remaining_distance_m=source_remaining,
            pre_merge_frames=pre_merge_frames,
            target_lane_persistent=target_lane_persistent,
        )

    # Gate 3: endpoint-to-target proximity (exact segment projection,
    # directly available geometry).
    source_end_xy = source_polyline.xy[-1]
    endpoint_projection = project_point_to_polyline(
        target_polyline, float(source_end_xy[0]), float(source_end_xy[1])
    )
    endpoint_target_distance = abs(endpoint_projection["lateral_distance_m"])
    endpoint_target_arc_length = endpoint_projection["arc_length_m"]

    if endpoint_target_distance > config.max_endpoint_target_distance_m:
        return _reject(
            REJECT_TARGET_TOO_FAR,
            source_lane_ends=True,
            source_remaining_distance_m=source_remaining,
            endpoint_target_distance_m=endpoint_target_distance,
            endpoint_target_arc_length_m=endpoint_target_arc_length,
            pre_merge_frames=pre_merge_frames,
            target_lane_persistent=target_lane_persistent,
        )

    # Gate 4: heading compatibility at the convergence point (directly
    # available: per-point/segment heading already computed by
    # project_point_to_polyline).
    source_end_heading_deg = np.degrees(
        project_point_to_polyline(
            source_polyline, float(source_end_xy[0]), float(source_end_xy[1])
        )["heading_rad"]
    )
    target_heading_at_endpoint_deg = np.degrees(
        endpoint_projection["heading_rad"]
    )
    heading_difference_deg = abs(
        _wrap_angle_deg(
            target_heading_at_endpoint_deg - source_end_heading_deg
        )
    )

    if heading_difference_deg > config.max_heading_difference_deg:
        return _reject(
            REJECT_HEADING_MISMATCH,
            source_lane_ends=True,
            source_remaining_distance_m=source_remaining,
            endpoint_target_distance_m=endpoint_target_distance,
            endpoint_target_arc_length_m=endpoint_target_arc_length,
            heading_difference_deg=heading_difference_deg,
            pre_merge_frames=pre_merge_frames,
            target_lane_persistent=target_lane_persistent,
        )

    # Serial map-segment continuation check (geometry-inferred
    # heuristic, see module docstring / Case 3 in
    # tests/scenarios/test_merge_detector.py): the same physical lane
    # is sometimes split into consecutive WOMD feature ids with A's
    # travel-direction endpoint landing essentially exactly on B's own
    # start or end with near-zero lateral offset -- i.e. this is not a
    # lateral convergence at all, source and target are collinear
    # continuations of one another.
    serial_continuation = (
        endpoint_target_distance <= config.serial_continuation_max_lateral_m
        and (
            endpoint_target_arc_length
            <= config.serial_continuation_max_lateral_m
            or endpoint_target_arc_length
            >= target_polyline.arc_length[-1]
            - config.serial_continuation_max_lateral_m
        )
    )

    # Gate 5: convergence vs. parallel continuation (geometry-inferred
    # from the sampled separation trend -- see module docstring and
    # _measure_convergence).
    separations, reduction_m, decreasing_fraction = _measure_convergence(
        source_polyline, target_polyline, config
    )

    lanes_converge = (
        reduction_m >= config.min_separation_reduction_m
        and decreasing_fraction >= config.min_decreasing_fraction
    )
    # Distinguish two failure shapes when convergence isn't confirmed:
    # "parallel" means separation barely changed at all (near-zero
    # reduction, e.g. an ordinary lane change where source and target
    # simply run alongside each other); "insufficient/ambiguous" means
    # some reduction was observed but it didn't clear both criteria
    # (e.g. a decreasing trend that isn't consistent enough, or a
    # borderline reduction amount) -- geometrically different from a
    # lane that never converges at all, even though neither is
    # accepted as a merge.
    near_zero_reduction = reduction_m < (
        0.5 * config.min_separation_reduction_m
    )
    parallel_continuation = (
        not lanes_converge
        and not serial_continuation
        and near_zero_reduction
    )

    if serial_continuation:
        return _reject(
            REJECT_SERIAL_CONTINUATION,
            source_lane_ends=True,
            source_remaining_distance_m=source_remaining,
            endpoint_target_distance_m=endpoint_target_distance,
            endpoint_target_arc_length_m=endpoint_target_arc_length,
            heading_difference_deg=heading_difference_deg,
            lanes_converge=lanes_converge,
            serial_continuation=True,
            separation_reduction_m=reduction_m,
            decreasing_fraction=decreasing_fraction,
            pre_merge_frames=pre_merge_frames,
            target_lane_persistent=target_lane_persistent,
        )

    if not lanes_converge:
        reason = (
            REJECT_PARALLEL_LANE_CHANGE
            if parallel_continuation
            else REJECT_INSUFFICIENT_CONVERGENCE
        )
        return _reject(
            reason,
            source_lane_ends=True,
            source_remaining_distance_m=source_remaining,
            endpoint_target_distance_m=endpoint_target_distance,
            endpoint_target_arc_length_m=endpoint_target_arc_length,
            heading_difference_deg=heading_difference_deg,
            lanes_converge=False,
            parallel_continuation=parallel_continuation,
            separation_reduction_m=reduction_m,
            decreasing_fraction=decreasing_fraction,
            pre_merge_frames=pre_merge_frames,
            target_lane_persistent=target_lane_persistent,
        )

    # Gate 6: target-lane persistence. This uses the transition's own
    # target_start/target_end_frame span, which Commit C's stable
    # -sequence computation already derived using FUTURE (post
    # -transition) frames -- offline-only, see module docstring. This
    # is not re-derived here; it is read from the already-computed
    # LaneTransition.
    if not target_lane_persistent:
        return _reject(
            REJECT_INSUFFICIENT_TARGET_PERSISTENCE,
            source_lane_ends=True,
            source_remaining_distance_m=source_remaining,
            endpoint_target_distance_m=endpoint_target_distance,
            endpoint_target_arc_length_m=endpoint_target_arc_length,
            heading_difference_deg=heading_difference_deg,
            lanes_converge=True,
            separation_reduction_m=reduction_m,
            decreasing_fraction=decreasing_fraction,
            pre_merge_frames=pre_merge_frames,
            target_lane_persistent=False,
        )

    return MergeDiagnostic(
        source_lane_id=transition.source_lane_id,
        target_lane_id=transition.target_lane_id,
        transition_frame=transition.transition_frame,
        source_lane_ends=True,
        source_remaining_distance_m=source_remaining,
        endpoint_target_distance_m=endpoint_target_distance,
        endpoint_target_arc_length_m=endpoint_target_arc_length,
        heading_difference_deg=heading_difference_deg,
        lanes_converge=True,
        parallel_continuation=False,
        serial_continuation=False,
        separation_reduction_m=reduction_m,
        decreasing_fraction=decreasing_fraction,
        pre_merge_frames=pre_merge_frames,
        target_lane_persistent=True,
        is_merge_candidate=True,
        reject_reason=None,
    )


def compute_merge_start_end_s(
    source_polyline: LanePolyline,
    target_polyline: LanePolyline,
    config: MergeTopologyConfig,
):
    """Defines merge_start_s / merge_end_s in source-lane arc length.

    merge_end_s is the source lane's travel-direction endpoint (its
    last arc-length value) -- directly available geometry.

    merge_start_s is the earliest sampled position (within
    convergence_window_m upstream of the endpoint) after which
    source-target separation is judged to be consistently decreasing:
    the first sample offset (walking from convergence_window_m down to
    0) at which the remaining samples toward the endpoint still meet
    min_decreasing_fraction. This is a geometry-inferred estimate, not
    a ground-truth "merge lane starts here" field (no such field
    exists in roadgraph_points).

    Returns:
        (merge_start_s, merge_end_s) in source-lane arc length. If no
        sampled prefix satisfies the decreasing-fraction criterion,
        merge_start_s falls back to the window's farthest sample
        (i.e. the whole configured window is used).
    """

    source_end_s = source_polyline.arc_length[-1]
    merge_end_s = source_end_s

    sample_offsets = np.linspace(
        config.convergence_window_m, 0.0, config.convergence_sample_count
    )
    sample_s = source_end_s - sample_offsets

    separations = []
    for s in sample_s:
        xy = _interpolate_xy(source_polyline, s)
        projection = project_point_to_polyline(
            target_polyline, float(xy[0]), float(xy[1])
        )
        separations.append(abs(projection["lateral_distance_m"]))
    separations = np.asarray(separations)

    for start_index in range(len(separations) - 1):
        tail = separations[start_index:]
        steps = np.diff(tail)
        if len(steps) == 0:
            continue
        decreasing_fraction = float(np.mean(steps < 0.0))
        if decreasing_fraction >= config.min_decreasing_fraction:
            return float(sample_s[start_index]), float(merge_end_s)

    return float(sample_s[0]), float(merge_end_s)


def compute_remaining_merge_distance(
    merge_end_s: float, ego_source_arc_length_m: Optional[float]
) -> float:
    """Remaining merge distance d_m, in source-lane arc length.

    d_m = merge_end_s - ego_source_s, clipped to be nonnegative: once
    ego is at or past the source lane's endpoint (or its source-lane
    projection is unavailable and treated as at the endpoint), d_m is
    0.0 rather than negative -- a physically interpretable "distance
    remaining until merge completion" value, per project plan section
    12. This function does not decide whether the merge is actually a
    valid candidate; callers apply it only to accepted candidates.
    """

    if ego_source_arc_length_m is None:
        return 0.0

    return float(max(merge_end_s - ego_source_arc_length_m, 0.0))
