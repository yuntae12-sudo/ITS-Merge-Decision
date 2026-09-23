"""Merge topology classification for stable lane transitions.

Commit C's ``find_lane_transitions`` only observes "ego moved from
lane A to lane B in a stable way" -- it makes no claim about *why*.
This module classifies each such ``LaneTransition`` against a series
of independent geometric/temporal gates and produces a three-way
decision: ACCEPT (confident merge candidate), REJECT (confidently not
a merge -- clearly parallel lane change, crossing lane, insufficient
convergence, etc.), or REVIEW (available geometry cannot confidently
distinguish a true merge from a serial map-segment continuation --
see "Serial-vs-merge ambiguity" below).

Important architectural conclusion -- no authoritative lane topology
----------------------------------------------------------------------
The current Waymax tf_example input pipeline (confirmed by inspecting
``waymax.dataloader.womd_utils.get_features_description()``'s full
feature list) exposes:

  - ``roadgraph_samples/*`` -- the flat point cloud reconstructed into
    ``LanePolyline`` by ``lane_geometry.py``: position, per-point
    direction, an id grouping, no connectivity.
  - ``path_samples/*`` -- SDC route path samples (``on_route`` etc.),
    which describe ego's own predicted route, NOT a lane connectivity
    graph between arbitrary lanes.
  - agent trajectories and traffic-light state -- unrelated to lane
    topology.

None of these expose lane predecessor/successor/neighbor topology.
That information exists only in WOMD's separate Scenario protobuf
format (``MapFeature.lane.entry_lanes`` / ``exit_lanes``), which
requires a different dataset download and the (currently uninstalled)
``waymo_open_dataset`` package -- explicitly out of scope for this
commit (see Known Limitations in the commit report). Every judgment
below is therefore either:

  (a) directly available geometry -- polyline endpoints, arc length,
      per-point heading, exact segment projection (all already
      provided by ``lane_geometry.py``), or
  (b) geometry-INFERRED evidence -- e.g. "these two lanes' endpoints
      are close and their separation shrinks approaching that point"
      as a proxy for merge topology, since no explicit
      "target_lane.entry_lanes" field exists to consult directly.

Geometry inference is never described as ground-truth topology in this
module's docstrings, logs, or diagnostic field names.

Serial-vs-merge ambiguity (why REVIEW exists)
-----------------------------------------------
An earlier version of this module tried to hard-classify "serial
map-segment continuation" (the same physical lane split across
multiple WOMD feature ids -- not a merge) from geometry alone, using
endpoint proximity, terminal collinearity with the target's tangent,
and the upstream source-target separation trend. Real-WOMD
investigation (30 scenes, 38 transitions) showed this does not work
reliably:

  - Upstream separation is NOT a discriminator: both confirmed serial
    transitions and the one true merge-like candidate found in this
    survey (record 28, lane 485->344) show the SAME curve shape --
    large separation far upstream (15-30 m), shrinking to near zero at
    the endpoint. Requiring "already small upstream" as a serial
    condition let true merges be misclassified as accepted merges
    just as often as it correctly flagged serial cases, once real data
    was checked (see Design Decisions in the commit report).
  - Terminal collinearity (perpendicular distance from source's
    upstream samples to the target's backward-extended local tangent)
    is unreliable whenever the target lane is long or curved: a local
    tangent line is only a valid straight-line approximation very
    close to the anchor point, so confirmed serial cases (e.g.
    196->206, endpoint distance exactly 0.0 m) can show a LARGE
    collinearity offset (up to ~15 m) purely because the target curves
    away from its own endpoint tangent, not because source and target
    are unrelated.

What DOES separate the one true merge from confirmed serial cases in
this survey is much simpler: ``endpoint_target_distance`` itself.
Confirmed serial transitions all land with the source endpoint
essentially exactly ON the target polyline (0.000 m lateral offset,
well under ``serial_continuation_max_lateral_m``); the one true merge
found landed 2.14 m off target -- comfortably inside
``max_endpoint_target_distance_m`` (this study's "close enough to be
plausibly converging") but well outside the much tighter
"essentially on top of it" band. This module uses exactly that
signal, and only that signal, to trigger REVIEW rather than an
automatic REJECT or ACCEPT: an endpoint landing within
``serial_continuation_max_lateral_m`` of the target is consistent with
BOTH a serial continuation and a merge whose target WOMD feature
happens to begin exactly at the convergence point -- available
geometry cannot tell those apart, so it is flagged for manual review
(Compressed Commit E) rather than confidently decided either way.
Collinearity and upstream separation are still computed and exposed on
``MergeDiagnostic``. Collinearity is now used only in a narrow, conservative
serial-boundary guard added after the MAN_0013 regression: target projection
at its start boundary, <=5 degree heading difference, <=1 m terminal
collinearity offset, and <=5 m endpoint gap. Outside that conjunction it
remains diagnostic context rather than a general merge/serial classifier.

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
import enum
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
    # Conservative legacy-v1 guard.  A source endpoint which projects
    # onto the very beginning of a nearly collinear target is normally a
    # sampled-map segment boundary, not a merge.  Authoritative v2 data uses
    # Scenario-protobuf topology instead (src.scenarios.merge_v2).
    serial_boundary_max_target_s_m: float = 0.5
    serial_boundary_max_collinear_offset_m: float = 1.0
    serial_boundary_max_heading_difference_deg: float = 5.0


class MergeDecision(enum.Enum):
    """Three-way outcome of classifying a LaneTransition.

    ACCEPT: confident merge candidate -- all gates passed and the
        endpoint did not land in the ambiguous serial-vs-merge band.
    REJECT: confidently NOT a merge -- a gate clearly failed (missing
        geometry, insufficient history, source not near its end,
        target too far, heading mismatch, clearly parallel, clearly
        insufficient convergence, or insufficient target persistence).
    REVIEW: available geometry cannot confidently distinguish a true
        merge from a serial map-segment continuation (see module
        docstring "Serial-vs-merge ambiguity"). Not a rejection --
        Compressed Commit E's manual validation step is expected to
        resolve these.
    """

    ACCEPT = "accept"
    REJECT = "reject"
    REVIEW = "review"


REJECT_MISSING_GEOMETRY = "missing_geometry"
REJECT_SOURCE_NOT_NEAR_END = "source_not_near_end"
REJECT_TARGET_TOO_FAR = "target_too_far"
REJECT_HEADING_MISMATCH = "heading_mismatch"
REJECT_PARALLEL_LANE_CHANGE = "parallel_lane_change"
REJECT_INSUFFICIENT_CONVERGENCE = "insufficient_convergence"
REJECT_INSUFFICIENT_PRE_MERGE = "insufficient_pre_merge_history"
REJECT_INSUFFICIENT_TARGET_PERSISTENCE = "insufficient_target_persistence"
REJECT_SERIAL_CONTINUATION = "serial_continuation"

REVIEW_AMBIGUOUS_SERIAL_OR_MERGE = "ambiguous_serial_or_merge"


@dataclasses.dataclass(frozen=True)
class MergeDiagnostic:
    """Structured result of classifying one LaneTransition.

    Every candidate -- ACCEPT, REJECT, or REVIEW -- keeps enough
    diagnostic information to answer "why". ``reason`` is one of the
    ``REJECT_*`` constants when ``decision == MergeDecision.REJECT``,
    ``REVIEW_AMBIGUOUS_SERIAL_OR_MERGE`` when
    ``decision == MergeDecision.REVIEW``, or ``None`` when
    ``decision == MergeDecision.ACCEPT``.

    ``is_merge_candidate`` is kept for backward compatibility with
    Commit D callers (e.g. scripts/inspect_merge_candidate.py's
    feature-extraction branch) and is True only for ACCEPT -- REVIEW is
    deliberately NOT treated as accepted merge for feature-extraction
    purposes, since it has not been confirmed.
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

    # Gate 5 / parallel-change vs. convergence: convergence evidence
    # (geometry-inferred from sampled separation trend; see module
    # docstring).
    lanes_converge: bool
    parallel_continuation: bool
    separation_reduction_m: Optional[float]
    decreasing_fraction: Optional[float]
    # Context used only by the narrow serial-boundary regression guard (see
    # module docstring; never a general topology substitute): max perpendicular distance from
    # the source's upstream samples to the target's own backward
    # -extended tangent line, and the farthest (most-upstream) sampled
    # source-target separation.
    max_collinear_offset_m: Optional[float]
    upstream_separation_m: Optional[float]

    # Gate 6: pre-merge source history and post-transition target
    # persistence (uses Commit C's stable-sequence frame ranges;
    # target persistence relies on frames AFTER the transition --
    # offline-only, see module docstring).
    pre_merge_frames: int
    target_lane_persistent: bool

    decision: MergeDecision
    reason: Optional[str]

    # Backward-compatible convenience field: True only for
    # decision == MergeDecision.ACCEPT (see class docstring).
    is_merge_candidate: bool


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


def _sample_source_upstream_arc_lengths(
    source_polyline: LanePolyline,
    config: MergeTopologyConfig,
) -> np.ndarray:
    """Arc-length positions to sample upstream of the source endpoint.

    Fix (project plan section 7): builds the window directly in-domain
    -- ``[max(source_end_s - convergence_window_m, source_start_s),
    source_end_s]`` -- and samples evenly across it, rather than
    sampling fixed offsets from the endpoint and clamping each one
    independently. The latter approach could construct
    negative/out-of-domain arc lengths internally and, whenever
    ``source_lane_length < convergence_window_m``, would repeatedly
    clamp several of the farthest samples to the same
    ``source_polyline.arc_length[0]`` point (redundant samples, and a
    window that silently extended past the source lane's own domain
    before clamping). Guarantees
    ``source_start_s <= every sample <= source_end_s``.
    """

    source_start_s = source_polyline.arc_length[0]
    source_end_s = source_polyline.arc_length[-1]

    window_start_s = max(
        source_end_s - config.convergence_window_m, source_start_s
    )

    return np.linspace(
        window_start_s, source_end_s, config.convergence_sample_count
    )


def _sample_source_upstream_points(
    source_polyline: LanePolyline,
    config: MergeTopologyConfig,
):
    """Samples (x, y) positions along the source polyline over a window
    upstream of its endpoint (farthest to nearest the endpoint).
    """

    sample_arc_lengths = _sample_source_upstream_arc_lengths(
        source_polyline, config
    )

    return np.asarray(
        [
            _interpolate_xy(source_polyline, s)
            for s in sample_arc_lengths
        ]
    )


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

    points = _sample_source_upstream_points(source_polyline, config)

    separations = []
    for xy in points:
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


def _measure_collinearity_with_target(
    source_polyline: LanePolyline,
    target_polyline: LanePolyline,
    endpoint_projection: dict,
    config: MergeTopologyConfig,
):
    """Measures how closely the source lane's terminal geometry follows
    a straight backward extension of the TARGET lane's own tangent at
    the convergence point.

    Conservative-boundary diagnostic (see module docstring
    "Serial-vs-merge ambiguity"): this is used only when the source endpoint
    also projects to the target's start with matching heading. Real-WOMD
    investigation showed it is unreliable whenever the target lane is
    long or curved -- a local tangent line is only a valid
    approximation very close to the anchor point, so confirmed serial
    continuation cases can show a large collinearity offset purely
    because the target curves away from its own endpoint tangent, not
    because source and target are geometrically unrelated. It is
    exposed on ``MergeDiagnostic`` purely as manual-review context.

    Geometry-inferred evidence (see module docstring): there is no
    explicit "same physical lane" flag. This measures the perpendicular
    distance from each upstream source sample to the straight line
    through the target's convergence point with the target's local
    tangent direction there.

    Returns:
        perpendicular_distances: array (farthest-to-nearest source
        endpoint, matching ``_measure_convergence``'s `separations`
        ordering) of perpendicular distance from each upstream source
        sample to the target's backward-extended tangent line.
    """

    heading = endpoint_projection["heading_rad"]
    tangent = np.array([np.cos(heading), np.sin(heading)])

    segment_index = endpoint_projection["segment_index"]
    anchor = target_polyline.xy[segment_index]

    points = _sample_source_upstream_points(source_polyline, config)

    relative = points - anchor
    along = relative @ tangent
    perpendicular = relative - along[:, None] * tangent
    perpendicular_distance = np.hypot(perpendicular[:, 0], perpendicular[:, 1])

    return perpendicular_distance


def _is_ambiguous_serial_or_merge(
    endpoint_target_distance: float,
    config: MergeTopologyConfig,
) -> bool:
    """Decides whether a transition falls in the geometry-ambiguous
    band between serial map-segment continuation and a true merge (see
    module docstring "Serial-vs-merge ambiguity" for the full
    investigation this is based on).

    This uses exactly ONE signal: whether the source endpoint lands
    within ``serial_continuation_max_lateral_m`` of the target polyline
    -- essentially exactly on top of it. That threshold is deliberately
    much tighter than ``max_endpoint_target_distance_m`` (this study's
    "plausibly converging" band): every confirmed serial-continuation
    transition observed in this study's 30-scene/38-transition survey
    landed with 0.000 m lateral offset there, while the one true
    merge-like candidate found (record 28, lane 485->344) landed 2.14 m
    off target -- comfortably inside ``max_endpoint_target_distance_m``
    but well outside this tighter band. Two other candidate signals
    (terminal collinearity with the target's tangent, and the upstream
    source-target separation trend) were investigated and found
    unreliable as additional conditions here -- see the module
    docstring for why -- so neither is used; they remain available on
    ``MergeDiagnostic`` purely as manual-review context.

    Landing inside this band does not by itself confirm serial
    continuation OR a merge: both are geometrically consistent with it,
    which is exactly why it triggers REVIEW instead of an automatic
    REJECT or ACCEPT.
    """

    return endpoint_target_distance <= config.serial_continuation_max_lateral_m


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
        it was reached, and ``decision`` / ``reason`` set based on the
        first hard-reject gate that failed, or REVIEW if the
        serial-vs-merge ambiguity band was hit, or ACCEPT if every
        gate passed cleanly (later gates are still computed where the
        necessary geometry is available, so rejected/review candidates
        keep useful diagnostics, but a gate that depends on an earlier
        failed gate's output may be left as ``None``/``False``
        defaults).
    """

    def _finalize(decision, reason, **evidence):
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
            separation_reduction_m=None,
            decreasing_fraction=None,
            max_collinear_offset_m=None,
            upstream_separation_m=None,
            pre_merge_frames=(
                transition.source_end_frame
                - transition.source_start_frame
                + 1
            ),
            target_lane_persistent=False,
            decision=decision,
            reason=reason,
            is_merge_candidate=(decision == MergeDecision.ACCEPT),
        )
        defaults.update(evidence)
        return MergeDiagnostic(**defaults)

    def _reject(reason, **evidence):
        return _finalize(MergeDecision.REJECT, reason, **evidence)

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

    # Gate 5 evidence: sampled source-target separation trend
    # (geometry-inferred, see module docstring and _measure_convergence)
    # and terminal collinearity with the target's own backward-extended
    # tangent (see _measure_collinearity_with_target). The latter participates
    # only in the narrow serial-boundary guard below; it is not treated as
    # authoritative topology.
    separations, reduction_m, decreasing_fraction = _measure_convergence(
        source_polyline, target_polyline, config
    )
    collinear_offsets = _measure_collinearity_with_target(
        source_polyline, target_polyline, endpoint_projection, config
    )
    max_collinear_offset = float(np.max(collinear_offsets))
    upstream_separation = float(separations[0])

    # Regression guard for MAN_0013 and the same finite-polyline failure
    # mode.  The old convergence metric projects every upstream source point
    # to a finite target polyline.  When the target begins just after a
    # collinear source ends, all upstream points clamp to target[0], making
    # ordinary longitudinal progress look like ~30 m of lateral convergence.
    # Treat this unmistakable boundary geometry as a serial continuation.
    # Less-clear geometry remains subject to the existing REVIEW path; v2
    # canonical data requires authoritative protobuf connectivity.
    serial_boundary = (
        endpoint_target_arc_length
        <= config.serial_boundary_max_target_s_m
        and endpoint_target_distance <= config.max_endpoint_target_distance_m
        and max_collinear_offset
        <= config.serial_boundary_max_collinear_offset_m
        and heading_difference_deg
        <= config.serial_boundary_max_heading_difference_deg
    )
    if serial_boundary:
        return _reject(
            REJECT_SERIAL_CONTINUATION,
            source_lane_ends=True,
            source_remaining_distance_m=source_remaining,
            endpoint_target_distance_m=endpoint_target_distance,
            endpoint_target_arc_length_m=endpoint_target_arc_length,
            heading_difference_deg=heading_difference_deg,
            lanes_converge=False,
            parallel_continuation=False,
            separation_reduction_m=reduction_m,
            decreasing_fraction=decreasing_fraction,
            max_collinear_offset_m=max_collinear_offset,
            upstream_separation_m=upstream_separation,
            pre_merge_frames=pre_merge_frames,
            target_lane_persistent=target_lane_persistent,
        )

    lanes_converge = (
        reduction_m >= config.min_separation_reduction_m
        and decreasing_fraction >= config.min_decreasing_fraction
    )
    # "Parallel" means separation barely changed at all (near-zero
    # reduction, e.g. an ordinary lane change where source and target
    # simply run alongside each other) -- a confident REJECT.
    # Anything else that fails to confirm convergence is
    # insufficient/ambiguous convergence -- also a confident REJECT
    # (this is a different question from the serial-vs-merge ambiguity
    # below: here the geometry does not even show a plausible
    # converging approach, so there is nothing to send for manual
    # review).
    near_zero_reduction = reduction_m < (
        0.5 * config.min_separation_reduction_m
    )
    parallel_continuation = not lanes_converge and near_zero_reduction

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
            max_collinear_offset_m=max_collinear_offset,
            upstream_separation_m=upstream_separation,
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
            max_collinear_offset_m=max_collinear_offset,
            upstream_separation_m=upstream_separation,
            pre_merge_frames=pre_merge_frames,
            target_lane_persistent=False,
        )

    # Serial-vs-merge ambiguity check (see module docstring): every
    # other gate has now passed, including a confirmed converging
    # approach and target persistence. The ONLY remaining question is
    # whether this converging transition is a true merge or a serial
    # map-segment continuation that happens to also show convergence
    # -like geometry. Available geometry cannot answer that reliably
    # (see module docstring), so this is REVIEW, not an automatic
    # ACCEPT or REJECT.
    common_evidence = dict(
        source_lane_ends=True,
        source_remaining_distance_m=source_remaining,
        endpoint_target_distance_m=endpoint_target_distance,
        endpoint_target_arc_length_m=endpoint_target_arc_length,
        heading_difference_deg=heading_difference_deg,
        lanes_converge=True,
        parallel_continuation=False,
        separation_reduction_m=reduction_m,
        decreasing_fraction=decreasing_fraction,
        max_collinear_offset_m=max_collinear_offset,
        upstream_separation_m=upstream_separation,
        pre_merge_frames=pre_merge_frames,
        target_lane_persistent=True,
    )

    if _is_ambiguous_serial_or_merge(endpoint_target_distance, config):
        return _finalize(
            MergeDecision.REVIEW,
            REVIEW_AMBIGUOUS_SERIAL_OR_MERGE,
            **common_evidence,
        )

    return _finalize(MergeDecision.ACCEPT, None, **common_evidence)


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

    merge_end_s = source_polyline.arc_length[-1]

    # Fix (project plan section 7): sample in-domain arc lengths via
    # the same helper used elsewhere, rather than constructing
    # `source_end_s - offset` directly -- guarantees every sample (and
    # therefore merge_start_s) satisfies
    # source_polyline.arc_length[0] <= sample <= merge_end_s, even when
    # the source lane is shorter than convergence_window_m.
    sample_s = _sample_source_upstream_arc_lengths(source_polyline, config)

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
