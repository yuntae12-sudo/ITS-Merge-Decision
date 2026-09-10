"""Phase 2 MERGE transition reference construction (Stage B-1 Section
5): avoids a steering discontinuity at the instant MERGE commits by
blending from the source lane's centerline toward the target lane's
centerline over a fixed longitudinal distance, rather than switching
the controller's tracked polyline outright.

Deliberately NOT a new Frenet-frame planner (Stage B-1 explicit
instruction) -- this is a minimal, deterministic, reproducible linear
lateral blend built directly from the two already-reconstructed
``LanePolyline`` objects Phase 1 provides, sampled at a fixed step
along the SOURCE lane's own arc length (the coordinate ego is
leaving, so the blend is well-defined immediately at commitment even
before ego has any presence on the target lane).
"""

import dataclasses

import numpy as np

from src.scenarios.lane_geometry import (
    LanePolyline,
    compute_arc_length,
    project_point_to_polyline,
    project_point_to_polyline_signed,
)

# Longitudinal distance over which the reference blends from 100%
# source-lane centerline to 100% target-lane centerline. Fixed,
# deterministic, not tuned to any performance criterion (Stage B-1
# scope) -- chosen to be a plausible single-lane-change distance,
# well within the shortest observed merge_distance_m in the training
# pool (Phase 1 Stage B report: median 10.3 m, min 0.0 m already
# excluded as a near-zero-margin edge case).
BLEND_DISTANCE_M = 8.0

# Number of sample points along the blend window. Deterministic;
# reproducible given the same two polylines and ego position.
BLEND_SAMPLE_COUNT = 9


@dataclasses.dataclass(frozen=True)
class MergeReference:
    """A blended reference the low-level controller can track exactly
    like any other LanePolyline (Stage B-1 Section 5: `reference_lane`
    just needs SOME polyline; this IS one, constructed on demand)."""

    polyline: LanePolyline


def build_merge_reference(
    source_polyline: LanePolyline,
    target_polyline: LanePolyline,
    ego_x: float,
    ego_y: float,
) -> MergeReference:
    """Builds a blended source->target reference polyline anchored at
    ego's CURRENT position (never a future/logged position).

    Samples ``BLEND_SAMPLE_COUNT`` points at even source-lane arc
    -length steps from ego's current source-lane position out to
    ``BLEND_DISTANCE_M`` ahead. At each sample, the point used is a
    linear blend (by fraction of the way through the blend window)
    between the source lane's own centerline at that arc length and
    the target lane's centerline at the correspondingly nearest arc
    length -- i.e. fraction 0 at ego's current position (100% source)
    to fraction 1 at the end of the window (100% target).

    Uses the SIGNED projection (Stage B-0 fix) for ego's source-lane
    position, so this remains well-defined even if ego is still
    upstream of the source lane's own reconstructed start (matching
    the same early-pre-merge situation that motivated the Stage B-0
    fix in the first place).

    The blend window is capped by whichever of the source lane's
    remaining length (from ego's position) or the target lane's own
    total length is shorter (bug found in real-data smoke testing,
    Stage B-1: a short target-lane segment, e.g. ~12 m total, made
    later blend samples all project past the target polyline's own
    end, clamping to nearly the same terminal point and producing a
    degenerate/oscillating reference heading that saturated the
    steering command and caused a spurious collision). Without this
    cap, the fixed BLEND_DISTANCE_M could exceed either lane's actual
    reconstructed geometry.
    """

    ego_source_s = project_point_to_polyline_signed(
        source_polyline, ego_x, ego_y
    )["arc_length_m"]

    source_remaining_m = max(source_polyline.arc_length[-1] - ego_source_s, 0.0)
    target_total_m = target_polyline.arc_length[-1]
    effective_blend_distance_m = max(
        min(BLEND_DISTANCE_M, source_remaining_m, target_total_m),
        1.0,  # never degenerate to a zero-length window
    )

    sample_fractions = np.linspace(0.0, 1.0, BLEND_SAMPLE_COUNT)
    sample_source_s = ego_source_s + sample_fractions * effective_blend_distance_m

    points = np.empty((BLEND_SAMPLE_COUNT, 2), dtype=np.float64)
    for i, (fraction, source_s) in enumerate(
        zip(sample_fractions, sample_source_s)
    ):
        source_xy = _interpolate_polyline_xy(source_polyline, source_s)

        # Project that source-lane point onto the target lane to find
        # the corresponding target-lane point at (approximately) the
        # same longitudinal progress, then blend toward it.
        target_projection = project_point_to_polyline(
            target_polyline, source_xy[0], source_xy[1]
        )
        target_xy = _interpolate_polyline_xy(
            target_polyline, target_projection["arc_length_m"]
        )

        points[i] = (1.0 - fraction) * source_xy + fraction * target_xy

    return MergeReference(
        polyline=LanePolyline(
            lane_id=target_polyline.lane_id,
            lane_type=target_polyline.lane_type,
            xy=points,
            direction=_finite_difference_directions(points),
            arc_length=compute_arc_length(points),
        )
    )


def _interpolate_polyline_xy(polyline: LanePolyline, arc_length_m: float):
    """Interpolates (x, y) at a given arc length, clamped to the
    polyline's own domain (deliberately clamped here, unlike the
    Stage B-0 signed projection -- a reference point should never
    extrapolate past a lane's physically reconstructed geometry)."""

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


def _finite_difference_directions(points: np.ndarray) -> np.ndarray:
    """Per-point unit direction vectors for a blended reference's own
    LanePolyline (matching lane_geometry.LanePolyline's field
    contract), via simple forward/backward finite differences."""

    directions = np.zeros_like(points)
    diffs = points[1:] - points[:-1]
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1e-9, norms)
    unit_diffs = diffs / norms

    directions[:-1] += unit_diffs
    directions[1:] += unit_diffs
    row_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    row_norms = np.where(row_norms == 0.0, 1e-9, row_norms)
    return directions / row_norms
