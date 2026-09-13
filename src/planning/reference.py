"""Phase 3 Stage 3-A: Frenet reference-line geometry layer.

Builds a ``ReferenceLine`` -- x, y, arc length (s), yaw, curvature, and
curvature-derivative per point, plus interpolation/projection queries
-- on top of the ``LanePolyline`` objects
``src.scenarios.lane_geometry.extract_lane_polylines`` already
reconstructs from Waymax's WOMD roadgraph point cloud.

This module is deliberately a thin layer ON TOP of ``LanePolyline``:
it does not re-derive point ordering, travel-direction alignment, or
arc length (``lane_geometry.py`` already owns all of that -- see its
own module docstring for why). It only adds what a Frenet-frame
planner needs beyond a bare polyline: yaw, curvature, curvature
derivative, and interpolation/projection queries expressed against
those.

Design decisions, backed by real-WOMD measurement (see
``scripts/audit_phase3_reference_geometry.py`` and
``docs/phase3/OVERNIGHT_PROGRESS.md`` for the full numbers):

  - Real WOMD roadgraph point spacing is already near-uniform, ~1.0 m
    (measured p50 ~= 0.9923 m, p95 ~= 0.9990 m, max ~= 1.0001 m across
    6076 segments from 85 real maneuvers spanning TRAIN, VALIDATION,
    and all 8 chained maneuvers). This is WOMD's own fixed roadgraph
    sampling rate, not this repo's choice. ASMC's reference smoothing
    stack (10 m window, 0.5 m resampling, 300-iteration curvature
    relaxation) was tuned for a DIFFERENT data source -- a global path
    planner's own dense, possibly irregular output -- not a
    pre-sampled, already-uniform roadgraph point cloud. Given the
    measured near-uniform spacing, this module does NOT resample:
    doing so would discard real WOMD points and manufacture geometry
    that was not observed, without a measured problem to justify it.

  - No near-duplicate point was found in that same real-data sample
    (minimum observed segment length ~= 0.394 m across the full
    85-maneuver/178-lane audit; zero segments under 0.1 m, let alone
    under 1 cm). ``ReferenceLine`` still performs exact/near-duplicate
    cleanup (points closer than ``DUPLICATE_POINT_EPS_M`` are merged)
    purely as defensive input validation -- e.g. against a future data
    source, a hand-built synthetic polyline, or a degenerate
    single/zero-length lane -- not because real WOMD data has been
    observed to need it. ``DUPLICATE_POINT_EPS_M = 1e-3`` (1 mm) was
    chosen well below the smallest real segment length actually
    measured (0.394 m, nearly 400x larger), so it will not fire on any
    real lane geometry seen so far, while still collapsing genuinely
    coincident/duplicated points (e.g. two roadgraph samples at the
    exact same location) so that finite-difference yaw/curvature never
    divides by ~0 spacing.

  - Yaw and curvature use simple finite differences over (x, y) --
    the same general approach ``lane_geometry.py`` itself uses
    elsewhere in this repo (e.g. ``merge_reference.py``'s
    ``_finite_difference_directions``), for consistency. Central
    differences are used at interior points and one-sided differences
    at the two endpoints; curvature is the turning rate of yaw with
    respect to arc length (``d(yaw)/ds``), and curvature derivative is
    a further central/one-sided finite difference of curvature with
    respect to s. At ~1 m point spacing this is well-conditioned (no
    observed non-finite output across the full real-WOMD audit); if a
    future data source has much finer or noisier spacing, this
    approach should be revisited then, backed by new measurements --
    not preemptively.

  - Interpolation at an arbitrary query s is linear, matching
    ``merge_reference.py``'s existing ``_interpolate_polyline_xy``
    convention in this repo, for (x, y). Yaw is interpolated via
    circular (angle-aware) linear interpolation so it does not jump
    across the +-pi branch cut; curvature and curvature_derivative use
    plain linear interpolation.

  - This is an OPEN path (merges are never loops in this domain, per
    the task brief): queries at s < 0 or s > s[-1] do not wrap or
    clamp silently. Following ``project_point_to_polyline_signed``'s
    own precedent in ``lane_geometry.py`` (the existing "unclamped,
    extrapolate along the boundary segment's own tangent" behavior,
    added specifically so relative ordering of two points on the same
    side of a boundary is preserved), ``ReferenceLine.interpolate``
    linearly extrapolates (x, y) along the first/last segment's own
    tangent direction past the boundary, and holds yaw/curvature/
    curvature_derivative constant at the boundary value (curvature
    extrapolation beyond measured geometry is not physically
    meaningful; holding is the documented, conservative choice).
    ``ReferenceLine.project`` reuses
    ``project_point_to_polyline_signed`` directly for the same reason:
    it is the one function in this repo already proven (Stage B-0) to
    get relative ordering right near a lane boundary, and Stage 3-A
    should not reinvent it.
"""

import dataclasses
import time
from typing import Optional

import numpy as np

from src.scenarios.lane_geometry import (
    LanePolyline,
    compute_arc_length,
    project_point_to_polyline_signed,
)

# See module docstring: real WOMD segment lengths measured well above
# this (min ~= 0.394 m across the full audit); this threshold exists
# for defensive/synthetic input robustness, not because real data
# needs it.
DUPLICATE_POINT_EPS_M = 1e-3


class InvalidReferenceGeometryError(ValueError):
    """Raised when input geometry cannot form a valid ReferenceLine
    (non-finite input, or fewer than 2 distinct points after
    duplicate cleanup)."""


@dataclasses.dataclass(frozen=True)
class ReferenceLine:
    """A Frenet reference line built from one ``LanePolyline``.

    All arrays share length N (N >= 2) and are indexed identically:
    index i is the point at arc length ``s[i]``.
    """

    lane_id: int
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    yaw: np.ndarray
    curvature: np.ndarray
    curvature_derivative: np.ndarray

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @staticmethod
    def from_lane_polyline(polyline: LanePolyline) -> "ReferenceLine":
        """Builds a ReferenceLine from an already-reconstructed
        ``LanePolyline`` (see ``src.scenarios.lane_geometry``).

        Raises:
            InvalidReferenceGeometryError: if ``polyline.xy`` contains
                any non-finite value, or fewer than 2 distinct points
                remain after duplicate-point cleanup.
        """

        xy = np.asarray(polyline.xy, dtype=np.float64)

        if xy.shape[0] == 0 or not np.all(np.isfinite(xy)):
            raise InvalidReferenceGeometryError(
                f"Lane {polyline.lane_id}: input xy is empty or contains "
                "non-finite values; cannot build a ReferenceLine."
            )

        clean_xy = _remove_near_duplicate_points(xy, DUPLICATE_POINT_EPS_M)

        if clean_xy.shape[0] < 2:
            raise InvalidReferenceGeometryError(
                f"Lane {polyline.lane_id}: fewer than 2 distinct points "
                f"remain after duplicate cleanup (had {xy.shape[0]} raw "
                f"points, {clean_xy.shape[0]} distinct); cannot build a "
                "ReferenceLine."
            )

        s = compute_arc_length(clean_xy)
        yaw = _compute_yaw(clean_xy)
        curvature = _compute_curvature(yaw, s)
        curvature_derivative = _compute_derivative_wrt_s(curvature, s)

        if not (
            np.all(np.isfinite(s))
            and np.all(np.isfinite(yaw))
            and np.all(np.isfinite(curvature))
            and np.all(np.isfinite(curvature_derivative))
        ):
            raise InvalidReferenceGeometryError(
                f"Lane {polyline.lane_id}: derived geometry (s/yaw/"
                "curvature/curvature_derivative) contains non-finite "
                "values."
            )

        return ReferenceLine(
            lane_id=polyline.lane_id,
            x=clean_xy[:, 0].copy(),
            y=clean_xy[:, 1].copy(),
            s=s,
            yaw=yaw,
            curvature=curvature,
            curvature_derivative=curvature_derivative,
        )

    # ------------------------------------------------------------------
    # derived properties
    # ------------------------------------------------------------------

    @property
    def length_m(self) -> float:
        return float(self.s[-1])

    @property
    def xy(self) -> np.ndarray:
        return np.stack([self.x, self.y], axis=1)

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def interpolate(self, query_s: float) -> "ReferenceState":
        """Interpolates (x, y, yaw, curvature, curvature_derivative) at
        an arbitrary arc length.

        For ``query_s`` inside ``[s[0], s[-1]]``: linear interpolation
        of x, y, curvature, curvature_derivative; circular (angle-
        aware) linear interpolation of yaw so the +-pi branch cut is
        never crossed incorrectly.

        For ``query_s`` outside that domain (upstream of the start or
        downstream of the end -- an open path, never a loop): (x, y)
        is linearly extrapolated along the boundary segment's own
        tangent (matching ``project_point_to_polyline_signed``'s
        existing precedent in ``lane_geometry.py``); yaw, curvature,
        and curvature_derivative are held at the boundary point's
        value (curvature is not measured beyond the reconstructed
        geometry, so extrapolating it would fabricate data).
        """

        s = self.s
        n = s.shape[0]

        if query_s < s[0]:
            return self._extrapolate(query_s, boundary_index=0)
        if query_s > s[-1]:
            return self._extrapolate(query_s, boundary_index=n - 1)

        # searchsorted gives the insertion index i such that
        # s[i-1] <= query_s <= s[i] (for interior query_s); clip to
        # a valid segment [idx-1, idx].
        idx = int(np.clip(np.searchsorted(s, query_s), 1, n - 1))
        s0, s1 = s[idx - 1], s[idx]
        t = 0.0 if s1 == s0 else (query_s - s0) / (s1 - s0)

        x = self.x[idx - 1] + t * (self.x[idx] - self.x[idx - 1])
        y = self.y[idx - 1] + t * (self.y[idx] - self.y[idx - 1])
        yaw = _slerp_angle(self.yaw[idx - 1], self.yaw[idx], t)
        curvature = self.curvature[idx - 1] + t * (
            self.curvature[idx] - self.curvature[idx - 1]
        )
        curvature_derivative = self.curvature_derivative[idx - 1] + t * (
            self.curvature_derivative[idx] - self.curvature_derivative[idx - 1]
        )

        return ReferenceState(
            s=float(query_s),
            x=float(x),
            y=float(y),
            yaw=float(yaw),
            curvature=float(curvature),
            curvature_derivative=float(curvature_derivative),
        )

    def _extrapolate(self, query_s: float, boundary_index: int) -> "ReferenceState":
        n = self.s.shape[0]
        if boundary_index == 0:
            tangent_idx0, tangent_idx1 = 0, 1
        else:
            tangent_idx0, tangent_idx1 = n - 2, n - 1

        p0 = np.array([self.x[tangent_idx0], self.y[tangent_idx0]])
        p1 = np.array([self.x[tangent_idx1], self.y[tangent_idx1]])
        seg_len = self.s[tangent_idx1] - self.s[tangent_idx0]
        tangent = (p1 - p0) / seg_len if seg_len > 0.0 else np.zeros(2)

        boundary_s = self.s[boundary_index]
        boundary_xy = np.array([self.x[boundary_index], self.y[boundary_index]])
        delta_s = query_s - boundary_s
        extrapolated_xy = boundary_xy + delta_s * tangent

        return ReferenceState(
            s=float(query_s),
            x=float(extrapolated_xy[0]),
            y=float(extrapolated_xy[1]),
            yaw=float(self.yaw[boundary_index]),
            curvature=float(self.curvature[boundary_index]),
            curvature_derivative=float(self.curvature_derivative[boundary_index]),
        )

    def project(self, x: float, y: float) -> dict:
        """Projects a Cartesian point onto this reference line.

        Reuses ``lane_geometry.project_point_to_polyline_signed``
        directly against a synthetic ``LanePolyline`` view of this
        reference line's own (cleaned) points -- the Stage B-0 fixed
        function is the one already proven in this repo to preserve
        relative ordering across a lane boundary (unclamped
        longitudinal coordinate), so Stage 3-A reuses it rather than
        reimplementing projection from scratch.

        Returns:
            The same dict shape as
            ``project_point_to_polyline_signed``: arc_length_m
            (signed, may be < 0 or > length_m), lateral_distance_m
            (positive = LEFT of travel direction, matching
            ``lane_geometry.py``'s own convention), heading_rad,
            segment_index.
        """

        view = LanePolyline(
            lane_id=self.lane_id,
            lane_type=0,
            xy=self.xy,
            direction=_unit_tangents(self.xy),
            arc_length=self.s,
        )
        return project_point_to_polyline_signed(view, x, y)


@dataclasses.dataclass(frozen=True)
class ReferenceState:
    """One interpolated/extrapolated point on a ReferenceLine."""

    s: float
    x: float
    y: float
    yaw: float
    curvature: float
    curvature_derivative: float


# ======================================================================
# helpers
# ======================================================================


def _remove_near_duplicate_points(xy: np.ndarray, eps_m: float) -> np.ndarray:
    """Drops consecutive points closer than ``eps_m`` (keeps the
    first of each near-duplicate run). Order-preserving; does not
    reorder or deduplicate non-adjacent points (a lane polyline is
    already ordered by ``lane_geometry.py``, so only adjacent
    duplicates are physically meaningful here)."""

    if xy.shape[0] <= 1:
        return xy

    keep = [0]
    for i in range(1, xy.shape[0]):
        last = xy[keep[-1]]
        if np.hypot(xy[i, 0] - last[0], xy[i, 1] - last[1]) > eps_m:
            keep.append(i)

    return xy[np.array(keep, dtype=np.int64)]


def _compute_yaw(xy: np.ndarray) -> np.ndarray:
    """Per-point yaw (rad) via finite differences: central difference
    at interior points, one-sided at the two endpoints. Matches the
    general finite-difference approach already used elsewhere in this
    repo (e.g. ``merge_reference._finite_difference_directions``)."""

    n = xy.shape[0]
    yaw = np.empty(n, dtype=np.float64)

    if n == 1:
        return np.zeros(1, dtype=np.float64)

    diffs = np.diff(xy, axis=0)  # (n-1, 2): segment i is xy[i+1]-xy[i]

    yaw[0] = np.arctan2(diffs[0, 1], diffs[0, 0])
    yaw[-1] = np.arctan2(diffs[-1, 1], diffs[-1, 0])

    if n > 2:
        # Central difference: average the incoming and outgoing
        # segment vectors at each interior point, via atan2 of the
        # summed vector (avoids naive angle-averaging wraparound
        # issues for any single point, since both vectors point in
        # a locally-similar direction for a real lane polyline).
        incoming = diffs[:-1]
        outgoing = diffs[1:]
        summed = incoming + outgoing
        yaw[1:-1] = np.arctan2(summed[:, 1], summed[:, 0])

    return yaw


def _compute_derivative_wrt_s(values: np.ndarray, s: np.ndarray) -> np.ndarray:
    """d(values)/ds via finite differences: central at interior
    points, one-sided at the two endpoints."""

    n = values.shape[0]
    derivative = np.empty(n, dtype=np.float64)

    if n == 1:
        return np.zeros(1, dtype=np.float64)

    ds_fwd = s[1:] - s[:-1]
    ds_fwd_safe = np.where(ds_fwd == 0.0, 1e-9, ds_fwd)

    derivative[0] = (values[1] - values[0]) / ds_fwd_safe[0]
    derivative[-1] = (values[-1] - values[-2]) / ds_fwd_safe[-1]

    if n > 2:
        s_prev = s[:-2]
        s_next = s[2:]
        span = s_next - s_prev
        span_safe = np.where(span == 0.0, 1e-9, span)
        derivative[1:-1] = (values[2:] - values[:-2]) / span_safe

    return derivative


def _compute_curvature(yaw: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Curvature kappa = d(yaw)/ds, with yaw differences unwrapped to
    (-pi, pi] before differencing so a branch-cut crossing does not
    manufacture a spurious near-2*pi/ds spike."""

    n = yaw.shape[0]
    if n == 1:
        return np.zeros(1, dtype=np.float64)

    unwrapped_yaw = np.unwrap(yaw)
    return _compute_derivative_wrt_s(unwrapped_yaw, s)


def _slerp_angle(a: float, b: float, t: float) -> float:
    """Circular linear interpolation between two angles (radians),
    taking the shortest angular path, so interpolated yaw near +-pi
    does not jump the wrong way around."""

    delta = _wrap_angle(b - a)
    return _wrap_angle(a + t * delta)


def _wrap_angle(angle_rad: float) -> float:
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


def _unit_tangents(xy: np.ndarray) -> np.ndarray:
    """Per-point unit direction vectors, matching the field
    ``LanePolyline.direction`` expects (used only so ``project`` can
    hand ``project_point_to_polyline_signed`` a well-formed
    ``LanePolyline`` view; that function does not itself read
    ``direction`` today, but a well-formed view is cheap and keeps
    this call site future-proof)."""

    n = xy.shape[0]
    if n == 1:
        return np.zeros((1, 2), dtype=np.float64)

    diffs = np.diff(xy, axis=0)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1e-9, norms)
    unit_diffs = diffs / norms

    directions = np.zeros_like(xy)
    directions[:-1] += unit_diffs
    directions[1:] += unit_diffs
    row_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    row_norms = np.where(row_norms == 0.0, 1e-9, row_norms)
    return directions / row_norms


def build_reference_line_timed(polyline: LanePolyline):
    """Convenience wrapper used by the audit script: returns
    (ReferenceLine, wall_clock_seconds) via ``time.perf_counter``."""

    start = time.perf_counter()
    reference = ReferenceLine.from_lane_polyline(polyline)
    elapsed = time.perf_counter() - start
    return reference, elapsed
