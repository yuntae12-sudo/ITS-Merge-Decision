"""Lane geometry reconstruction from Waymax roadgraph points.

Waymax's ``RoadgraphPoints`` gives a flat, sampled point cloud: each
point has an ``id`` (map feature it belongs to), a ``type``, an (x, y,
z) position, a unit direction vector, and a validity flag. There is no
explicit polyline ordering or topology (predecessor/successor lane
ids) in this representation -- only what can be derived from the
sampled points themselves.

What we can rely on (per-point, directly from the data):
    - id groups points belonging to the same map feature.
    - type identifies lane vs. non-lane features.
    - the direction vector at a point is the feature's local heading,
      i.e. WOMD's own encoding of lane travel direction.

What we do NOT assume:
    - That points sharing an id are stored in polyline order in the
      underlying array. WOMD does not document this as guaranteed, so
      each polyline's point order is reconstructed explicitly here via
      nearest-neighbor chaining, rather than trusting raw array order.
    - That the reconstructed geometric sequence runs the same way as
      WOMD's travel direction. Chaining alone only recovers *a* valid
      spatial order, not which end is the start, so the sequence is
      compared against the WOMD ``direction`` field and reversed if it
      runs opposite (see ``_align_to_travel_direction``). Downstream
      arc_length, heading, and lateral-distance sign all depend on
      this alignment being correct.
    - Any lane connectivity (predecessor/successor/neighbor lane ids).
      That topology is not present in roadgraph_points at all; only
      what can be inferred geometrically (e.g. endpoint proximity) is
      available, and this module does not infer it -- it only
      reconstructs per-lane polylines.
"""

import dataclasses
from typing import List, Optional

import numpy as np

# Map element type ids that represent a drivable lane centerline, per
# the WOMD roadgraph type mapping already validated in Phase 0
# (scripts/run_scene.py MAP_ELEMENT_TYPE_NAMES).
#
# ALL_LANE_TYPE_IDS includes every lane-shaped centerline type, used
# for visualization/context. VEHICLE_LANE_TYPE_IDS excludes bike lanes
# and is the default candidate set for anything related to an
# autonomous *vehicle* driving on a lane (ego lane assignment, merge
# detection): a vehicle cannot merge into or be assigned to a bike
# lane in this study.
ALL_LANE_TYPE_IDS = frozenset({1, 2, 3})  # FREEWAY, SURFACE_STREET, BIKE_LANE
VEHICLE_LANE_TYPE_IDS = frozenset({1, 2})  # FREEWAY, SURFACE_STREET


@dataclasses.dataclass(frozen=True)
class LanePolyline:
    """A single lane feature's points, ordered along its centerline."""

    lane_id: int
    lane_type: int
    xy: np.ndarray  # shape (N, 2), ordered along the polyline
    direction: np.ndarray  # shape (N, 2), unit heading vectors
    arc_length: np.ndarray  # shape (N,), cumulative distance from xy[0]


def _chain_from(start: int, xy: np.ndarray) -> np.ndarray:
    """Nearest-neighbor chain over all points, starting at `start`."""

    num_points = xy.shape[0]

    visited = np.zeros(num_points, dtype=bool)
    order = np.empty(num_points, dtype=np.int64)

    order[0] = start
    visited[start] = True

    for step in range(1, num_points):

        last_xy = xy[order[step - 1]]
        remaining = np.flatnonzero(~visited)

        distances = np.hypot(
            xy[remaining, 0] - last_xy[0],
            xy[remaining, 1] - last_xy[1],
        )

        next_idx = remaining[int(np.argmin(distances))]

        order[step] = next_idx
        visited[next_idx] = True

    return order


def _order_points_by_chaining(
    xy: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Reconstructs polyline point order via nearest-neighbor chaining.

    The per-point direction vector is not used to pick the starting
    point: on a real polyline, consecutive points all share nearly the
    same heading, so "points away from centroid" does not reliably
    single out an endpoint. Instead this uses a standard two-pass
    farthest-point heuristic: chain once from an arbitrary point, take
    the point that chain ends on (the graph's approximate diameter
    endpoint), then chain again from there. This does not require
    trusting the raw array order of same-id points.

    Args:
        xy: (N, 2) point positions for a single feature id.
        direction: (N, 2) per-point unit direction vectors (unused by
            the ordering itself; kept in the signature so callers can
            rely on a stable interface if direction-aware ordering is
            added later).

    Returns:
        Index array of length N giving the chained visiting order.
    """

    del direction  # not used by the current ordering heuristic

    num_points = xy.shape[0]

    if num_points <= 2:
        return np.arange(num_points)

    first_pass = _chain_from(0, xy)
    farthest_point = int(first_pass[-1])

    return _chain_from(farthest_point, xy)


def _align_to_travel_direction(
    xy: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Orients a geometrically-ordered polyline to match WOMD direction.

    Geometric chaining (``_order_points_by_chaining``) only recovers a
    valid spatial sequence -- it has no notion of which end is the
    start. This walks the *given* order's segment vectors and compares
    them against the per-point WOMD ``direction`` (dir_x, dir_y), which
    is Waymax's own encoding of local lane travel direction. If the
    sequence runs opposite to WOMD's direction on average, it is
    reversed so callers can rely on ``xy[0]`` being the travel-direction
    start and ``arc_length[0] == 0`` at that start.

    Args:
        xy: (N, 2) points in geometric chain order.
        direction: (N, 2) WOMD per-point direction vectors, indexed the
            same as `xy`.

    Returns:
        Index array of length N: identity if already aligned, reversed
        (N-1 .. 0) if the chain ran opposite to WOMD's direction. A
        single point has no direction to compare and is always
        returned as identity.
    """

    num_points = xy.shape[0]
    identity = np.arange(num_points)

    if num_points < 2:
        return identity

    segment_vec = np.diff(xy, axis=0)
    # Compare each segment to the WOMD direction at its start point:
    # a segment vector should point the same way as the lane's local
    # direction there if the sequence runs with the lane, opposite if
    # it runs against it.
    alignment = np.einsum("ij,ij->i", segment_vec, direction[:-1])

    # A robust (median, not mean) alignment score: a handful of noisy
    # or near-duplicate points should not flip the decision for an
    # otherwise consistently-oriented polyline.
    if np.median(alignment) < 0.0:
        return identity[::-1]

    return identity


def compute_arc_length(xy: np.ndarray) -> np.ndarray:
    """Cumulative Euclidean arc length along an ordered polyline.

    Args:
        xy: (N, 2) point positions, ordered along the polyline. When
            called from ``extract_lane_polylines``, this order has
            already been aligned to WOMD's travel direction, so
            element 0 is the travel-direction start.

    Returns:
        (N,) array where element 0 is 0.0 and element i is the summed
        segment length from xy[0] to xy[i].
    """

    if xy.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    segment_lengths = np.hypot(
        np.diff(xy[:, 0]),
        np.diff(xy[:, 1]),
    )

    return np.concatenate(
        [[0.0], np.cumsum(segment_lengths)]
    )


def extract_lane_polylines(
    roadgraph_points,
    lane_type_ids: frozenset = VEHICLE_LANE_TYPE_IDS,
) -> List[LanePolyline]:
    """Reconstructs one ordered polyline per lane feature id.

    Defaults to ``VEHICLE_LANE_TYPE_IDS`` (excludes bike lanes): this
    study's ego lane assignment and merge detection only consider
    lanes a vehicle can occupy. Pass ``ALL_LANE_TYPE_IDS`` explicitly
    for visualization/context that should also show bike lanes.

    Args:
        roadgraph_points: Waymax ``RoadgraphPoints`` (as found on
            ``SimulatorState.roadgraph_points``).
        lane_type_ids: map element type ids considered a lane
            centerline (see ``VEHICLE_LANE_TYPE_IDS``,
            ``ALL_LANE_TYPE_IDS``).

    Returns:
        List of ``LanePolyline``, one per distinct valid lane feature
        id, each with points ordered along the centerline.
    """

    ids = np.asarray(roadgraph_points.ids)
    types = np.asarray(roadgraph_points.types)
    valid = np.asarray(roadgraph_points.valid).astype(bool)

    x = np.asarray(roadgraph_points.x)
    y = np.asarray(roadgraph_points.y)
    dir_x = np.asarray(roadgraph_points.dir_x)
    dir_y = np.asarray(roadgraph_points.dir_y)

    lane_mask = valid & np.isin(types, list(lane_type_ids))

    polylines = []

    for lane_id in np.unique(ids[lane_mask]):

        feature_mask = lane_mask & (ids == lane_id)
        feature_idx = np.flatnonzero(feature_mask)

        xy = np.stack(
            [x[feature_idx], y[feature_idx]], axis=1
        ).astype(np.float64)

        direction = np.stack(
            [dir_x[feature_idx], dir_y[feature_idx]], axis=1
        ).astype(np.float64)

        chain_order = _order_points_by_chaining(xy, direction)

        chained_xy = xy[chain_order]
        chained_direction = direction[chain_order]

        # Geometric chaining recovers a valid sequence but not which
        # end is the start; align it to WOMD's own travel direction
        # (dir_x, dir_y) so xy[0] is the travel-direction start and
        # arc_length is measured from there.
        alignment_order = _align_to_travel_direction(
            chained_xy, chained_direction
        )

        ordered_xy = chained_xy[alignment_order]
        ordered_direction = chained_direction[alignment_order]

        polylines.append(
            LanePolyline(
                lane_id=int(lane_id),
                lane_type=int(types[feature_idx[0]]),
                xy=ordered_xy,
                direction=ordered_direction,
                arc_length=compute_arc_length(ordered_xy),
            )
        )

    return polylines


def nearest_lane_candidates(
    polylines: List[LanePolyline],
    x: float,
    y: float,
    k: Optional[int] = None,
) -> List[tuple]:
    """Ranks lane polylines by distance from a query point.

    Distance to a polyline is the minimum distance from (x, y) to any
    of its points (a cheap proxy; ``project_point_to_polyline`` gives
    the exact segment-projected distance for a chosen candidate).

    Args:
        polylines: candidate lanes, e.g. from ``extract_lane_polylines``.
        x: query point x.
        y: query point y.
        k: if set, only the k nearest candidates are returned.

    Returns:
        List of (lane_id, min_point_distance) sorted ascending by
        distance.
    """

    ranked = []

    for polyline in polylines:

        distances = np.hypot(
            polyline.xy[:, 0] - x,
            polyline.xy[:, 1] - y,
        )

        ranked.append((polyline.lane_id, float(np.min(distances))))

    ranked.sort(key=lambda item: item[1])

    if k is not None:
        ranked = ranked[:k]

    return ranked


def project_point_to_polyline(polyline: LanePolyline, x: float, y: float):
    """Projects a point onto the nearest segment of a lane polyline.

    Args:
        polyline: the lane to project onto.
        x: query point x.
        y: query point y.

    Returns:
        A dict with:
            arc_length_m: longitudinal position along the polyline at
                the projection (interpolated between the segment's
                endpoints' arc lengths).
            lateral_distance_m: signed perpendicular distance from the
                query point to the segment (positive = left of the
                segment direction).
            heading_rad: the segment's heading (atan2 of its direction).
            segment_index: index of the polyline segment used.
    """

    xy = polyline.xy

    if xy.shape[0] < 2:
        raise ValueError(
            f"Polyline {polyline.lane_id} has fewer than 2 points; "
            "cannot project onto a segment."
        )

    seg_start = xy[:-1]
    seg_end = xy[1:]
    seg_vec = seg_end - seg_start
    seg_len_sq = np.einsum("ij,ij->i", seg_vec, seg_vec)
    seg_len_sq = np.where(seg_len_sq == 0.0, 1e-12, seg_len_sq)

    point = np.array([x, y], dtype=np.float64)
    to_point = point[None, :] - seg_start

    t = np.einsum("ij,ij->i", to_point, seg_vec) / seg_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)

    projected = seg_start + t_clamped[:, None] * seg_vec
    distances = np.hypot(
        point[0] - projected[:, 0],
        point[1] - projected[:, 1],
    )

    segment_index = int(np.argmin(distances))

    seg_direction = seg_vec[segment_index]
    seg_direction_norm = np.linalg.norm(seg_direction)

    if seg_direction_norm == 0.0:
        heading = 0.0
        lateral_sign = 1.0
    else:
        unit_direction = seg_direction / seg_direction_norm
        heading = float(np.arctan2(unit_direction[1], unit_direction[0]))
        # Left-hand perpendicular of travel direction is positive.
        perpendicular = np.array([-unit_direction[1], unit_direction[0]])
        offset = point - projected[segment_index]
        lateral_sign = np.sign(np.dot(offset, perpendicular)) or 1.0

    arc_start = polyline.arc_length[segment_index]
    arc_end = polyline.arc_length[segment_index + 1]
    arc_length_m = arc_start + t_clamped[segment_index] * (
        arc_end - arc_start
    )

    return {
        "arc_length_m": float(arc_length_m),
        "lateral_distance_m": float(
            lateral_sign * distances[segment_index]
        ),
        "heading_rad": heading,
        "segment_index": segment_index,
    }
