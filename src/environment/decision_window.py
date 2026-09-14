"""Phase 2 Stage B-2.8: production ``decision_start_frame`` computation.

Productionizes Stage B-2.7's diagnostic ``earliest_causal_decision_frame``
finding. This is a Phase-2-only environment concept -- it does NOT
redefine Phase 1's ``source_start_frame`` (which remains the delayed,
hysteresis-filtered stable-lane-entry frame used by the canonical
detector/manifest) or ``merge_start_frame`` (which remains the
physical merge-region reference the maneuver tables store).

Definition:

    decision_start_frame :=
        the first frame t (scanning forward from frame 0) at which
        the RAW (pre-persistence, pre-hysteresis) per-frame lane
        assignment ``assign_lane(ego_x[t], ego_y[t], ego_yaw[t], ...)``
        already equals the maneuver's known source lane id.

This reuses ``assign_lane`` exactly as Phase 1's own pipeline does
(same config, same scoring) -- it is a single forward pass over
already-materialized ``record.state.log_trajectory`` arrays, with:

  - no future information: frame t's assignment only reads frame t's
    (x, y, yaw) and the scene's static lane polylines;
  - no policy dependency: it never touches any policy/rollout, only
    the logged trajectory that already exists once a scenario is
    loaded;
  - no persistence/hysteresis delay: unlike ``source_start_frame``
    (Stage B-2.7 Section 4/5 finding: delayed by up to
    ``persistence_frames - 1`` frames), this is the raw match itself.

If no frame in the trajectory raw-matches the source lane, resolution
fails and the caller must decide how to handle it (Stage B-2.8 Section
5: never silently fall back to ``merge_start_frame``).
"""

import dataclasses
from typing import List, Optional

import numpy as np

from src.scenarios.lane_assignment import LaneAssignmentConfig, assign_lane
from src.scenarios.lane_geometry import LanePolyline


class DecisionStartUnresolvedError(RuntimeError):
    """Raised when no frame in the logged trajectory raw-matches the
    maneuver's source lane -- i.e. ``decision_start_frame`` cannot be
    resolved. Callers must not silently fall back to
    ``merge_start_frame`` on this error (Stage B-2.8 Section 6)."""


@dataclasses.dataclass(frozen=True)
class DecisionWindowResult:
    decision_start_frame: int
    """The resolved Phase-2 episode/simulation start frame."""

    raw_available_history: int
    """merge_start_frame - decision_start_frame. Always >= 0 for a
    resolvable maneuver (Stage B-2.7 Section 4 confirmed 0/168
    negative cases); a negative value would indicate a causality
    violation and must be treated as a hard error by callers, never
    clamped silently."""


def compute_decision_start_frame(
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_yaw: np.ndarray,
    ego_valid: np.ndarray,
    polylines: List[LanePolyline],
    lane_assignment_config: LaneAssignmentConfig,
    source_lane_id: int,
    merge_start_frame: int,
) -> DecisionWindowResult:
    """Computes ``decision_start_frame`` for one maneuver from its
    already-loaded logged trajectory arrays (Stage B-2.7's exact
    method, productionized).

    Args:
        ego_x, ego_y, ego_yaw, ego_valid: (T,) logged SDC trajectory
            arrays for the FULL scene (frame 0 through the scene's
            last logged frame) -- never a simulated/future array.
        polylines: the scene's static lane polylines.
        lane_assignment_config: reused unmodified from Phase 1.
        source_lane_id: the maneuver's known source lane id (map/route
            information, from the canonical ``lane_chain`` -- NOT
            derived from the ego's future trajectory; see module
            docstring and Stage B-2.8 Section 9).
        merge_start_frame: the maneuver's canonical physical
            merge-region start frame, used only to compute
            ``raw_available_history`` for diagnostics/tests -- not an
            upper search bound (a later raw match than
            ``merge_start_frame`` is not expected but is not silently
            excluded either; see ``DecisionWindowResult``).

    Returns:
        DecisionWindowResult.

    Raises:
        DecisionStartUnresolvedError: no valid frame's raw lane
            assignment ever matches ``source_lane_id``.
    """

    num_frames = ego_x.shape[0]
    for t in range(num_frames):
        if not ego_valid[t]:
            continue
        assignment = assign_lane(
            float(ego_x[t]),
            float(ego_y[t]),
            float(ego_yaw[t]),
            polylines,
            lane_assignment_config,
            frame_index=t,
        )
        if assignment.lane_id == source_lane_id:
            return DecisionWindowResult(
                decision_start_frame=t,
                raw_available_history=merge_start_frame - t,
            )

    raise DecisionStartUnresolvedError(
        f"No frame's raw lane assignment matches source_lane_id="
        f"{source_lane_id} (merge_start_frame={merge_start_frame}); "
        "decision_start_frame is unresolvable for this maneuver."
    )
