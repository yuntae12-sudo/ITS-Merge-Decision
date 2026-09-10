"""Phase 2 policy-independent observation builder.

Builds the FINAL_OBSERVATION_VECTOR confirmed in Stage A's Final
Closure Patch and revised in Stage B-0 (Section 1/2 of the Stage B-0
completion report):

    0  v_e
    1  d_m

    2  target_front_present
    3  target_front_gap
    4  target_front_relative_speed
    5  target_front_ttc

    6  target_rear_present
    7  target_rear_gap
    8  target_rear_relative_speed
    9  target_rear_ttc

    10 source_front_present
    11 source_front_gap
    12 source_front_relative_speed
    13 source_front_ttc

This module is POLICY-INDEPENDENT: both the FSM baseline and the PPO
policy must call ``build_observation`` and receive byte-identical
vectors for the same simulated state, per the FSM/PPO fairness
constraint fixed in Stage A. It contains no decision logic of its own.

Reuses (does not reimplement) ``src.scenarios.scenario_features``'s
``extract_interaction_features``/``find_target_lane_front_rear`` for
both the TARGET lane (merge destination) and the SOURCE lane (current
lane, needed for the FOLLOW action's lead-vehicle information -- see
Stage A Final Closure Patch Section 1 and the Stage B-0 prompt's
14D-candidate resolution). Front/rear selection there already carries
the Stage B-0 target-lane longitudinal-ordering fix
(``project_point_to_polyline_signed``); this module does not duplicate
or re-derive that fix.

v_e definition (Stage B-0 Section 2, overriding Stage A's original
target-lane-tangent-projected `ego_longitudinal_speed_mps`): SCALAR
ground speed ``sqrt(vx**2 + vy**2)``, not a lane-tangent projection.
This was changed after a diagnostic proved the target-lane-tangent
projection can go negative purely because the target lane's local
tangent, sampled 22+ m away from ego's true position at an early
pre-merge frame, points in an unrelated direction -- a projection
artifact, not real motion. Scalar speed has no such lane-dependence.
"""

import dataclasses
from typing import Optional

import numpy as np

from src.scenarios.lane_geometry import LanePolyline
from src.scenarios.merge_detector import compute_remaining_merge_distance
from src.scenarios.scenario_features import (
    AgentSelectionConfig,
    extract_interaction_features,
)

# Fixed finite substitute for an infinite/absent TTC (Stage A Section
# 6, unchanged in Stage B-0): exceeds any physically meaningful TTC
# observed in the Phase 1 training pool, so it never collides with a
# real value, while keeping every observation entry finite.
TTC_CAP_S = 100.0

OBSERVATION_DIM = 14

OBSERVATION_FIELD_NAMES = (
    "v_e",
    "d_m",
    "target_front_present",
    "target_front_gap",
    "target_front_relative_speed",
    "target_front_ttc",
    "target_rear_present",
    "target_rear_gap",
    "target_rear_relative_speed",
    "target_rear_ttc",
    "source_front_present",
    "source_front_gap",
    "source_front_relative_speed",
    "source_front_ttc",
)


@dataclasses.dataclass(frozen=True)
class ObservationInputs:
    """Raw per-frame state ``build_observation`` needs.

    All array fields are (num_objects,) at a single frame, matching
    the calling convention already used by
    ``extract_interaction_features``/``find_target_lane_front_rear``.
    Callers (the live environment, or an offline re-materialization
    script) are responsible for slicing WOMD log/simulated trajectory
    arrays to one frame before constructing this.
    """

    ego_id: int
    ego_x: float
    ego_y: float
    ego_vel_x: float
    ego_vel_y: float
    ego_length_m: float

    source_polyline: LanePolyline
    target_polyline: LanePolyline

    merge_end_s: float
    """Static per-scene source-lane arc length at the merge point (from
    ``merge_detector.compute_merge_start_end_s``); passed through, not
    recomputed here."""

    ego_source_arc_length_m: Optional[float]
    """Ego's current SOURCE-lane arc length, used only for d_m. Pass
    ``None`` if unavailable/off-lane (see
    ``compute_remaining_merge_distance``)."""

    object_ids: np.ndarray
    object_types: np.ndarray
    valid: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    vel_x: np.ndarray
    vel_y: np.ndarray
    length: np.ndarray

    agent_selection_config: AgentSelectionConfig


def _scalar_speed(vel_x: float, vel_y: float) -> float:
    """v_e (Stage B-0 Section 2): scalar ground speed, not a lane
    -tangent projection. See module docstring for why the projection
    was rejected."""

    return float(np.hypot(vel_x, vel_y))


def _encode_vehicle_slot(
    present: bool,
    gap_m: Optional[float],
    relative_speed_mps: Optional[float],
    ttc_s: float,
):
    """Applies the Stage A/B-0 missing-value encoding to one
    front/rear slot: presence flag + zero-sentinel gap/relative-speed
    + capped TTC. Never emits None, NaN, or inf."""

    if not present:
        return 0.0, 0.0, TTC_CAP_S

    gap = float(gap_m) if gap_m is not None else 0.0
    relative_speed = (
        float(relative_speed_mps) if relative_speed_mps is not None else 0.0
    )
    ttc = min(float(ttc_s), TTC_CAP_S) if np.isfinite(ttc_s) else TTC_CAP_S

    return gap, relative_speed, ttc


def build_observation(inputs: ObservationInputs) -> np.ndarray:
    """Builds the FINAL_OBSERVATION_VECTOR (shape ``(14,)``, dtype
    float64, guaranteed finite) for one simulated/logged frame.

    Policy-independent: FSM and PPO both call this and get the same
    vector for the same ``inputs``.
    """

    d_m = compute_remaining_merge_distance(
        inputs.merge_end_s, inputs.ego_source_arc_length_m
    )

    target_features = extract_interaction_features(
        frame_index=-1,  # bookkeeping only; unused by the callee itself
        target_polyline=inputs.target_polyline,
        merge_distance_m=d_m,
        ego_id=inputs.ego_id,
        ego_x=inputs.ego_x,
        ego_y=inputs.ego_y,
        ego_vel_x=inputs.ego_vel_x,
        ego_vel_y=inputs.ego_vel_y,
        ego_length_m=inputs.ego_length_m,
        object_ids=inputs.object_ids,
        object_types=inputs.object_types,
        valid=inputs.valid,
        x=inputs.x,
        y=inputs.y,
        yaw=inputs.yaw,
        vel_x=inputs.vel_x,
        vel_y=inputs.vel_y,
        length=inputs.length,
        config=inputs.agent_selection_config,
    )

    # SOURCE-lane front only (Stage A Section 1 / Stage B-0 Section 1):
    # reuses the identical function with the source polyline in place
    # of target -- no new geometry code, no source-lane rear (not
    # needed by any confirmed action semantic).
    source_features = extract_interaction_features(
        frame_index=-1,
        target_polyline=inputs.source_polyline,
        merge_distance_m=d_m,
        ego_id=inputs.ego_id,
        ego_x=inputs.ego_x,
        ego_y=inputs.ego_y,
        ego_vel_x=inputs.ego_vel_x,
        ego_vel_y=inputs.ego_vel_y,
        ego_length_m=inputs.ego_length_m,
        object_ids=inputs.object_ids,
        object_types=inputs.object_types,
        valid=inputs.valid,
        x=inputs.x,
        y=inputs.y,
        yaw=inputs.yaw,
        vel_x=inputs.vel_x,
        vel_y=inputs.vel_y,
        length=inputs.length,
        config=inputs.agent_selection_config,
    )

    v_e = _scalar_speed(inputs.ego_vel_x, inputs.ego_vel_y)

    target_front_gap, target_front_rel_speed, target_front_ttc = (
        _encode_vehicle_slot(
            target_features.front_vehicle_id is not None,
            target_features.front_gap_m,
            target_features.front_relative_speed_mps,
            target_features.front_ttc_s,
        )
    )
    target_rear_gap, target_rear_rel_speed, target_rear_ttc = (
        _encode_vehicle_slot(
            target_features.rear_vehicle_id is not None,
            target_features.rear_gap_m,
            target_features.rear_relative_speed_mps,
            target_features.rear_ttc_s,
        )
    )
    source_front_gap, source_front_rel_speed, source_front_ttc = (
        _encode_vehicle_slot(
            source_features.front_vehicle_id is not None,
            source_features.front_gap_m,
            source_features.front_relative_speed_mps,
            source_features.front_ttc_s,
        )
    )

    observation = np.array(
        [
            v_e,
            d_m,
            float(target_features.front_vehicle_id is not None),
            target_front_gap,
            target_front_rel_speed,
            target_front_ttc,
            float(target_features.rear_vehicle_id is not None),
            target_rear_gap,
            target_rear_rel_speed,
            target_rear_ttc,
            float(source_features.front_vehicle_id is not None),
            source_front_gap,
            source_front_rel_speed,
            source_front_ttc,
        ],
        dtype=np.float64,
    )

    assert observation.shape == (OBSERVATION_DIM,)
    assert np.all(np.isfinite(observation)), (
        "build_observation produced a non-finite value: "
        f"{dict(zip(OBSERVATION_FIELD_NAMES, observation.tolist()))}"
    )

    return observation
