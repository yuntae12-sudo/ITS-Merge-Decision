"""Offline temporal interaction evidence for dataset schema v2.

This module may inspect future logged frames to label an event.  It is kept in
``src.scenarios`` and must never be called by the online observation builder.
"""

from typing import Iterable, Optional, Tuple

import numpy as np

from src.scenarios.lane_geometry import LanePolyline, project_point_to_polyline_signed
from src.scenarios.merge_v2 import InteractionEvidence
from src.scenarios.scenario_features import AgentSelectionConfig, extract_interaction_features


def _trailing_gap_order_persistence(
    samples: Iterable[dict], front_id: Optional[int], rear_id: Optional[int]
) -> int:
    count = 0
    for sample in samples:
        if sample["front_vehicle_id"] != front_id or sample["rear_vehicle_id"] != rear_id:
            break
        count += 1
    return count


def extract_temporal_interaction_evidence(
    *,
    record,
    source_polyline: LanePolyline,
    target_polyline: LanePolyline,
    decision_start_frame: int,
    transition_frame: int,
    completion_frame: int,
    target_end_frame: int,
    source_occupancy_frames: int,
    target_stable_frames: int,
    agent_config: AgentSelectionConfig,
    conflict_vehicle_ids: Tuple[int, ...] = (),
    required_gap_order_persistence_frames: int = 5,
):
    """Extract per-frame target-lane gap evidence and its compact summary."""

    traj = record.state.log_trajectory
    ego = record.sdc_index
    object_ids = np.asarray(record.state.object_metadata.ids)
    object_types = np.asarray(record.state.object_metadata.object_types)
    x = np.asarray(traj.x)
    y = np.asarray(traj.y)
    yaw = np.asarray(traj.yaw)
    vel_x = np.asarray(traj.vel_x)
    vel_y = np.asarray(traj.vel_y)
    length = np.asarray(traj.length)
    valid = np.asarray(traj.valid).astype(bool)

    end = min(target_end_frame, x.shape[1] - 1)
    samples = []
    for frame in range(decision_start_frame, end + 1):
        if not valid[ego, frame]:
            continue
        features = extract_interaction_features(
            frame_index=frame,
            target_polyline=target_polyline,
            merge_distance_m=0.0,
            ego_id=record.sdc_id,
            ego_x=float(x[ego, frame]), ego_y=float(y[ego, frame]),
            ego_vel_x=float(vel_x[ego, frame]), ego_vel_y=float(vel_y[ego, frame]),
            ego_length_m=float(length[ego, frame]),
            object_ids=object_ids, object_types=object_types,
            valid=valid[:, frame], x=x[:, frame], y=y[:, frame], yaw=yaw[:, frame],
            vel_x=vel_x[:, frame], vel_y=vel_y[:, frame], length=length[:, frame],
            config=agent_config,
        )
        samples.append({
            "frame": frame,
            "front_vehicle_id": features.front_vehicle_id,
            "front_gap_m": features.front_gap_m,
            "front_relative_speed_mps": features.front_relative_speed_mps,
            "front_ttc_s": features.front_ttc_s,
            "rear_vehicle_id": features.rear_vehicle_id,
            "rear_gap_m": features.rear_gap_m,
            "rear_relative_speed_mps": features.rear_relative_speed_mps,
            "rear_ttc_s": features.rear_ttc_s,
        })

    completion_sample = next(
        (s for s in samples if s["frame"] >= completion_frame), None
    )
    front_id = completion_sample["front_vehicle_id"] if completion_sample else None
    rear_id = completion_sample["rear_vehicle_id"] if completion_sample else None
    post_completion = [s for s in samples if s["frame"] >= completion_frame]
    persistence = _trailing_gap_order_persistence(post_completion, front_id, rear_id)

    start_projection = project_point_to_polyline_signed(
        target_polyline, float(x[ego, decision_start_frame]), float(y[ego, decision_start_frame])
    )
    finish_projection = project_point_to_polyline_signed(
        target_polyline, float(x[ego, completion_frame]), float(y[ego, completion_frame])
    )
    lateral_displacement = max(
        0.0,
        abs(start_projection["lateral_distance_m"])
        - abs(finish_projection["lateral_distance_m"]),
    )
    longitudinal_progress = max(
        0.0,
        finish_projection["arc_length_m"] - start_projection["arc_length_m"],
    )
    evidence = InteractionEvidence(
        decision_start_frame=decision_start_frame,
        commit_frame=transition_frame,
        completion_frame=completion_frame,
        source_occupancy_frames=source_occupancy_frames,
        target_stable_frames=target_stable_frames,
        longitudinal_progress_m=float(longitudinal_progress),
        lateral_displacement_m=float(lateral_displacement),
        front_vehicle_id=front_id,
        rear_vehicle_id=rear_id,
        conflict_vehicle_ids=tuple(conflict_vehicle_ids),
        gap_order_valid_at_entry=(front_id is not None and rear_id is not None),
        gap_order_persistence_frames=persistence,
        required_gap_order_persistence_frames=required_gap_order_persistence_frames,
    )
    return evidence, samples


__all__ = ["extract_temporal_interaction_evidence"]
