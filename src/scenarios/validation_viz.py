"""Pure rendering for manual merge-candidate validation figures.

This module contains NO dataset-selection logic (which scenes/decisions
to render lives in ``dataset_builder.py`` and the CLI scripts, e.g.
``scripts/render_merge_validation.py``) -- only how to draw one PNG for
a single already-selected candidate.

Roadgraph point type ids used below were empirically confirmed by
inspecting a real WOMD scene's ``roadgraph_points.types`` (see Commit E
report): lane-type ids (see ``lane_geometry.ALL_LANE_TYPE_IDS`` -- 1/2/3)
are rendered as polylines via ``extract_lane_polylines``. Every other
valid roadgraph point id observed (0, 6, 7, 11, 12, 15, 16, 17, 18, 19,
20 in the inspected scene) is rendered generically as "other roadgraph
points" -- this module does not claim specific semantic labels
(road edge vs. crosswalk vs. stop sign, etc.) beyond what
``lane_geometry``/``run_scene.py``'s ``MAP_ELEMENT_TYPE_NAMES`` mapping
already documents, and the legend says so explicitly.
"""

import matplotlib

matplotlib.use("Agg")

import dataclasses
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.scenarios.lane_geometry import (
    ALL_LANE_TYPE_IDS,
    LanePolyline,
    extract_lane_polylines,
)
from src.scenarios.merge_detector import MergeDecision

DECISION_COLORS = {
    "accept": "#2ecc71",  # green
    "review": "#f1c40f",  # yellow
    "reject": "#e74c3c",  # red
}


def compute_obb_corners(
    x: float, y: float, yaw: float, length: float, width: float
) -> np.ndarray:
    """Computes the 4 corners of an oriented bounding box.

    Args:
        x, y: center position.
        yaw: heading (rad), 0 = +x axis.
        length: extent along the heading direction (full length, not
            half).
        width: extent perpendicular to heading (full width).

    Returns:
        (4, 2) array of corner (x, y) positions, in order: front-left,
        front-right, rear-right, rear-left.
    """

    half_l = length / 2.0
    half_w = width / 2.0

    # Corners in the box's own local frame (front = +local_x).
    local_corners = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ]
    )

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

    rotated = local_corners @ rotation.T
    return rotated + np.array([x, y])


def compute_viewport_bounds(
    center_x: float, center_y: float, radius_m: float
):
    """Returns (xlim, ylim) tuples symmetric around (center_x, center_y).

    Returns:
        (xmin, xmax), (ymin, ymax).
    """

    return (
        (center_x - radius_m, center_x + radius_m),
        (center_y - radius_m, center_y + radius_m),
    )


def _draw_obb(ax, x, y, yaw, length, width, facecolor, edgecolor="none", zorder=3, linewidth=1.0):
    corners = compute_obb_corners(x, y, yaw, length, width)
    polygon = plt.Polygon(
        corners,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(polygon)


def _draw_other_roadgraph_points(ax, roadgraph_points):
    """Renders non-lane roadgraph points (road edges/boundaries/
    crosswalks/etc. -- see module docstring) as small scattered points.
    """

    types = np.asarray(roadgraph_points.types)
    valid = np.asarray(roadgraph_points.valid).astype(bool)
    x = np.asarray(roadgraph_points.x)
    y = np.asarray(roadgraph_points.y)

    other_mask = valid & ~np.isin(types, list(ALL_LANE_TYPE_IDS))

    if np.any(other_mask):
        ax.scatter(
            x[other_mask],
            y[other_mask],
            s=1.5,
            c="#3a6ea5",
            alpha=0.5,
            zorder=1,
            label="other roadgraph points",
        )


def _draw_lane_polylines(ax, polylines: List[LanePolyline]):
    for polyline in polylines:
        ax.plot(
            polyline.xy[:, 0],
            polyline.xy[:, 1],
            color="#666666",
            linewidth=0.8,
            zorder=2,
        )


def render_candidate_figure(
    record,
    candidate,
    source_polyline: Optional[LanePolyline],
    target_polyline: Optional[LanePolyline],
    all_polylines: List[LanePolyline],
    front_features,
    output_path: Path,
    viewport_radius_m: float = 75.0,
):
    """Renders one validation PNG for a single merge candidate.

    Args:
        record: ``ScenarioRecord`` for the scene the candidate belongs
            to (used for roadgraph context + agent boxes + trajectory).
        candidate: ``dataset_builder.CandidateRecord`` for this row.
        source_polyline / target_polyline: reconstructed
            ``LanePolyline`` for the transition's source/target lane
            (may be None if unavailable -- e.g. REJECT_MISSING_GEOMETRY
            -- in which case only the fallback viewport center is used).
        all_polylines: every reconstructed vehicle-lane polyline in the
            scene (for gray/white context lines).
        front_features: ``InteractionFeatures`` for this candidate if
            decision == ACCEPT and features were computed, else None.
        output_path: PNG destination (parent dirs created if needed).
        viewport_radius_m: half-width/height of the rendered map view.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.82])
    ax.set_facecolor("black")
    ax.set_aspect("equal")

    state = record.state
    sdc_index = record.sdc_index
    log_trajectory = state.log_trajectory

    ego_x = np.asarray(log_trajectory.x[sdc_index])
    ego_y = np.asarray(log_trajectory.y[sdc_index])
    ego_yaw = np.asarray(log_trajectory.yaw[sdc_index])
    ego_valid = np.asarray(log_trajectory.valid[sdc_index]).astype(bool)
    ego_length = np.asarray(log_trajectory.length[sdc_index])
    ego_width = np.asarray(log_trajectory.width[sdc_index])

    frame = candidate.transition_frame

    # --- Roadgraph context --------------------------------------------
    if state.roadgraph_points is not None:
        _draw_other_roadgraph_points(ax, state.roadgraph_points)
    _draw_lane_polylines(ax, all_polylines)

    # --- Source / target highlighted polylines -------------------------
    if source_polyline is not None:
        ax.plot(
            source_polyline.xy[:, 0],
            source_polyline.xy[:, 1],
            color="yellow",
            linewidth=3.0,
            zorder=4,
            label=f"Source {candidate.source_lane_id}",
        )
        mid = source_polyline.xy[len(source_polyline.xy) // 2]
        ax.annotate(
            f"Source {candidate.source_lane_id}",
            xy=mid,
            color="yellow",
            fontsize=8,
            zorder=6,
        )

    if target_polyline is not None:
        ax.plot(
            target_polyline.xy[:, 0],
            target_polyline.xy[:, 1],
            color="orange",
            linewidth=3.0,
            zorder=4,
            label=f"Target {candidate.target_lane_id}",
        )
        mid = target_polyline.xy[len(target_polyline.xy) // 2]
        ax.annotate(
            f"Target {candidate.target_lane_id}",
            xy=mid,
            color="orange",
            fontsize=8,
            zorder=6,
        )

    # --- Merge start/end markers (if computable) ------------------------
    if (
        source_polyline is not None
        and candidate.merge_start_s is not None
        and candidate.merge_end_s is not None
    ):
        from src.scenarios.merge_detector import _interpolate_xy

        start_xy = _interpolate_xy(source_polyline, candidate.merge_start_s)
        end_xy = _interpolate_xy(source_polyline, candidate.merge_end_s)

        ax.scatter(
            [start_xy[0]], [start_xy[1]],
            marker="^", s=60, c="white", edgecolors="black",
            zorder=7, label="Merge Start (inferred)",
        )
        ax.scatter(
            [end_xy[0]], [end_xy[1]],
            marker="v", s=60, c="white", edgecolors="black",
            zorder=7, label="Merge End (source endpoint)",
        )

    # --- All other agents at the transition frame -----------------------
    object_ids = np.asarray(state.object_metadata.ids)
    valid = np.asarray(log_trajectory.valid[:, frame]).astype(bool)
    x = np.asarray(log_trajectory.x[:, frame])
    y = np.asarray(log_trajectory.y[:, frame])
    yaw = np.asarray(log_trajectory.yaw[:, frame])
    length = np.asarray(log_trajectory.length[:, frame])
    width = np.asarray(log_trajectory.width[:, frame])

    highlight_ids = set()
    if candidate.front_vehicle_id is not None:
        highlight_ids.add(int(candidate.front_vehicle_id))
    if candidate.rear_vehicle_id is not None:
        highlight_ids.add(int(candidate.rear_vehicle_id))

    for i in range(object_ids.shape[0]):
        if not valid[i]:
            continue
        obj_id = int(object_ids[i])
        if obj_id == record.sdc_id:
            continue
        if obj_id in highlight_ids:
            continue
        _draw_obb(
            ax, float(x[i]), float(y[i]), float(yaw[i]),
            float(length[i]), float(width[i]),
            facecolor="#ff66cc", zorder=3,
        )

    # Front/Rear highlight boxes.
    for obj_id, label in (
        (candidate.front_vehicle_id, "F"),
        (candidate.rear_vehicle_id, "R"),
    ):
        if obj_id is None:
            continue
        matches = np.flatnonzero(object_ids == int(obj_id))
        if matches.size == 0 or not valid[int(matches[0])]:
            continue
        idx = int(matches[0])
        _draw_obb(
            ax, float(x[idx]), float(y[idx]), float(yaw[idx]),
            float(length[idx]), float(width[idx]),
            facecolor="#ff66cc", edgecolor="white", linewidth=1.5, zorder=5,
        )
        ax.annotate(
            label, xy=(float(x[idx]), float(y[idx])),
            color="white", fontsize=10, fontweight="bold", zorder=8,
        )

    # --- Ego logged trajectory (full, all valid frames) -----------------
    if np.any(ego_valid):
        ax.plot(
            ego_x[ego_valid], ego_y[ego_valid],
            linestyle=(0, (3, 2)), color="lime", linewidth=1.2, zorder=4,
            label="Ego logged trajectory",
        )

    # --- Ego OBB at transition frame -------------------------------------
    if frame < ego_valid.shape[0] and ego_valid[frame]:
        _draw_obb(
            ax, float(ego_x[frame]), float(ego_y[frame]),
            float(ego_yaw[frame]), float(ego_length[frame]),
            float(ego_width[frame]), facecolor="#00ff00", zorder=6,
        )
        center_x, center_y = float(ego_x[frame]), float(ego_y[frame])
    elif source_polyline is not None:
        center_x, center_y = source_polyline.xy[-1]
    else:
        center_x, center_y = float(np.nanmean(ego_x)), float(np.nanmean(ego_y))

    # --- Viewport --------------------------------------------------------
    (xmin, xmax), (ymin, ymax) = compute_viewport_bounds(
        center_x, center_y, viewport_radius_m
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#444444")

    # --- Decision banner ---------------------------------------------------
    decision_str = candidate.decision
    color = DECISION_COLORS.get(decision_str, "white")
    banner = f"Decision: {decision_str.upper()}"
    if candidate.reason:
        banner += f"\nReason: {candidate.reason}"
    fig.text(
        0.5, 0.97, banner, color=color, fontsize=16, fontweight="bold",
        ha="center", va="top",
    )

    # --- Diagnostic info panel --------------------------------------------
    lines = [
        f"scene_key: {candidate.scene_key}",
        f"candidate_id: {candidate.candidate_id}",
        f"transition_frame: {candidate.transition_frame}",
        f"{candidate.source_lane_id} -> {candidate.target_lane_id}",
        f"decision: {candidate.decision}  reason: {candidate.reason}",
        f"endpoint_target_distance_m: {candidate.endpoint_target_distance_m}",
        f"heading_difference_deg: {candidate.heading_difference_deg}",
        f"separation_reduction_m: {candidate.separation_reduction_m}",
        f"decreasing_fraction: {candidate.decreasing_fraction}",
        f"source_remaining_distance_m: {candidate.source_remaining_distance_m}",
        f"target_lane_persistent: {candidate.target_lane_persistent}",
    ]
    if candidate.merge_start_s is not None:
        lines.append(
            f"merge_start_s/end_s: {candidate.merge_start_s:.2f} / "
            f"{candidate.merge_end_s:.2f}"
        )
    if candidate.front_vehicle_id is not None or candidate.rear_vehicle_id is not None:
        lines.append(
            f"front: id={candidate.front_vehicle_id} "
            f"gap={candidate.front_gap_m} ttc={candidate.front_ttc_s}"
        )
        lines.append(
            f"rear: id={candidate.rear_vehicle_id} "
            f"gap={candidate.rear_gap_m} ttc={candidate.rear_ttc_s}"
        )

    fig.text(
        0.01, 0.86, "\n".join(lines),
        color="white", fontsize=6.5, family="monospace",
        ha="left", va="top",
    )

    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
