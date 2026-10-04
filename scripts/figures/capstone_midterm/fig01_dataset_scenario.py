#!/usr/bin/env python3
"""Capstone midterm Fig 1: representative WOMD MERGE scenario (top view).

Real data only: resets the actual MergeEnvironment on one real,
already-verified maneuver (one of the 64 canonical Fixed-TRAIN64 IDs
used throughout this project's own diagnostics,
outputs/diagnostics/train_diag64_seed20260928.txt), reads the real
Waymax scene state at the maneuver's own merge_start_frame, and plots:
  - main (source) lane centerline
  - merge (target) lane centerline
  - ego vehicle
  - the real target-lane Front/Rear vehicles, identified via the
    SAME src.scenarios.scenario_features.find_target_lane_front_rear
    function the production 14D observation builder uses (never a
    manual/approximate reselection).

No synthetic trajectories, no placeholder vehicles.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.environment.full_split_evaluator import load_decision_dataset_maneuver_specs
from src.environment.merge_environment import MergeEnvironment
from src.scenarios.scenario_features import find_target_lane_front_rear
from src.visualization import style as _style

MANEUVER_ID = "training_tfexample.tfrecord-00000-of-01000#378__t18__237_456"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                           "outputs", "capstone_midterm_figures"))
OUTPUT_BASENAME = "fig01_dataset_scenario"

COLOR_MAIN_LANE = "#808080"
COLOR_MERGE_LANE = _style.COLOR_TARGET_LANE
COLOR_EGO = _style.COLOR_EGO
COLOR_FRONT = _style.COLOR_TARGET_FRONT
COLOR_REAR = _style.COLOR_TARGET_REAR


def main():
    env = MergeEnvironment(dataset_config_path="configs/dataset.yaml", downstream_mode="frenet_mpc")
    specs = {s.maneuver_id: s for s in load_decision_dataset_maneuver_specs("train")}
    spec = specs[MANEUVER_ID]

    observation, info = env.reset(spec)

    source_lane_id = info["active_source_lane_id"]
    target_lane_id = info["active_target_lane_id"]
    source_poly = env._polylines_by_id[source_lane_id]
    target_poly = env._polylines_by_id[target_lane_id]

    ego_x, ego_y, ego_yaw, _ = env._current_ego_pose_and_speed()

    # Real target-lane Front/Rear selection -- identical function the
    # production observation builder calls (see module docstring).
    traj = env._state.current_sim_trajectory
    object_ids = np.asarray(env._state.object_metadata.ids)
    object_types = np.asarray(env._state.object_metadata.object_types)
    x = np.asarray(traj.x)[:, 0]
    y = np.asarray(traj.y)[:, 0]
    yaw = np.asarray(traj.yaw)[:, 0]
    length = np.asarray(traj.length)[:, 0]
    width = np.asarray(traj.width)[:, 0]
    valid = np.asarray(traj.valid)[:, 0].astype(bool)

    from src.scenarios.lane_geometry import project_point_to_polyline_signed
    ego_s = project_point_to_polyline_signed(target_poly, ego_x, ego_y)["arc_length_m"]

    front_id, _front_s, rear_id, _rear_s = find_target_lane_front_rear(
        target_polyline=target_poly, ego_s_m=ego_s, ego_id=env._sdc_id, frame_index=0,
        object_ids=object_ids, object_types=object_types, valid=valid,
        x=x, y=y, yaw=yaw, config=env._agent_selection_config,
    )

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        "font.size": 11, "font.family": "DejaVu Sans",
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    ax.plot(source_poly.xy[:, 0], source_poly.xy[:, 1], color=COLOR_MAIN_LANE,
             linewidth=2.2, label="Main lane", zorder=1)
    ax.plot(target_poly.xy[:, 0], target_poly.xy[:, 1], color=COLOR_MERGE_LANE,
             linewidth=2.2, linestyle="-.", label="Merge lane", zorder=1)

    _style.draw_oriented_box(ax, ego_x, ego_y, ego_yaw, length=5.0, width=2.0,
                              color=COLOR_EGO, zorder=5, label="Ego")

    for agent_id, color, label in ((front_id, COLOR_FRONT, "Front"), (rear_id, COLOR_REAR, "Rear")):
        if agent_id is None:
            continue
        idx = int(np.where(object_ids == agent_id)[0][0])
        _style.draw_oriented_box(
            ax, float(x[idx]), float(y[idx]), float(yaw[idx]),
            length=float(length[idx]), width=float(width[idx]),
            color=color, zorder=4, label=label,
        )

    half_window = 70.0
    ax.set_xlim(ego_x - half_window * 0.3, ego_x + half_window)
    ax.set_ylim(ego_y - half_window * 0.55, ego_y + half_window * 0.55)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, handlelength=1.5)

    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME)
    fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"maneuver_id: {MANEUVER_ID}")
    print(f"source_lane_id: {source_lane_id}, target_lane_id: {target_lane_id}")
    print(f"front_id: {front_id}, rear_id: {rear_id}")
    print(f"Wrote {out_path}.png / .pdf")


if __name__ == "__main__":
    main()
