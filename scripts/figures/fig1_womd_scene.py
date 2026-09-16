"""Figure 1 -- real WOMD merge scene (clean + annotated).

EXPERIMENTAL figure (real data only). Loads maneuver MAN_0041 (default;
--maneuver overrides, e.g. MAN_0001 fallback) via the real, frozen
``MergeEnvironment.reset()`` at its own production ``decision_start_frame``
-- no synthetic geometry, no hand-placed agents. Draws:
  - every real lane centerline polyline within a padded bounding box
    around ego (context lanes in gray),
  - the maneuver's own active source lane (blue) / active target lane
    (orange), from ``EpisodeContext``,
  - ego as an oriented rectangle using its REAL length/width/yaw
    (Waymax ``current_sim_trajectory``), never a bare dot,
  - real nearby agents (green boxes),
  - (annotated variant only) Target Front / Target Rear / Source Front
    labels, computed by DIRECTLY calling the real, frozen
    ``src.scenarios.scenario_features.find_target_lane_front_rear``
    selection function against the same reset-frame state -- not
    guessed, not a different selection rule than production uses for
    the target side. Source Front is found using the SAME function
    called against the source polyline (a legitimate reuse: the
    function's docstring generalizes to "front/rear on a given lane",
    it is only ever invoked against the target lane in production
    because that's the only side MergeEnvironment's observation needs,
    but the underlying geometry check is lane-agnostic).

Outputs (paper + ppt variants, PNG+PDF):
  outputs/paper_ppt_figures/phase3/paper/fig1_womd_scene_clean_paper.{png,pdf}
  outputs/paper_ppt_figures/phase3/paper/fig1_womd_scene_annotated_paper.{png,pdf}
  outputs/paper_ppt_figures/phase3/ppt/fig1_womd_scene_clean_ppt.png
  outputs/paper_ppt_figures/phase3/ppt/fig1_womd_scene_annotated_ppt.png
Data: outputs/paper_ppt_figures/phase3/data/fig1_MAN_0041_scene.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from scripts.figures import _style
from scripts.figures._rollout import build_env, load_spec_by_id
from src.scenarios.scenario_features import (
    find_target_lane_front_rear,
    load_agent_selection_config,
)
from src.scenarios.lane_geometry import project_point_to_polyline_signed

OUT_PAPER = "outputs/paper_ppt_figures/phase3/paper"
OUT_PPT = "outputs/paper_ppt_figures/phase3/ppt"
OUT_DATA = "outputs/paper_ppt_figures/phase3/data"
MERGE_CONFIG_PATH = "configs/phase1_merge.yaml"


def gather_scene(maneuver_id: str, pad_m: float = 60.0):
    env = build_env(downstream_mode="legacy")  # geometry-only figure; legacy is sufficient/cheaper
    spec, split = load_spec_by_id(maneuver_id)
    observation, info = env.reset(spec)

    ego_x, ego_y, ego_yaw, ego_len, ego_wid, ego_speed = _ego(env)
    source_id = info["active_source_lane_id"]
    target_id = info["active_target_lane_id"]

    # Nearby agents (current frame, real Waymax state).
    from scripts.figures._rollout import _snapshot_agents
    agents = _snapshot_agents(env)
    nearby = [a for a in agents if np.hypot(a.x - ego_x, a.y - ego_y) <= pad_m]

    # Real lane polylines within the padded box.
    context_lanes = []
    for lane_id, poly in env._polylines_by_id.items():
        if np.min(np.hypot(poly.xy[:, 0] - ego_x, poly.xy[:, 1] - ego_y)) <= pad_m:
            context_lanes.append((lane_id, poly.xy.copy()))

    source_poly = env._polylines_by_id[source_id]
    target_poly = env._polylines_by_id[target_id]

    # Real selection-function calls (annotated variant): source/target front+rear.
    agent_selection_config = load_agent_selection_config(MERGE_CONFIG_PATH)
    traj = env._state.current_sim_trajectory
    x_all = np.asarray(traj.x)[:, 0]
    y_all = np.asarray(traj.y)[:, 0]
    yaw_all = np.asarray(traj.yaw)[:, 0]
    valid_all = np.asarray(traj.valid)[:, 0].astype(bool)
    object_ids = np.asarray(env._state.object_metadata.ids)
    object_types = np.asarray(env._state.object_metadata.object_types)

    ego_s_target = project_point_to_polyline_signed(target_poly, ego_x, ego_y)["arc_length_m"]
    t_front_id, t_front_s, t_rear_id, t_rear_s = find_target_lane_front_rear(
        target_poly, ego_s_target, env._sdc_id, 0,
        object_ids, object_types, valid_all, x_all, y_all, yaw_all,
        agent_selection_config,
    )
    ego_s_source = project_point_to_polyline_signed(source_poly, ego_x, ego_y)["arc_length_m"]
    s_front_id, s_front_s, s_rear_id, s_rear_s = find_target_lane_front_rear(
        source_poly, ego_s_source, env._sdc_id, 0,
        object_ids, object_types, valid_all, x_all, y_all, yaw_all,
        agent_selection_config,
    )

    nearby_dicts = [dataclasses_to_dict(a) for a in nearby]
    scene = {
        "maneuver_id": maneuver_id,
        "split": split,
        "decision_start_frame": info["decision_start_frame"],
        "merge_start_frame": info["merge_start_frame"],
        "ego": {"x": ego_x, "y": ego_y, "yaw": ego_yaw, "length": ego_len, "width": ego_wid, "speed_mps": ego_speed},
        "active_source_lane_id": int(source_id),
        "active_target_lane_id": int(target_id),
        "target_front_id": t_front_id, "target_rear_id": t_rear_id,
        "source_front_id": s_front_id,
        "agents": nearby_dicts,
    }
    return scene, context_lanes, source_poly, target_poly, nearby_dicts, agent_selection_config


def dataclasses_to_dict(a):
    return {"agent_id": a.agent_id, "x": a.x, "y": a.y, "yaw": a.yaw,
            "length": a.length, "width": a.width}


def _ego(env):
    traj = env._state.current_sim_trajectory
    i = env._sdc_index
    return (
        float(np.asarray(traj.x)[i, 0]), float(np.asarray(traj.y)[i, 0]),
        float(np.asarray(traj.yaw)[i, 0]), float(np.asarray(traj.length)[i, 0]),
        float(np.asarray(traj.width)[i, 0]),
        float(np.hypot(np.asarray(traj.vel_x)[i, 0], np.asarray(traj.vel_y)[i, 0])),
    )


def render(scene, context_lanes, source_poly, target_poly, nearby, annotated: bool, ppt: bool):
    import matplotlib.pyplot as plt

    if ppt:
        _style.apply_ppt_style()
    else:
        _style.apply_paper_style()

    fig, ax = plt.subplots(figsize=(7.5, 7.5) if not ppt else (10, 10))

    for lane_id, xy in context_lanes:
        if lane_id in (scene["active_source_lane_id"], scene["active_target_lane_id"]):
            continue
        ax.plot(xy[:, 0], xy[:, 1], color=_style.COLOR_OTHER_LANE, linewidth=1.0, zorder=1)

    ax.plot(source_poly.xy[:, 0], source_poly.xy[:, 1], color=_style.COLOR_SOURCE_LANE,
            linewidth=2.2, zorder=2, label="Active source lane")
    ax.plot(target_poly.xy[:, 0], target_poly.xy[:, 1], color=_style.COLOR_TARGET_LANE,
            linewidth=2.2, zorder=2, label="Active target lane")

    for a in nearby:
        is_tf = annotated and a["agent_id"] == scene["target_front_id"]
        is_tr = annotated and a["agent_id"] == scene["target_rear_id"]
        is_sf = annotated and a["agent_id"] == scene["source_front_id"]
        color = _style.COLOR_SURROUNDING_AGENT
        if is_tf:
            color = _style.COLOR_TARGET_FRONT
        elif is_tr:
            color = _style.COLOR_TARGET_REAR
        elif is_sf:
            color = _style.COLOR_SOURCE_FRONT
        _style.draw_oriented_box(ax, a["x"], a["y"], a["yaw"], a["length"], a["width"], color, zorder=4)
        if annotated and (is_tf or is_tr or is_sf):
            label = "Target Front" if is_tf else ("Target Rear" if is_tr else "Source Front")
            ax.annotate(label, (a["x"], a["y"]), textcoords="offset points",
                        xytext=(8, 8), fontsize=(13 if ppt else 7), color=color, weight="bold")

    ego = scene["ego"]
    _style.draw_oriented_box(ax, ego["x"], ego["y"], ego["yaw"], ego["length"], ego["width"],
                              _style.COLOR_EGO, zorder=6, label="Ego")
    if annotated:
        ax.annotate("Ego", (ego["x"], ego["y"]), textcoords="offset points",
                    xytext=(8, -14), fontsize=(13 if ppt else 7), weight="bold")

    ax.set_aspect("equal")
    ax.set_xlabel("X (m, WOMD scene frame)")
    ax.set_ylabel("Y (m, WOMD scene frame)")
    pad = 62
    ax.set_xlim(ego["x"] - pad, ego["x"] + pad)
    ax.set_ylim(ego["y"] - pad, ego["y"] + pad)
    if not ppt:
        ax.set_title(f"{scene['maneuver_id']} ({scene['split']}) -- real WOMD merge scene"
                      + (" (annotated)" if annotated else ""))
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maneuver", default="MAN_0041")
    args = parser.parse_args()

    os.makedirs(OUT_PAPER, exist_ok=True)
    os.makedirs(OUT_PPT, exist_ok=True)
    os.makedirs(OUT_DATA, exist_ok=True)

    print(f"[fig1] gathering real scene for {args.maneuver} ...")
    scene, context_lanes, source_poly, target_poly, nearby, _ = gather_scene(args.maneuver)
    print(f"[fig1] maneuver={scene['maneuver_id']} split={scene['split']} "
          f"source_lane={scene['active_source_lane_id']} target_lane={scene['active_target_lane_id']} "
          f"n_nearby_agents={len(nearby)} target_front={scene['target_front_id']} "
          f"target_rear={scene['target_rear_id']} source_front={scene['source_front_id']}")

    with open(os.path.join(OUT_DATA, f"fig1_{args.maneuver}_scene.json"), "w") as f:
        json.dump(scene, f, indent=2, default=str)

    for annotated in (False, True):
        suffix = "annotated" if annotated else "clean"
        fig = render(scene, context_lanes, source_poly, target_poly, nearby, annotated, ppt=False)
        _style.savefig_paper(fig, os.path.join(OUT_PAPER, f"fig1_womd_scene_{suffix}_paper"))
        import matplotlib.pyplot as plt
        plt.close(fig)

        fig = render(scene, context_lanes, source_poly, target_poly, nearby, annotated, ppt=True)
        _style.savefig_ppt(fig, os.path.join(OUT_PPT, f"fig1_womd_scene_{suffix}_ppt"))
        plt.close(fig)

    print("[fig1] done.")


if __name__ == "__main__":
    main()
