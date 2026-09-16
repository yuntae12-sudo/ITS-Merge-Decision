"""Failure/limitation figure -- MAN_0071 (Phase 3 downstream limitation example).

EXPERIMENTAL figure, explicitly a PHASE 3 DOWNSTREAM LIMITATION
EXAMPLE, NOT a final FSM/PPO comparison result. Reproduces (does not
merely cite) Stage 3-H's documented finding for MAN_0071
(lane_chain 102->129->119): at the reset frame of the SECOND
transition (source lane 102 -> intermediate target lane 129), ego's
real Frenet lateral offset against lane 129, computed via the SAME
production wrapper the planner itself uses
(``src.planning.candidate_generator.project_cartesian_to_frame``), is
recomputed here directly from a real ``MergeEnvironment.reset()`` call
-- NOT copied from the doc. An independent Euclidean-distance check
against lane 129's own real polyline is also recomputed and must agree
with the Frenet ``d`` to confirm this is a genuine large real lateral
gap (not a Stage 3-G-style extrapolation-branch artifact).

This maneuver's lane_chain has 3 lanes (102, 129, 119); the FIRST
transition (102->129) is the one MergeEnvironment's ``reset()``
actually activates first (``active_transition_index=0``), so this
figure geometrically illustrates the 102->129 transition -- matching
Stage 3-H's own finding location ("this transition becomes active").
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from scripts.figures import _style
from scripts.figures._rollout import build_env, load_spec_by_id, run_rollout
from src.planning.candidate_generator import project_cartesian_to_frame
from src.planning.reference import ReferenceLine

OUT_PAPER = "outputs/paper_ppt_figures/phase3/paper"
OUT_PPT = "outputs/paper_ppt_figures/phase3/ppt"
OUT_DATA = "outputs/paper_ppt_figures/phase3/data"
MANEUVER_ID = "MAN_0071"
DOCUMENTED_D = -19.49


def main():
    os.makedirs(OUT_PAPER, exist_ok=True)
    os.makedirs(OUT_PPT, exist_ok=True)
    os.makedirs(OUT_DATA, exist_ok=True)

    env = build_env(downstream_mode="frenet_mpc")
    spec, split = load_spec_by_id(MANEUVER_ID)
    observation, info = env.reset(spec)
    print(f"[fig_MAN_0071] maneuver={MANEUVER_ID} split={split} lane_chain={spec.lane_chain}")

    source_id = info["active_source_lane_id"]
    target_id = info["active_target_lane_id"]
    print(f"[fig_MAN_0071] active_source_lane_id={source_id} active_target_lane_id={target_id}")

    traj = env._state.current_sim_trajectory
    i = env._sdc_index
    ego_x = float(np.asarray(traj.x)[i, 0])
    ego_y = float(np.asarray(traj.y)[i, 0])
    ego_yaw = float(np.asarray(traj.yaw)[i, 0])
    vx = float(np.asarray(traj.vel_x)[i, 0])
    vy = float(np.asarray(traj.vel_y)[i, 0])
    ego_speed = float(np.hypot(vx, vy))
    ego_len = float(np.asarray(traj.length)[i, 0])
    ego_wid = float(np.asarray(traj.width)[i, 0])

    target_poly = env._polylines_by_id[target_id]
    source_poly = env._polylines_by_id[source_id]
    target_ref = ReferenceLine.from_lane_polyline(target_poly)

    fs = project_cartesian_to_frame(ego_x, ego_y, ego_yaw, ego_speed, target_ref)
    recomputed_d = float(fs.d)

    dists = np.hypot(target_poly.xy[:, 0] - ego_x, target_poly.xy[:, 1] - ego_y)
    min_dist = float(dists.min())
    nearest_idx = int(dists.argmin())

    print(f"[fig_MAN_0071] recomputed Frenet d against target lane {target_id}: {recomputed_d:.4f}")
    print(f"[fig_MAN_0071] independent Euclidean min-distance check: {min_dist:.4f} "
          f"at polyline index {nearest_idx} of {len(target_poly.xy)} (interior={0 < nearest_idx < len(target_poly.xy)-1})")
    agree = abs(abs(recomputed_d) - min_dist) < 0.01
    print(f"[fig_MAN_0071] Frenet |d| vs Euclidean min-distance agreement (<0.01m): {agree}")
    if not agree:
        print("[fig_MAN_0071] DISCREPANCY: Frenet lateral offset and independent Euclidean check disagree; "
              "reporting both numbers as-is, not forcing a match.")
    close_to_documented = abs(recomputed_d - DOCUMENTED_D) < 0.01
    print(f"[fig_MAN_0071] matches documented Stage 3-H value ({DOCUMENTED_D}): {close_to_documented}")

    # Run the real rollout to also show the eventual failure_collision outcome.
    spec2, split2, snaps, env2 = run_rollout(MANEUVER_ID, max_steps=25, downstream_mode="frenet_mpc")
    terminal = snaps[-1]
    print(f"[fig_MAN_0071] real rollout terminal: step={terminal.step_index} "
          f"reason={terminal.termination_reason}")

    meta = {
        "maneuver_id": MANEUVER_ID, "split": split, "lane_chain": spec.lane_chain,
        "active_source_lane_id": int(source_id), "active_target_lane_id": int(target_id),
        "recomputed_frenet_d_m": recomputed_d,
        "independent_euclidean_min_distance_m": min_dist,
        "nearest_polyline_index": nearest_idx,
        "target_lane_polyline_point_count": len(target_poly.xy),
        "documented_value_m": DOCUMENTED_D,
        "matches_documented": close_to_documented,
        "rollout_terminal_step": terminal.step_index,
        "rollout_termination_reason": terminal.termination_reason,
    }
    with open(os.path.join(OUT_DATA, f"fig_failure_{MANEUVER_ID}.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[fig_MAN_0071] wrote data file.")

    import matplotlib.pyplot as plt

    for ppt in (False, True):
        if ppt:
            _style.apply_ppt_style()
        else:
            _style.apply_paper_style()

        fig, ax = plt.subplots(figsize=(7, 6) if not ppt else (10, 8.5))
        ax.plot(source_poly.xy[:, 0], source_poly.xy[:, 1], color=_style.COLOR_SOURCE_LANE,
                linewidth=2.2, label=f"Source lane {source_id}", zorder=1)
        ax.plot(target_poly.xy[:, 0], target_poly.xy[:, 1], color=_style.COLOR_TARGET_LANE,
                linewidth=2.2, label=f"Intermediate target lane {target_id}", zorder=1)

        _style.draw_oriented_box(ax, ego_x, ego_y, ego_yaw, ego_len, ego_wid,
                                  _style.COLOR_EGO, zorder=5, label="Ego (reset frame)")

        nearest_pt = target_poly.xy[nearest_idx]
        ax.plot([ego_x, nearest_pt[0]], [ego_y, nearest_pt[1]], color="red", linewidth=1.5,
                linestyle=":", zorder=4)
        mid = ((ego_x + nearest_pt[0]) / 2, (ego_y + nearest_pt[1]) / 2)
        ax.annotate(f"{min_dist:.2f} m", mid, fontsize=(13 if ppt else 9), color="red", weight="bold")

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        if not ppt:
            ax.set_title(f"{MANEUVER_ID} -- Phase 3 downstream limitation example (NOT FSM/PPO result)\n"
                          f"Large real lateral gap to intermediate target lane at transition start "
                          f"(d={recomputed_d:.2f} m); episode ends {terminal.termination_reason}", fontsize=8)
        ax.legend(loc="best", fontsize=(11 if ppt else 8))
        fig.tight_layout()

        if ppt:
            _style.savefig_ppt(fig, os.path.join(OUT_PPT, f"fig_failure_{MANEUVER_ID}_ppt"))
        else:
            _style.savefig_paper(fig, os.path.join(OUT_PAPER, f"fig_failure_{MANEUVER_ID}_paper"))
        plt.close(fig)

    print("[fig_MAN_0071] done.")


if __name__ == "__main__":
    main()
