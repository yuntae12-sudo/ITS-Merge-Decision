"""Failure/limitation figure -- MAN_0107 (Phase 3 downstream limitation example).

EXPERIMENTAL figure, explicitly a PHASE 3 DOWNSTREAM LIMITATION
EXAMPLE, NOT a final FSM/PPO comparison result. Reproduces Stage 3-H's
documented MAN_0107 finding (lane_chain 126->120->119, 96%
PLANNER_INFEASIBLE across 71 audited steps, oscillating
curvature/longitudinal_accel failures, chain never advances -- ego's
lateral position (d) against the MERGE target frame oscillates
between roughly -2.7 and +2.9 while longitudinal speed (s_d)
repeatedly crashes toward near-zero) by DIRECTLY re-running the real
rollout and, at each real step, calling
``src.planning.candidate_generator.project_cartesian_to_frame`` against
the ACTIVE TARGET reference (the exact frame the MERGE candidate is
generated in -- see ``frenet_planner.py``'s MERGE dispatch) to recover
the real Frenet (d, s_d) time series -- not a re-citation of the
documented numbers.

Two panels:
  1. XY: real target-lane geometry + real executed ego trajectory.
  2. Frame vs. (Frenet d, speed s_d) line plot showing the real
     oscillation/near-stall pattern.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from scripts.figures import _style
from scripts.figures._rollout import build_env, load_spec_by_id
from src.environment.behavior_action import BehaviorAction
from src.planning.candidate_generator import project_cartesian_to_frame
from src.planning.reference import ReferenceLine

OUT_PAPER = "outputs/paper_ppt_figures/phase3/paper"
OUT_PPT = "outputs/paper_ppt_figures/phase3/ppt"
OUT_DATA = "outputs/paper_ppt_figures/phase3/data"
MANEUVER_ID = "MAN_0107"
MAX_STEPS = 80


def main():
    os.makedirs(OUT_PAPER, exist_ok=True)
    os.makedirs(OUT_PPT, exist_ok=True)
    os.makedirs(OUT_DATA, exist_ok=True)

    env = build_env(downstream_mode="frenet_mpc")
    spec, split = load_spec_by_id(MANEUVER_ID)
    observation, info = env.reset(spec)
    print(f"[fig_MAN_0107] maneuver={MANEUVER_ID} split={split} lane_chain={spec.lane_chain}")

    target_id0 = info["active_target_lane_id"]
    target_poly = env._polylines_by_id[target_id0]

    steps, ds, s_ds, xs, ys, statuses = [], [], [], [], [], []
    terminal_reason = None
    terminated = truncated = False

    for step in range(MAX_STEPS):
        traj = env._state.current_sim_trajectory
        i = env._sdc_index
        ego_x = float(np.asarray(traj.x)[i, 0])
        ego_y = float(np.asarray(traj.y)[i, 0])
        ego_yaw = float(np.asarray(traj.yaw)[i, 0])
        vx = float(np.asarray(traj.vel_x)[i, 0])
        vy = float(np.asarray(traj.vel_y)[i, 0])
        ego_speed = float(np.hypot(vx, vy))

        target_id = info["active_target_lane_id"]
        target_ref = ReferenceLine.from_lane_polyline(env._polylines_by_id[target_id])
        fs = project_cartesian_to_frame(ego_x, ego_y, ego_yaw, ego_speed, target_ref)

        steps.append(step)
        ds.append(float(fs.d))
        s_ds.append(float(fs.s_d))
        xs.append(ego_x)
        ys.append(ego_y)

        observation, reward, terminated, truncated, info = env.step(BehaviorAction.MERGE)
        statuses.append(info.get("downstream_status"))
        if terminated or truncated:
            terminal_reason = info.get("termination_reason")
            break

    print(f"[fig_MAN_0107] ran {len(steps)} steps, terminated={terminated} truncated={truncated} reason={terminal_reason}")
    print(f"[fig_MAN_0107] real Frenet d range: [{min(ds):.3f}, {max(ds):.3f}]  (documented: approx [-2.7, +2.9])")
    print(f"[fig_MAN_0107] real min s_d (longitudinal speed): {min(s_ds):.4f} m/s  (documented: approx 0.08 m/s)")

    n_infeasible = sum(1 for s in statuses if s == "PLANNER_INFEASIBLE")
    frac_infeasible = n_infeasible / len(statuses) if statuses else 0.0
    print(f"[fig_MAN_0107] PLANNER_INFEASIBLE fraction this run: {n_infeasible}/{len(statuses)} = {frac_infeasible:.2%} "
          f"(documented: 96% across 71 audited steps)")

    data_path = os.path.join(OUT_DATA, f"fig_failure_{MANEUVER_ID}_frenet_timeseries.csv")
    with open(data_path, "w") as f:
        f.write("step,ego_x,ego_y,frenet_d_target_frame,frenet_s_d_target_frame,downstream_status_after_step\n")
        for k in range(len(steps)):
            f.write(f"{steps[k]},{xs[k]},{ys[k]},{ds[k]},{s_ds[k]},{statuses[k]}\n")
    print(f"[fig_MAN_0107] wrote {data_path}")

    meta = {
        "maneuver_id": MANEUVER_ID, "split": split, "lane_chain": spec.lane_chain,
        "steps_run": len(steps), "terminated": bool(terminated), "truncated": bool(truncated),
        "termination_reason": terminal_reason,
        "frenet_d_min": min(ds), "frenet_d_max": max(ds), "frenet_s_d_min": min(s_ds),
        "planner_infeasible_fraction": frac_infeasible,
    }
    with open(os.path.join(OUT_DATA, f"fig_failure_{MANEUVER_ID}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    import matplotlib.pyplot as plt

    for ppt in (False, True):
        if ppt:
            _style.apply_ppt_style()
        else:
            _style.apply_paper_style()

        fig, (ax_xy, ax_ts) = plt.subplots(1, 2, figsize=(11, 4.5) if not ppt else (17, 7))

        ax_xy.plot(target_poly.xy[:, 0], target_poly.xy[:, 1], color=_style.COLOR_TARGET_LANE,
                   linewidth=2.2, label=f"Target lane {target_id0}", zorder=1)
        ax_xy.plot(xs, ys, color=_style.COLOR_EXECUTED_TRAJ, linewidth=1.8, linestyle="--",
                   label="Executed trajectory", zorder=3)
        ax_xy.scatter([xs[0]], [ys[0]], color="black", marker="x", s=50, zorder=4)
        ax_xy.set_aspect("equal")
        ax_xy.set_xlabel("X (m)")
        ax_xy.set_ylabel("Y (m)")
        ax_xy.legend(fontsize=(11 if ppt else 8))

        ax_d = ax_ts
        ax_d.plot(steps, ds, color=_style.COLOR_PLANNED_TRAJ, marker="o", markersize=3, label="Frenet d (target frame, m)")
        ax_d.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
        ax_d.set_xlabel("Rollout step")
        ax_d.set_ylabel("Frenet d (m)", color=_style.COLOR_PLANNED_TRAJ)
        ax_d.tick_params(axis="y", labelcolor=_style.COLOR_PLANNED_TRAJ)

        ax_speed = ax_d.twinx()
        ax_speed.plot(steps, s_ds, color=_style.COLOR_EXECUTED_TRAJ, marker="s", markersize=3, label="Frenet s_d (speed, m/s)")
        ax_speed.set_ylabel("Longitudinal speed s_d (m/s)", color=_style.COLOR_EXECUTED_TRAJ)
        ax_speed.tick_params(axis="y", labelcolor=_style.COLOR_EXECUTED_TRAJ)

        if not ppt:
            fig.suptitle(f"{MANEUVER_ID} -- Phase 3 downstream limitation example (NOT FSM/PPO result)\n"
                         f"Lateral (d) oscillation + longitudinal near-stall in the MERGE target frame; "
                         f"chain never advances, episode {terminal_reason}", fontsize=8)
        fig.tight_layout()

        if ppt:
            _style.savefig_ppt(fig, os.path.join(OUT_PPT, f"fig_failure_{MANEUVER_ID}_ppt"))
        else:
            _style.savefig_paper(fig, os.path.join(OUT_PAPER, f"fig_failure_{MANEUVER_ID}_paper"))
        plt.close(fig)

    print("[fig_MAN_0107] done.")


if __name__ == "__main__":
    main()
