"""Figure 2 -- real closed-loop rollout sequence (4-panel).

EXPERIMENTAL figure. Runs one real, deterministic
``MergeEnvironment(..., downstream_mode="frenet_mpc")`` rollout on
MAN_0041, repeating BehaviorAction.MERGE every step (matching Phase
3's own precedent script, ``scripts/audit_phase3_robustness.py``, and
``tests/environment/test_merge_environment*.py``'s own success-test
convention). This is explicitly a DOWNSTREAM QUALITATIVE
DEMONSTRATION of the frenet_mpc execution stack, NOT an FSM/PPO
decision-quality result -- captions must say so.

Four real moments are selected directly from the rollout's own info
dict (never manually chosen coordinates):
  1. decision start   -- step 0 (reset() frame)
  2. merge initiation -- the first step where info["chain_advanced"]
     is True (the source/target lane pair actually transitions)
  3. mid-merge         -- the temporal midpoint between merge
     initiation and the terminal step
  4. target-lane entry / success -- the terminal step
     (info["termination_reason"] == "success")

All four panels share ONE consistent map window (bounds computed from
the full real ego trajectory, padded), so panel-to-panel motion is
visually comparable. Each panel draws real roadgraph, the oriented ego
box at that frame, real surrounding vehicles at that frame, and the
ego's REAL trajectory history up to that frame (a polyline through the
actual recorded ego (x, y) samples, never smoothed/interpolated
by hand).
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from scripts.figures import _style
from scripts.figures._rollout import run_rollout

OUT_PAPER = "outputs/paper_ppt_figures/phase3/paper"
OUT_PPT = "outputs/paper_ppt_figures/phase3/ppt"
OUT_DATA = "outputs/paper_ppt_figures/phase3/data"
MANEUVER_ID = "MAN_0041"
MAX_STEPS = 90


def pick_moments(snaps):
    step0 = snaps[0]
    merge_init = next((s for s in snaps if s.info.get("chain_advanced")), None)
    terminal = snaps[-1]
    if merge_init is None:
        raise RuntimeError("No chain_advanced step found in this rollout -- cannot pick 'merge initiation' moment from real data.")
    mid_idx = (merge_init.step_index + terminal.step_index) // 2
    mid = next(s for s in snaps if s.step_index == mid_idx)
    return step0, merge_init, mid, terminal


def render(snaps, moments, ppt: bool):
    import matplotlib.pyplot as plt

    if ppt:
        _style.apply_ppt_style()
    else:
        _style.apply_paper_style()

    all_y = [s.ego_y for s in snaps]
    # Real trajectory is highway-like (long in x, short in y): ~75m of
    # ego travel across the whole rollout but well under 1m of net
    # lateral change. Each panel uses its OWN local window, centered on
    # that moment's real ego position, with a FIXED half-width/height
    # shared across all four panels (so panel-to-panel scale is
    # comparable) -- this is a display-window choice only; the plotted
    # ego/agent/lane x,y DATA is exactly the real recorded state at
    # that frame, never altered or rescaled.
    half_window_m = 22.0
    y_center_global = (min(all_y) + max(all_y)) / 2.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 6) if not ppt else (26, 8))
    titles = ["1. Decision start", "2. Merge initiation", "3. Mid-merge", "4. Target-lane entry (success)"]

    for ax, moment, title in zip(axes, moments, titles):
        env_polylines = moment_polylines[moment.step_index]
        for lane_id, xy in env_polylines:
            color = _style.COLOR_OTHER_LANE
            if lane_id == moment.active_source_lane_id:
                color = _style.COLOR_SOURCE_LANE
            elif lane_id == moment.active_target_lane_id:
                color = _style.COLOR_TARGET_LANE
            lw = 2.0 if lane_id in (moment.active_source_lane_id, moment.active_target_lane_id) else 1.0
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=lw, zorder=1)

        for a in moment.agents:
            _style.draw_oriented_box(ax, a.x, a.y, a.yaw, a.length, a.width,
                                      _style.COLOR_SURROUNDING_AGENT, zorder=3)

        hist_x = [s.ego_x for s in snaps if s.step_index <= moment.step_index]
        hist_y = [s.ego_y for s in snaps if s.step_index <= moment.step_index]
        ax.plot(hist_x, hist_y, color=_style.COLOR_EXECUTED_TRAJ, linewidth=1.6,
                linestyle="--", zorder=4, label="Ego history" if ax is axes[0] else None)

        _style.draw_oriented_box(ax, moment.ego_x, moment.ego_y, moment.ego_yaw,
                                  moment.ego_length, moment.ego_width,
                                  _style.COLOR_EGO, zorder=6)

        ax.set_xlim(moment.ego_x - half_window_m, moment.ego_x + half_window_m)
        ax.set_ylim(y_center_global - half_window_m, y_center_global + half_window_m)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nstep={moment.step_index}", fontsize=(15 if ppt else 9))
        ax.set_xlabel("X (m)")
        if ax is axes[0]:
            ax.set_ylabel("Y (m)")

    if not ppt:
        fig.suptitle(f"{MANEUVER_ID} -- real closed-loop frenet_mpc rollout (MERGE every step; "
                      "downstream qualitative demonstration, not an FSM/PPO result)", fontsize=10)
    fig.tight_layout()
    return fig


def main():
    os.makedirs(OUT_PAPER, exist_ok=True)
    os.makedirs(OUT_PPT, exist_ok=True)
    os.makedirs(OUT_DATA, exist_ok=True)

    print(f"[fig2] running real rollout for {MANEUVER_ID} (frenet_mpc, MERGE every step) ...")
    spec, split, snaps, env = run_rollout(MANEUVER_ID, max_steps=MAX_STEPS, downstream_mode="frenet_mpc")
    print(f"[fig2] maneuver={spec.maneuver_id} split={split} steps={len(snaps)-1} "
          f"terminated_at={snaps[-1].step_index} reason={snaps[-1].termination_reason}")

    moments = pick_moments(snaps)
    for m, label in zip(moments, ("decision_start", "merge_initiation", "mid_merge", "terminal")):
        print(f"[fig2]  moment {label}: step={m.step_index} x={m.ego_x:.2f} y={m.ego_y:.2f} "
              f"source={m.active_source_lane_id} target={m.active_target_lane_id}")

    # Re-derive per-moment lane polylines: need the *scene's* full lane
    # set, which is fixed for the whole episode (same scene loaded once
    # by reset()) -- reuse env._polylines_by_id (same object throughout
    # this single-episode rollout).
    global moment_polylines
    pad_m = 55.0
    moment_polylines = {}
    for m in moments:
        polys = []
        for lane_id, poly in env._polylines_by_id.items():
            if np.min(np.hypot(poly.xy[:, 0] - m.ego_x, poly.xy[:, 1] - m.ego_y)) <= pad_m:
                polys.append((lane_id, poly.xy.copy()))
        moment_polylines[m.step_index] = polys

    # Save exact plotted numerical data (full ego trajectory).
    data_path = os.path.join(OUT_DATA, f"fig2_rollout_{MANEUVER_ID}_ego_trajectory.csv")
    with open(data_path, "w") as f:
        f.write("step_index,ego_x,ego_y,ego_yaw,ego_speed_mps,active_source_lane_id,active_target_lane_id,downstream_status,intervention_rate,terminated,truncated,termination_reason\n")
        for s in snaps:
            f.write(f"{s.step_index},{s.ego_x},{s.ego_y},{s.ego_yaw},{s.ego_speed_mps},"
                    f"{s.active_source_lane_id},{s.active_target_lane_id},{s.downstream_status},"
                    f"{s.intervention_rate},{s.terminated},{s.truncated},{s.termination_reason}\n")
    print(f"[fig2] wrote {data_path}")

    moments_meta = [
        {"label": label, "step_index": m.step_index, "ego_x": m.ego_x, "ego_y": m.ego_y,
         "active_source_lane_id": m.active_source_lane_id, "active_target_lane_id": m.active_target_lane_id}
        for m, label in zip(moments, ("decision_start", "merge_initiation", "mid_merge", "terminal"))
    ]
    with open(os.path.join(OUT_DATA, f"fig2_rollout_{MANEUVER_ID}_moments.json"), "w") as f:
        json.dump({"maneuver_id": MANEUVER_ID, "split": split, "moments": moments_meta}, f, indent=2, default=str)

    import matplotlib.pyplot as plt
    fig = render(snaps, moments, ppt=False)
    _style.savefig_paper(fig, os.path.join(OUT_PAPER, f"fig2_rollout_sequence_{MANEUVER_ID}_paper"))
    plt.close(fig)

    fig = render(snaps, moments, ppt=True)
    _style.savefig_ppt(fig, os.path.join(OUT_PPT, f"fig2_rollout_sequence_{MANEUVER_ID}_ppt"))
    plt.close(fig)

    print("[fig2] done.")


if __name__ == "__main__":
    main()
