"""Optional diagnostic figure -- legacy vs. frenet_mpc downstream comparison.

EXPERIMENTAL, LOWER-PRIORITY figure. Explicitly labeled "diagnostic
downstream comparison" -- this is NOT an "FSM vs PPO" result of any
kind; both runs use the identical fixed action sequence (MERGE every
step) and only differ in which Stage 3 downstream execution stack
(``downstream_mode="legacy"`` vs ``"frenet_mpc"``) converts that
high-level decision into a physical command.

Runs MAN_0041 twice, once per downstream_mode, from two INDEPENDENT
``MergeEnvironment`` instances (matching Stage 3-E's own determinism-
test pattern: no shared state between runs), and overlays both real
executed ego trajectories on the same real roadgraph.
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


def main():
    os.makedirs(OUT_PAPER, exist_ok=True)
    os.makedirs(OUT_PPT, exist_ok=True)
    os.makedirs(OUT_DATA, exist_ok=True)

    print(f"[fig_legacy_vs_frenet_mpc] running {MANEUVER_ID} under legacy ...")
    spec_l, split_l, snaps_l, env_l = run_rollout(MANEUVER_ID, max_steps=MAX_STEPS, downstream_mode="legacy")
    print(f"[fig_legacy_vs_frenet_mpc] legacy: steps={len(snaps_l)-1} terminal={snaps_l[-1].termination_reason}")

    print(f"[fig_legacy_vs_frenet_mpc] running {MANEUVER_ID} under frenet_mpc ...")
    spec_f, split_f, snaps_f, env_f = run_rollout(MANEUVER_ID, max_steps=MAX_STEPS, downstream_mode="frenet_mpc")
    print(f"[fig_legacy_vs_frenet_mpc] frenet_mpc: steps={len(snaps_f)-1} terminal={snaps_f[-1].termination_reason}")

    xs_l = [s.ego_x for s in snaps_l]
    ys_l = [s.ego_y for s in snaps_l]
    xs_f = [s.ego_x for s in snaps_f]
    ys_f = [s.ego_y for s in snaps_f]

    data_path = os.path.join(OUT_DATA, f"fig_legacy_vs_frenet_mpc_{MANEUVER_ID}.csv")
    with open(data_path, "w") as f:
        f.write("mode,step_index,ego_x,ego_y\n")
        for s in snaps_l:
            f.write(f"legacy,{s.step_index},{s.ego_x},{s.ego_y}\n")
        for s in snaps_f:
            f.write(f"frenet_mpc,{s.step_index},{s.ego_x},{s.ego_y}\n")
    print(f"[fig_legacy_vs_frenet_mpc] wrote {data_path}")

    meta = {
        "maneuver_id": MANEUVER_ID,
        "legacy": {"steps": len(snaps_l) - 1, "termination_reason": snaps_l[-1].termination_reason},
        "frenet_mpc": {"steps": len(snaps_f) - 1, "termination_reason": snaps_f[-1].termination_reason},
    }
    with open(os.path.join(OUT_DATA, f"fig_legacy_vs_frenet_mpc_{MANEUVER_ID}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # Real roadgraph, from the frenet_mpc env instance (same scene both runs).
    ego_x0, ego_y0 = xs_l[0], ys_l[0]
    pad_m = 55.0
    context_lanes = []
    for lane_id, poly in env_f._polylines_by_id.items():
        if np.min(np.hypot(poly.xy[:, 0] - ego_x0, poly.xy[:, 1] - ego_y0)) <= pad_m:
            context_lanes.append(poly.xy.copy())

    import matplotlib.pyplot as plt

    for ppt in (False, True):
        if ppt:
            _style.apply_ppt_style()
        else:
            _style.apply_paper_style()

        fig, ax = plt.subplots(figsize=(8, 6) if not ppt else (11, 8.5))
        for xy in context_lanes:
            ax.plot(xy[:, 0], xy[:, 1], color=_style.COLOR_OTHER_LANE, linewidth=1.0, zorder=1)

        ax.plot(xs_l, ys_l, color=_style.COLOR_COLLISION, linewidth=2.2, linestyle="-",
                label=f"legacy downstream (terminal: {snaps_l[-1].termination_reason})", zorder=3)
        ax.plot(xs_f, ys_f, color=_style.COLOR_SUCCESS, linewidth=2.2, linestyle="--",
                label=f"frenet_mpc downstream (terminal: {snaps_f[-1].termination_reason})", zorder=4)
        ax.scatter([ego_x0], [ego_y0], color="black", marker="x", s=50, zorder=5, label="Start (shared)")

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        if not ppt:
            ax.set_title(f"{MANEUVER_ID}: diagnostic downstream comparison (legacy vs. frenet_mpc)\n"
                          "Same fixed MERGE-every-step action sequence -- NOT an FSM-vs-PPO result", fontsize=8)
        ax.legend(loc="best", fontsize=(11 if ppt else 8))
        fig.tight_layout()

        if ppt:
            _style.savefig_ppt(fig, os.path.join(OUT_PPT, f"fig_legacy_vs_frenet_mpc_{MANEUVER_ID}_ppt"))
        else:
            _style.savefig_paper(fig, os.path.join(OUT_PAPER, f"fig_legacy_vs_frenet_mpc_{MANEUVER_ID}_paper"))
        plt.close(fig)

    print("[fig_legacy_vs_frenet_mpc] done.")


if __name__ == "__main__":
    main()
