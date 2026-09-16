"""Figure 7 -- decision window timeline (MAN_0041).

EXPERIMENTAL figure (visual style is schematic-like -- a horizontal
timeline -- but every plotted frame position is a REAL number read
directly from a real ``MergeEnvironment`` reset/rollout, never
invented). For MAN_0041:

  - decision_start_frame: real value from ``info["decision_start_frame"]``
    at reset() (the earliest causal frame the source lane is
    raw-identifiable -- Stage B-2.8's production computation).
  - merge_start_frame: real value from ``info["merge_start_frame"]``
    (the maneuver's own canonical physical merge-region reference,
    from the maneuver table).
  - termination/success frame: the REAL rollout's own terminal
    absolute frame, computed as
    ``decision_start_frame + steps_elapsed_at_termination`` from an
    actual frenet_mpc MERGE-every-step rollout (same convention as
    Figures 2/3) -- not assumed from Stage 3-H's prose citation
    (frames 40/59-60), independently re-run here.

All three frame numbers are also written to the sidecar data file so
the figure is fully traceable.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.figures import _style
from scripts.figures._rollout import run_rollout

OUT_PAPER = "outputs/paper_ppt_figures/phase3/paper"
OUT_PPT = "outputs/paper_ppt_figures/phase3/ppt"
OUT_DATA = "outputs/paper_ppt_figures/phase3/data"
MANEUVER_ID = "MAN_0041"


def main():
    os.makedirs(OUT_PAPER, exist_ok=True)
    os.makedirs(OUT_PPT, exist_ok=True)
    os.makedirs(OUT_DATA, exist_ok=True)

    spec, split, snaps, env = run_rollout(MANEUVER_ID, max_steps=90, downstream_mode="frenet_mpc")
    info0 = snaps[0].info
    terminal = snaps[-1]

    decision_start_frame = info0["decision_start_frame"]
    merge_start_frame = info0["merge_start_frame"]
    terminal_absolute_frame = decision_start_frame + terminal.step_index

    print(f"[fig7] maneuver={MANEUVER_ID} split={split}")
    print(f"[fig7] decision_start_frame={decision_start_frame} merge_start_frame={merge_start_frame} "
          f"terminal_step={terminal.step_index} terminal_absolute_frame={terminal_absolute_frame} "
          f"termination_reason={terminal.termination_reason}")

    meta = {
        "maneuver_id": MANEUVER_ID, "split": split,
        "decision_start_frame": decision_start_frame,
        "merge_start_frame": merge_start_frame,
        "terminal_step_index": terminal.step_index,
        "terminal_absolute_frame": terminal_absolute_frame,
        "termination_reason": terminal.termination_reason,
    }
    with open(os.path.join(OUT_DATA, f"fig7_decision_timeline_{MANEUVER_ID}.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[fig7] wrote data file.")

    import matplotlib.pyplot as plt

    events = [
        ("Decision start\n(causal)", decision_start_frame, _style.COLOR_SOURCE_LANE),
        ("Merge start\n(canonical)", merge_start_frame, _style.COLOR_TARGET_LANE),
        (f"Terminal\n({terminal.termination_reason})", terminal_absolute_frame, _style.COLOR_SUCCESS),
    ]
    events.sort(key=lambda e: e[1])

    for ppt in (False, True):
        if ppt:
            _style.apply_ppt_style()
        else:
            _style.apply_paper_style()

        fig, ax = plt.subplots(figsize=(8, 2.4) if not ppt else (12, 4))
        y = 0
        xmin = min(e[1] for e in events)
        xmax = max(e[1] for e in events)
        pad = max((xmax - xmin) * 0.15, 2)
        ax.hlines(y, xmin - pad, xmax + pad, color="gray", linewidth=1.5, zorder=1)
        for label, frame, color in events:
            ax.scatter([frame], [y], color=color, s=(160 if ppt else 90), zorder=3, edgecolor="black")
            ax.annotate(f"{label}\nframe {frame}", (frame, y), textcoords="offset points",
                        xytext=(0, 18), ha="center", fontsize=(12 if ppt else 8))
        ax.set_yticks([])
        ax.set_xlabel("Absolute scene frame index")
        if not ppt:
            ax.set_title(f"{MANEUVER_ID} ({split}) -- decision window timeline (real frame values; "
                          "schematic layout)", fontsize=9)
        ax.set_ylim(-1, 1.5)
        fig.tight_layout()

        if ppt:
            _style.savefig_ppt(fig, os.path.join(OUT_PPT, f"fig7_decision_timeline_{MANEUVER_ID}_ppt"))
        else:
            _style.savefig_paper(fig, os.path.join(OUT_PAPER, f"fig7_decision_timeline_{MANEUVER_ID}_paper"))
        plt.close(fig)

    print("[fig7] done.")


if __name__ == "__main__":
    main()
