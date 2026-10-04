#!/usr/bin/env python3
"""Capstone midterm Fig 3: PPO training progress.

Real data only. Source:
outputs/diagnostics/retrain_freeze3213b1b/wandb_per_update_raw.csv
-- 36 rows, one per completed PPO update of the Full PPO Retraining
run (W&B run zid8h94i), parsed from that run's own
wandb/run-20260930_203415-zid8h94i/files/output.log by
scripts/diagnostics/analyze_wandb_training_history.py (already run in
a prior session; this script only re-plots the same CSV, it does not
regenerate or alter it).

Left panel: train/episode_return per update.
Right panel: train/success_rate and train/collision_rate per update.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "outputs", "diagnostics", "retrain_freeze3213b1b", "wandb_per_update_raw.csv",
))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                           "outputs", "capstone_midterm_figures"))
OUTPUT_BASENAME = "fig03_ppo_training"

COLOR_RETURN = "#1a1a1a"
COLOR_SUCCESS = "#4C72B0"
COLOR_COLLISION = "#C44E52"


def main():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    updates = [int(r["ppo_update_step"]) for r in rows]
    episode_return = [float(r["train/episode_return"]) for r in rows]
    success_rate = [float(r["train/success_rate"]) for r in rows]
    collision_rate = [float(r["train/collision_rate"]) for r in rows]

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        "font.size": 10, "font.family": "DejaVu Sans",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.3))

    ax = axes[0]
    ax.plot(updates, episode_return, color=COLOR_RETURN, linewidth=1.4)
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Episode return")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(updates, success_rate, color=COLOR_SUCCESS, linewidth=1.4,
             linestyle="-", label="Success")
    ax.plot(updates, collision_rate, color=COLOR_COLLISION, linewidth=1.4,
             linestyle="--", label="Collision")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, handlelength=1.8)

    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME)
    fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Source: {CSV_PATH}")
    print(f"Updates plotted: {updates[0]}-{updates[-1]} ({len(updates)} rows)")
    print(f"Wrote {out_path}.png / .pdf")


if __name__ == "__main__":
    main()
