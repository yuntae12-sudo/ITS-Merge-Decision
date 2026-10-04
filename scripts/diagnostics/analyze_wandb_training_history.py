#!/usr/bin/env python3
"""W&B local output.log training-history analysis (read-only).

Parses the plaintext "  update N: {...}" metric-dict lines already
written to a W&B offline/online run's files/output.log (printed by
src.training.trainer.run_training's own per-update logging -- this
script never touches W&B's API or binary .wandb format), and
summarizes them into 4 quarters (updates 1-9, 10-18, 19-27, 28-36).

Never modifies training/PPO/reward/environment behavior -- pure
analysis of already-produced training output.
"""

import ast
import csv
import json
import re
import statistics
import sys

QUARTER_BOUNDARIES = [(1, 9), (10, 18), (19, 27), (28, 36)]

KEY_METRICS = [
    "train/success_rate", "train/collision_rate", "train/offroad_rate", "train/timeout_rate",
    "train/episode_return",
    "reward/terminal", "reward/safety", "reward/progress", "reward/decision",
    "ppo/entropy_mean", "ppo/approx_kl_mean", "ppo/exact_kl_mean", "ppo/clip_fraction_mean",
    "ppo/policy_loss_mean", "ppo/value_loss_mean", "ppo/explained_variance",
    "action/keep_ratio", "action/follow_ratio", "action/merge_ratio", "action/stop_ratio",
    "downstream/intervention_rate", "downstream/planner_infeasible_rate",
    "downstream/collision_blocked_rate", "downstream/controller_failure_rate",
    "downstream/invalid_reference_rate",
    "runtime/env_steps_per_sec",
]


def parse_output_log(path):
    """Returns list of dicts, one per update (0-indexed as logged, i.e.
    update_index -- update 0 corresponds to ppo_update_step=1)."""

    pattern = re.compile(r"^\s*update (\d+): (\{.*\})\s*$")
    updates = {}
    with open(path) as f:
        for line in f:
            m = pattern.match(line)
            if not m:
                continue
            update_index = int(m.group(1))
            metrics = ast.literal_eval(m.group(2))
            updates[update_index] = metrics
    return [updates[i] for i in sorted(updates.keys())]


def summarize_quarter(updates_1_indexed, lo, hi):
    """updates_1_indexed: dict {ppo_update_step (1-indexed): metrics}."""

    rows = [updates_1_indexed[s] for s in range(lo, hi + 1) if s in updates_1_indexed]
    if not rows:
        return None
    summary = {}
    for key in KEY_METRICS:
        values = [r[key] for r in rows if key in r]
        if values:
            summary[key] = {
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "trend": values[-1] - values[0],
            }
    return summary


def main():
    log_path = "wandb/run-20260930_203415-zid8h94i/files/output.log"
    output_dir = "outputs/diagnostics/retrain_freeze3213b1b"

    updates_list = parse_output_log(log_path)
    print(f"Parsed {len(updates_list)} updates from {log_path}")

    # update_index 0 -> ppo_update_step 1
    updates_1_indexed = {i + 1: m for i, m in enumerate(updates_list)}

    quarter_rows = []
    for lo, hi in QUARTER_BOUNDARIES:
        summary = summarize_quarter(updates_1_indexed, lo, hi)
        if summary is None:
            continue
        row = {"updates": f"{lo}-{hi}"}
        for key in KEY_METRICS:
            if key in summary:
                row[f"{key}_mean"] = summary[key]["mean"]
                row[f"{key}_trend"] = summary[key]["trend"]
        quarter_rows.append(row)
        print(f"\n--- updates {lo}-{hi} ---")
        for key in KEY_METRICS:
            if key in summary:
                s = summary[key]
                print(f"  {key}: mean={s['mean']:.4f} min={s['min']:.4f} max={s['max']:.4f} trend={s['trend']:+.4f}")

    with open(f"{output_dir}/wandb_training_summary.csv", "w", newline="") as f:
        if quarter_rows:
            writer = csv.DictWriter(f, fieldnames=list(quarter_rows[0].keys()))
            writer.writeheader()
            for row in quarter_rows:
                writer.writerow(row)

    # Full per-update raw CSV too, for direct inspection.
    with open(f"{output_dir}/wandb_per_update_raw.csv", "w", newline="") as f:
        fieldnames = ["ppo_update_step"] + KEY_METRICS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step, metrics in sorted(updates_1_indexed.items()):
            row = {"ppo_update_step": step}
            for key in KEY_METRICS:
                row[key] = metrics.get(key)
            writer.writerow(row)

    print(f"\nWrote {output_dir}/wandb_training_summary.csv")
    print(f"Wrote {output_dir}/wandb_per_update_raw.csv")


if __name__ == "__main__":
    main()
