#!/usr/bin/env python3
"""Capstone midterm Fig 4: evaluation framework (representative outcomes).

Real data only. Source:
outputs/visualizations/train_diag64_step000036/selected_episodes.json
-- the real outcome-scan output of scripts/visualize_ppo.py
(checkpoint outputs/checkpoints/full_seed0_step000036.pkl, deterministic
policy, the 64 Fixed-TRAIN64 canonical maneuvers). The first listed
maneuver_id for Success/Collision/Timeout is re-run here (same
checkpoint, same deterministic policy, identical downstream) only to
re-render it at readable size/style for this document -- the outcome
itself is never altered.

Offroad is deliberately OMITTED: selected_episodes.json's own
"offroad" list is empty (offroad never occurred across the 64
evaluated maneuvers at this checkpoint) -- no offroad panel is
fabricated.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.environment.full_split_evaluator import load_decision_dataset_maneuver_specs
from src.environment.merge_environment import MergeEnvironment
from src.scenarios.merge_v2 import MERGE_DATASET_SCHEMA
from src.training.config import load_reward_config
from src.visualization import style as _style
from src.visualization.ppo_checkpoint_policy import restore_ppo_checkpoint
from src.visualization.ppo_rollout import run_ppo_episode

CHECKPOINT_PATH = "outputs/checkpoints/full_seed0_step000036.pkl"
SELECTED_EPISODES_PATH = "outputs/visualizations/train_diag64_step000036/selected_episodes.json"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                           "outputs", "capstone_midterm_figures"))
OUTPUT_BASENAME = "fig04_evaluation_framework"

PANEL_OUTCOMES = ["success", "collision", "timeout"]
PANEL_TITLES = {"success": "Success", "collision": "Collision", "timeout": "Timeout"}
PANEL_COLOR = {
    "success": _style.COLOR_SUCCESS,
    "collision": _style.COLOR_COLLISION,
    "timeout": _style.COLOR_TRUNCATION,
}


def main():
    with open(SELECTED_EPISODES_PATH) as f:
        selected = json.load(f)

    missing = [o for o in PANEL_OUTCOMES if not selected.get(o)]
    if missing:
        print(f"WARNING: no representative episode for outcomes {missing}; skipping those panels.")
    panel_outcomes = [o for o in PANEL_OUTCOMES if selected.get(o)]
    if not selected.get("offroad"):
        print("Note: 'offroad' outcome list is empty in the source data -- omitted, not fabricated.")

    restored = restore_ppo_checkpoint(CHECKPOINT_PATH, expected_dataset_schema_version=MERGE_DATASET_SCHEMA)
    reward_config = load_reward_config(restored.reward_config_path or "configs/reward.yaml")
    train_specs = {s.maneuver_id: s for s in load_decision_dataset_maneuver_specs("train")}
    env = MergeEnvironment(dataset_config_path="configs/dataset.yaml", downstream_mode="frenet_mpc",
                            required_dataset_schema_version=MERGE_DATASET_SCHEMA)

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        "font.size": 10, "font.family": "DejaVu Sans",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })
    fig, axes = plt.subplots(1, len(panel_outcomes), figsize=(5.8, 2.4))
    if len(panel_outcomes) == 1:
        axes = [axes]

    for ax, outcome in zip(axes, panel_outcomes):
        maneuver_id = selected[outcome][0]
        spec = train_specs[maneuver_id]
        episode = run_ppo_episode(
            env=env, maneuver=spec, policy=restored.policy,
            value_network=restored.value_network, value_params=restored.value_params,
            reward_config=reward_config, max_steps=100, policy_mode="deterministic",
        )

        for lane_id in (episode.steps[0].active_source_lane_id, episode.steps[0].active_target_lane_id):
            if lane_id is None:
                continue
            poly = env._polylines_by_id[lane_id]
            ax.plot(poly.xy[:, 0], poly.xy[:, 1], color="#D9D9D9", linewidth=1.5, zorder=1)

        ego_x = np.array([s.ego_x for s in episode.steps])
        ego_y = np.array([s.ego_y for s in episode.steps])
        ax.plot(ego_x, ego_y, color=PANEL_COLOR[outcome], linewidth=1.8, zorder=3)
        ax.scatter(ego_x[0], ego_y[0], color=PANEL_COLOR[outcome], s=18, zorder=4)
        ax.scatter(ego_x[-1], ego_y[-1], color=PANEL_COLOR[outcome], s=28, marker="x", zorder=4)

        pad = 8.0
        ax.set_xlim(ego_x.min() - pad, ego_x.max() + pad)
        ax.set_ylim(ego_y.min() - pad, ego_y.max() + pad)
        ax.set_aspect("equal")
        ax.set_title(PANEL_TITLES[outcome], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        print(f"{outcome}: maneuver_id={maneuver_id}, steps={episode.physical_step_count}, "
              f"termination_reason={episode.termination_reason}")

    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME)
    fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_path}.png / .pdf")


if __name__ == "__main__":
    main()
