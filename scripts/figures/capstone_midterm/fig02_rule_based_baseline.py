#!/usr/bin/env python3
"""Capstone midterm Fig 2: FSM -> Frenet Planner -> LTV-MPC baseline.

Real data only. Two real artifacts are combined:

1. The ACTUAL closed-loop executed trajectory: FsmPolicy
   (src/environment/fsm_policy.py, the project's real rule-based
   behavior-decision module) driving a real MergeEnvironment rollout
   step-by-step via the public reset()/step() API -- the same
   BehaviorExecutor + CommonDownstream (Frenet planner + LTV-MPC) path
   PPO also uses, just with FsmPolicy instead of a PPO policy choosing
   the action each step.

2. The REAL per-action candidates at the decision frame where the FSM
   first commits to MERGE: this project's planner
   (src.planning.frenet_planner.plan) generates exactly ONE candidate
   trajectory per BehaviorAction (confirmed: no multi-candidate search
   exists in this codebase) -- so "candidate paths" here means the 4
   real, actually-computed per-action candidates at that one frame
   (KEEP/FOLLOW/MERGE/STOP), not a synthesized/invented path family.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.environment.behavior_action import BehaviorAction, BehaviorExecutor
from src.environment.fsm_policy import FsmPolicy
from src.environment.full_split_evaluator import load_decision_dataset_maneuver_specs
from src.environment.merge_environment import MergeEnvironment
from src.planning.frenet_planner import EgoKinematicState, FollowInputs, PlanRequest, load_planner_config, plan
from src.visualization import style as _style

MANEUVER_ID = "training_tfexample.tfrecord-00000-of-01000#378__t18__237_456"
MAX_STEPS = 100
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                           "outputs", "capstone_midterm_figures"))
OUTPUT_BASENAME = "fig02_rule_based_baseline"

COLOR_CANDIDATE = "#B0B0B0"
COLOR_SELECTED = _style.COLOR_TARGET_LANE
COLOR_EXECUTED = _style.COLOR_EGO

ACTION_LABELS = {
    BehaviorAction.KEEP: "KEEP", BehaviorAction.FOLLOW: "FOLLOW",
    BehaviorAction.MERGE: "MERGE", BehaviorAction.STOP: "STOP",
}


def _follow_inputs_from_observation(observation):
    return FollowInputs(
        source_front_gap_m=float(observation[11]) if observation[10] != 0.0 else None,
        source_front_relative_speed_mps=float(observation[12]) if observation[10] != 0.0 else None,
        target_front_gap_m=float(observation[3]) if observation[2] != 0.0 else None,
        target_front_relative_speed_mps=float(observation[4]) if observation[2] != 0.0 else None,
    )


def main():
    env = MergeEnvironment(dataset_config_path="configs/dataset.yaml", downstream_mode="frenet_mpc")
    specs = {s.maneuver_id: s for s in load_decision_dataset_maneuver_specs("train")}
    spec = specs[MANEUVER_ID]

    fsm = FsmPolicy()
    executor = BehaviorExecutor()
    planner_config = load_planner_config()

    observation, info = env.reset(spec)
    ego_xy_history = [(env._current_ego_pose_and_speed()[0], env._current_ego_pose_and_speed()[1])]

    # Capture the REAL per-action candidates at the FIRST decision
    # frame (reset-time, ego still in the source lane) -- this is
    # where KEEP/FOLLOW/STOP (source-lane-centered) and MERGE
    # (target-lane-centered) candidates visibly diverge; later frames
    # (near/at merge completion) converge toward the same lane and are
    # not visually informative.
    source_ref, target_ref = env._get_active_reference_lines()
    ego_x, ego_y, ego_yaw, ego_speed = env._current_ego_pose_and_speed()
    ego_state = EgoKinematicState(x=ego_x, y=ego_y, yaw=ego_yaw, speed_mps=ego_speed)
    follow_inputs = _follow_inputs_from_observation(observation)
    candidates = {}
    for candidate_action in (BehaviorAction.KEEP, BehaviorAction.FOLLOW,
                              BehaviorAction.MERGE, BehaviorAction.STOP):
        objective = executor.compute_objective(candidate_action, observation)
        request = PlanRequest(
            behavior_action=candidate_action, objective=objective, ego_state=ego_state,
            source_reference=source_ref, target_reference=target_ref,
            follow_inputs=follow_inputs, surrounding_agents=env._build_surrounding_agents(),
        )
        result = plan(request, planner_config)
        if result.cartesian_trajectory is not None:
            candidates[candidate_action] = result.cartesian_trajectory

    first_decision = fsm.decide(observation).action
    merge_decision_frame_data = {"candidates": candidates, "selected": first_decision}

    for step_index in range(MAX_STEPS):
        decision = fsm.decide(observation)
        action = decision.action

        next_observation, _, terminated, truncated, info = env.step(action)
        ego_x, ego_y, _, _ = env._current_ego_pose_and_speed()
        ego_xy_history.append((ego_x, ego_y))
        observation = next_observation
        if terminated or truncated:
            break

    ego_xy_history = np.array(ego_xy_history)

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        "font.size": 11, "font.family": "DejaVu Sans",
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for lane_id in (info["active_source_lane_id"], info["active_target_lane_id"]):
        poly = env._polylines_by_id[lane_id]
        ax.plot(poly.xy[:, 0], poly.xy[:, 1], color="#D9D9D9", linewidth=2.0, zorder=1)

    if merge_decision_frame_data is not None:
        for candidate_action, cart in merge_decision_frame_data["candidates"].items():
            is_selected = candidate_action == merge_decision_frame_data["selected"]
            color = COLOR_SELECTED if is_selected else COLOR_CANDIDATE
            lw = 2.2 if is_selected else 1.4
            ls = "-" if is_selected else "--"
            label = f"Selected path ({ACTION_LABELS[candidate_action]})" if is_selected else "Candidate paths"
            ax.plot(cart.x, cart.y, color=color, linewidth=lw, linestyle=ls, zorder=3,
                     label=label if not is_selected or True else None)

    ax.plot(ego_xy_history[:, 0], ego_xy_history[:, 1], color=COLOR_EXECUTED,
             linewidth=2.0, linestyle=":", zorder=4, label="Ego trajectory")
    ax.scatter(ego_xy_history[0, 0], ego_xy_history[0, 1], color=COLOR_EXECUTED,
               s=40, zorder=5, marker="o")

    # De-duplicate the repeated "Candidate paths" legend entries.
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), loc="best", frameon=False, handlelength=1.8)

    ax.text(0.03, 0.95, f"Behavior: {ACTION_LABELS[merge_decision_frame_data['selected']]}",
             transform=ax.transAxes, fontsize=9, verticalalignment="top")

    all_x = np.concatenate([ego_xy_history[:, 0]] + [
        c.x for c in (merge_decision_frame_data["candidates"].values() if merge_decision_frame_data else [])
    ])
    all_y = np.concatenate([ego_xy_history[:, 1]] + [
        c.y for c in (merge_decision_frame_data["candidates"].values() if merge_decision_frame_data else [])
    ])
    pad_x = max(12.0, 0.15 * (all_x.max() - all_x.min()))
    pad_y = max(12.0, 0.15 * (all_y.max() - all_y.min()))
    ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME)
    fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"maneuver_id: {MANEUVER_ID}")
    print(f"executed steps: {len(ego_xy_history) - 1}")
    print(f"merge_decision_frame captured: {merge_decision_frame_data is not None}")
    print(f"Wrote {out_path}.png / .pdf")


if __name__ == "__main__":
    main()
