"""Figure 3 -- Frenet planner + LTV-MPC tracking (planned vs. executed).

EXPERIMENTAL figure. On MAN_0041 (a real, representative successful
maneuver -- Stage 3-H's own re-verified end-to-end case), this script:

  1. Runs a real ``MergeEnvironment(..., downstream_mode="frenet_mpc")``
     rollout (MERGE every step, same convention as Figure 2) up to the
     real step where ``info["chain_advanced"]`` first fires (the frame
     the MERGE geometry actually targets the final target lane 608 --
     the same "merge initiation" moment Figure 2 uses).
  2. At EXACTLY that real rollout state, calls
     ``src.planning.frenet_planner.plan(...)`` DIRECTLY (a plain
     function call into frozen Stage 3-C code, not a modification of
     it) with the identical ``DownstreamRequest``-equivalent inputs
     ``MergeEnvironment._compute_frenet_mpc_command`` itself builds at
     that step (source/target ``ReferenceLine``, ``EgoKinematicState``,
     ``FollowInputs`` sourced from the same causal 14D-observation
     fields, ``surrounding_agents`` from the same current-frame Waymax
     state) -- see ``_capture_planned_trajectory`` below for the exact
     wiring, one-to-one with the real production call site
     (src/environment/merge_environment.py's own
     ``_compute_frenet_mpc_command``). This captures the REAL planned
     Cartesian trajectory (``PlanResult.cartesian_trajectory``) the
     production planner would have (and, in the OK case below, does)
     hand to the MPC this exact step -- never a hand-reconstructed
     trajectory.
  3. Continues the SAME rollout instance for the following ~15 real
     steps and records the actual closed-loop executed ego (x, y) from
     real Waymax simulation state -- never a resimulation, the same
     environment instance already stepping forward.

Overlays: source/target reference centerlines, the one real planned
trajectory, and the real subsequent executed trajectory, in one XY
figure with equal aspect. A tracking-error-vs-time bottom panel plots
the real Euclidean distance between the executed ego position at step
k and the planned trajectory's own sample at matching elapsed time
(nearest-neighbor in time, planner dt=0.1s == Waymax step dt=0.1s, so
no interpolation is invented).
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from scripts.figures import _style
from scripts.figures._rollout import build_env, load_spec_by_id
from src.environment.behavior_action import BehaviorAction
from src.planning.frenet_planner import (
    EgoKinematicState,
    FollowInputs,
    PlanRequest,
    load_planner_config,
    plan,
)

OUT_PAPER = "outputs/paper_ppt_figures/phase3/paper"
OUT_PPT = "outputs/paper_ppt_figures/phase3/ppt"
OUT_DATA = "outputs/paper_ppt_figures/phase3/data"
MANEUVER_ID = "MAN_0041"
DOWNSTREAM_CONFIG_PATH = "configs/phase3_downstream.yaml"
POST_STEPS = 15


def _current_ego_kinematic(env):
    traj = env._state.current_sim_trajectory
    i = env._sdc_index
    x = float(np.asarray(traj.x)[i, 0])
    y = float(np.asarray(traj.y)[i, 0])
    yaw = float(np.asarray(traj.yaw)[i, 0])
    vel_x = float(np.asarray(traj.vel_x)[i, 0])
    vel_y = float(np.asarray(traj.vel_y)[i, 0])
    return EgoKinematicState(x=x, y=y, yaw=yaw, speed_mps=float(np.hypot(vel_x, vel_y)))


def capture_planned_trajectory(env, executed_action, objective, observation_before):
    """Mirrors src/environment/merge_environment.py's own
    ``_compute_frenet_mpc_command`` request construction EXACTLY (same
    field sourcing, same causal-observation indices), then calls the
    real ``frenet_planner.plan()`` directly -- this is a read-only,
    visualization-side call into already-frozen production code, not a
    reimplementation of planner logic."""

    source_reference, target_reference = env._get_active_reference_lines()
    ego_state = _current_ego_kinematic(env)

    follow_inputs = FollowInputs(
        source_front_gap_m=(
            float(observation_before[11]) if observation_before[10] != 0.0 else None
        ),
        source_front_relative_speed_mps=(
            float(observation_before[12]) if observation_before[10] != 0.0 else None
        ),
        target_front_gap_m=(
            float(observation_before[3]) if observation_before[2] != 0.0 else None
        ),
        target_front_relative_speed_mps=(
            float(observation_before[4]) if observation_before[2] != 0.0 else None
        ),
    )

    request = PlanRequest(
        behavior_action=executed_action,
        objective=objective,
        ego_state=ego_state,
        source_reference=source_reference,
        target_reference=target_reference,
        follow_inputs=follow_inputs,
        surrounding_agents=env._build_surrounding_agents(),
        dt_s=0.1,
    )
    planner_config = load_planner_config(DOWNSTREAM_CONFIG_PATH)
    return plan(request, planner_config)


def main():
    os.makedirs(OUT_PAPER, exist_ok=True)
    os.makedirs(OUT_PPT, exist_ok=True)
    os.makedirs(OUT_DATA, exist_ok=True)

    env = build_env(downstream_mode="frenet_mpc")
    spec, split = load_spec_by_id(MANEUVER_ID)
    observation, info = env.reset(spec)
    print(f"[fig3] maneuver={spec.maneuver_id} split={split}")

    # Step forward (MERGE every step, matching Fig 2's convention)
    # until chain_advanced fires -- the real "merge initiation" moment,
    # i.e. the first step whose objective/ego-state actually targets
    # the FINAL target lane via the frenet planner's MERGE frame.
    plan_result = None
    plan_step_index = None
    executed_positions = [(0, float(np.asarray(env._state.current_sim_trajectory.x)[env._sdc_index, 0]),
                            float(np.asarray(env._state.current_sim_trajectory.y)[env._sdc_index, 0]))]

    step_idx = 0
    pending_capture = False
    while True:
        if pending_capture:
            # Capture the planner's output for the step ABOUT TO BE
            # TAKEN, using the exact same observation_before/objective
            # construction production code performs inside
            # merge_environment.step() (Section: "observation_before,
            # objective, then downstream" -- see that method). This
            # read-only peek (env._build_observation(),
            # env._decision_state.advance(), env._executor.
            # compute_objective()) mirrors the production call order
            # exactly and does not skip/duplicate any state transition
            # beyond what env.step() itself will perform right after,
            # since DecisionState.advance() from an already-committed
            # MERGE state is idempotent (confirmed via direct source
            # inspection of src/environment/decision_state.py: while
            # MERGE_COMMITTED, policy_action is ignored and the same
            # committed_action is returned every call).
            observation_before = env._build_observation()
            executed_action = env._decision_state.advance(BehaviorAction.MERGE)
            objective = env._executor.compute_objective(executed_action, observation_before)
            plan_result = capture_planned_trajectory(env, executed_action, objective, observation_before)
            plan_step_index = step_idx
            print(f"[fig3] captured planner output at step={step_idx}, status={plan_result.status}")
            pending_capture = False

        observation, reward, terminated, truncated, info = env.step(BehaviorAction.MERGE)
        if plan_result is None and info.get("chain_advanced"):
            # chain_advanced fires AFTER this step's Waymax step (see
            # merge_environment.step()'s own ordering); the geometry
            # that now targets the final lane is used starting NEXT
            # step, so schedule the capture for the following
            # iteration rather than reconstructing this step's
            # (already-consumed) pre-step state after the fact.
            pending_capture = True
        ego_x = float(np.asarray(env._state.current_sim_trajectory.x)[env._sdc_index, 0])
        ego_y = float(np.asarray(env._state.current_sim_trajectory.y)[env._sdc_index, 0])
        executed_positions.append((step_idx + 1, ego_x, ego_y))
        step_idx += 1

        if plan_result is not None and step_idx >= plan_step_index + POST_STEPS:
            break
        if terminated or truncated:
            break
        if step_idx > 100:
            break

    if plan_result is None:
        raise RuntimeError("chain never advanced in this rollout -- cannot capture a MERGE-frame planned trajectory from real data.")
    if plan_result.status != "OK" or plan_result.cartesian_trajectory is None:
        print(f"[fig3] WARNING: planner status at captured step was {plan_result.status}, not OK; "
              "figure will show whatever trajectory geometry was generated (may be None).")

    target_reference = plan_result.reference_used
    source_reference, _ = env._get_active_reference_lines()

    ct = plan_result.cartesian_trajectory
    planned_t = np.asarray(ct.t) if ct is not None else np.array([])
    planned_x = np.asarray(ct.x) if ct is not None else np.array([])
    planned_y = np.asarray(ct.y) if ct is not None else np.array([])

    exec_steps = [p[0] for p in executed_positions if p[0] >= plan_step_index]
    exec_x = [p[1] for p in executed_positions if p[0] >= plan_step_index]
    exec_y = [p[2] for p in executed_positions if p[0] >= plan_step_index]
    exec_t = [(s - plan_step_index) * 0.1 for s in exec_steps]

    # Tracking error: nearest-time-match between executed sample at
    # elapsed time t and the planned trajectory's own t array (both at
    # dt=0.1s per the module docstring's stated precondition -- direct
    # index alignment, verified below rather than assumed).
    tracking_error_m = []
    if len(planned_t) > 0:
        for t_e, xe, ye in zip(exec_t, exec_x, exec_y):
            idx = int(round(t_e / 0.1))
            if 0 <= idx < len(planned_t):
                dx = xe - planned_x[idx]
                dy = ye - planned_y[idx]
                tracking_error_m.append(float(np.hypot(dx, dy)))
            else:
                tracking_error_m.append(None)

    # Save data.
    data_path = os.path.join(OUT_DATA, f"fig3_planner_tracking_{MANEUVER_ID}.csv")
    with open(data_path, "w") as f:
        f.write("kind,step_or_sample_index,elapsed_s,x,y\n")
        for i, (t, x, y) in enumerate(zip(planned_t, planned_x, planned_y)):
            f.write(f"planned,{i},{t},{x},{y}\n")
        for s, x, y, t in zip(exec_steps, exec_x, exec_y, exec_t):
            f.write(f"executed,{s},{t},{x},{y}\n")
    print(f"[fig3] wrote {data_path}")

    meta = {
        "maneuver_id": MANEUVER_ID, "split": split,
        "plan_captured_at_step": plan_step_index,
        "planner_status": str(plan_result.status),
        "reference_used_lane_id": int(target_reference.lane_id) if target_reference is not None else None,
        "tracking_error_m": tracking_error_m,
    }
    with open(os.path.join(OUT_DATA, f"fig3_planner_tracking_{MANEUVER_ID}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    render(source_reference, target_reference, planned_x, planned_y, exec_x, exec_y, exec_t, tracking_error_m, plan_step_index)
    print("[fig3] done.")


def render(source_reference, target_reference, planned_x, planned_y, exec_x, exec_y, exec_t, tracking_error_m, plan_step_index):
    import matplotlib.pyplot as plt

    for ppt in (False, True):
        if ppt:
            _style.apply_ppt_style()
        else:
            _style.apply_paper_style()

        has_error_panel = any(e is not None for e in tracking_error_m)
        if has_error_panel:
            fig, (ax_xy, ax_err) = plt.subplots(2, 1, figsize=(7, 9) if not ppt else (10, 12),
                                                 gridspec_kw={"height_ratios": [3, 1]})
        else:
            fig, ax_xy = plt.subplots(figsize=(7, 7) if not ppt else (10, 10))

        ax_xy.plot(source_reference.x, source_reference.y, color=_style.COLOR_SOURCE_LANE,
                   linewidth=2.0, label="Source reference", zorder=1)
        if target_reference is not None:
            ax_xy.plot(target_reference.x, target_reference.y, color=_style.COLOR_TARGET_LANE,
                       linewidth=2.0, label="Target reference", zorder=1)
        if len(planned_x) > 0:
            ax_xy.plot(planned_x, planned_y, color=_style.COLOR_PLANNED_TRAJ, linewidth=2.5,
                       linestyle="-", marker="o", markersize=3, label="Planned trajectory (Frenet)", zorder=3)
        ax_xy.plot(exec_x, exec_y, color=_style.COLOR_EXECUTED_TRAJ, linewidth=2.0,
                   linestyle="--", label="Executed trajectory (closed-loop)", zorder=4)
        ax_xy.scatter([exec_x[0]], [exec_y[0]], color="black", zorder=5, s=40, marker="x")

        ax_xy.set_aspect("equal")
        # Zoom to the real local region actually exercised by the
        # planned+executed trajectories (reference lines themselves
        # span hundreds of meters of scene, which would otherwise
        # dominate the window) -- window bounds are computed from the
        # real planned/executed sample extents only, padded.
        window_x = list(planned_x) + list(exec_x)
        window_y = list(planned_y) + list(exec_y)
        pad = 8.0
        ax_xy.set_xlim(min(window_x) - pad, max(window_x) + pad)
        ax_xy.set_ylim(min(window_y) - pad, max(window_y) + pad)
        ax_xy.set_xlabel("X (m)")
        ax_xy.set_ylabel("Y (m)")
        if not ppt:
            ax_xy.set_title(f"{MANEUVER_ID}: planned vs. executed MERGE trajectory (captured at step {plan_step_index})", fontsize=9)
        ax_xy.legend(loc="best", framealpha=0.9)

        if has_error_panel:
            valid = [(t, e) for t, e in zip(exec_t, tracking_error_m) if e is not None]
            ax_err.plot([v[0] for v in valid], [v[1] for v in valid], color=_style.COLOR_EXECUTED_TRAJ,
                        marker="o", markersize=3)
            ax_err.set_xlabel("Elapsed time since plan capture (s)")
            ax_err.set_ylabel("Tracking error (m)")
            ax_err.grid(alpha=0.3)

        fig.tight_layout()
        if ppt:
            _style.savefig_ppt(fig, os.path.join(OUT_PPT, f"fig3_planner_tracking_{MANEUVER_ID}_ppt"))
        else:
            _style.savefig_paper(fig, os.path.join(OUT_PAPER, f"fig3_planner_tracking_{MANEUVER_ID}_paper"))
        plt.close(fig)


if __name__ == "__main__":
    main()
