"""Rendering (GIF / trace.png / rollout_4panel.png) for one PPO
visualization episode (docs/ppo/PPO_VISUALIZATION_GUIDE.md).

Reuses this repo's existing Phase-3 figure conventions
(``scripts/figures/_style.py``'s colors/oriented-box drawing) so PPO
visualization figures use the same visual language as the existing
paper/PPT figures, rather than inventing a new one. VISUALIZATION-ONLY:
every plotted value comes from a real ``PPOStepRecord``
(``src.visualization.ppo_rollout``) captured during an actual
``MergeEnvironment`` rollout -- nothing here generates or interpolates
trajectory/observation/metric data.
"""

import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.figures import _style
from src.visualization.ppo_rollout import PPOEpisodeResult, PPOStepRecord

_HALF_WINDOW_M = 22.0
_LANE_PAD_M = 55.0


def _draw_frame(ax, env_polylines, record: PPOStepRecord, ego_history_xy):
    for lane_id, xy in env_polylines:
        color = _style.COLOR_OTHER_LANE
        if lane_id == record.active_source_lane_id:
            color = _style.COLOR_SOURCE_LANE
        elif lane_id == record.active_target_lane_id:
            color = _style.COLOR_TARGET_LANE
        lw = 2.0 if lane_id in (record.active_source_lane_id, record.active_target_lane_id) else 1.0
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=lw, zorder=1)

    for agent in record.agents:
        _style.draw_oriented_box(
            ax, agent.x, agent.y, agent.yaw, agent.length, agent.width,
            _style.COLOR_SURROUNDING_AGENT, zorder=3,
        )

    if len(ego_history_xy) > 1:
        hist = np.asarray(ego_history_xy)
        ax.plot(hist[:, 0], hist[:, 1], color=_style.COLOR_EXECUTED_TRAJ,
                 linewidth=1.6, linestyle="--", zorder=4)

    _style.draw_oriented_box(
        ax, record.ego_x, record.ego_y, record.ego_yaw,
        length=5.0, width=2.0,  # generic vehicle footprint for GIF frames
        color=_style.COLOR_EGO, zorder=6,
    )

    ax.set_xlim(record.ego_x - _HALF_WINDOW_M, record.ego_x + _HALF_WINDOW_M)
    ax.set_ylim(record.ego_y - _HALF_WINDOW_M, record.ego_y + _HALF_WINDOW_M)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")


def _polylines_near(env, x: float, y: float):
    polys = []
    for lane_id, poly in env._polylines_by_id.items():
        if np.min(np.hypot(poly.xy[:, 0] - x, poly.xy[:, 1] - y)) <= _LANE_PAD_M:
            polys.append((lane_id, poly.xy.copy()))
    return polys


def render_rollout_gif(
    episode: PPOEpisodeResult,
    env,
    output_path: str,
    frame_stride: int = 1,
    duration_ms: int = 150,
) -> None:
    """Renders one GIF covering the full recorded episode: road/lane
    geometry, ego, surrounding vehicles, and the real executed ego
    history, with a small per-frame diagnostic overlay (step, PPO
    action, ego speed, d_m, target front/rear TTC, intervention,
    terminal reason on the last frame)."""

    if not episode.steps:
        return

    _style.apply_paper_style()

    frames = []

    selected_indices = set(range(0, len(episode.steps), frame_stride))
    selected_indices.add(len(episode.steps) - 1)  # always include the terminal frame

    history_so_far = []
    for i, record in enumerate(episode.steps):
        history_so_far.append((record.ego_x, record.ego_y))
        if i not in selected_indices:
            continue

        fig, ax = plt.subplots(figsize=(7, 7))
        polylines = _polylines_near(env, record.ego_x, record.ego_y)
        _draw_frame(ax, polylines, record, history_so_far)

        overlay_lines = [
            f"step={record.step_index}  action={record.selected_action}",
            f"type={record.maneuver_type or 'legacy'}  lanes={record.active_source_lane_id}->{record.active_target_lane_id}",
            f"stable_lane={record.current_stable_lane_id}  interaction={'yes' if record.v2_saw_relevant_interaction else 'no'}",
            f"speed={record.ego_speed_mps:.1f} m/s   d_m={record.observation[1]:.1f} m",
            f"front_ttc={record.observation[5]:.1f}s  rear_ttc={record.observation[9]:.1f}s",
            f"intervention={'yes' if record.downstream_status not in (None, 'OK') else 'no'}",
        ]
        if record.terminated or record.truncated:
            overlay_lines.append(f"TERMINAL: {record.termination_reason}")
        ax.text(
            0.02, 0.98, "\n".join(overlay_lines),
            transform=ax.transAxes, fontsize=7, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
        )
        ax.set_title(f"{episode.maneuver_id} -- PPO rollout ({episode.outcome})")
        fig.tight_layout()

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf).convert("RGB"))
        plt.close(fig)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames[0].save(
        output_path, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0,
    )


def render_trace_png(episode: PPOEpisodeResult, output_path: str) -> None:
    """Diagnostic multi-row time-series figure: Behavior Action,
    Ego Speed, Remaining merge distance d_m, Target Front/Rear TTC,
    Target Front/Rear Gap, Reward, intervention -- for Reward-tuning
    analysis ("why did the policy MERGE at this frame?")."""

    if not episode.steps:
        return

    _style.apply_paper_style()

    steps = [r.step_index for r in episode.steps]
    action_idx = [r.action_index for r in episode.steps]
    speed = [r.ego_speed_mps for r in episode.steps]
    d_m = [r.observation[1] for r in episode.steps]
    front_ttc = [r.observation[5] for r in episode.steps]
    rear_ttc = [r.observation[9] for r in episode.steps]
    front_gap = [r.observation[3] for r in episode.steps]
    rear_gap = [r.observation[7] for r in episode.steps]
    reward = [r.reward_total for r in episode.steps]
    intervention = [0 if r.downstream_status in (None, "OK") else 1 for r in episode.steps]
    is_policy = [r.is_policy_step for r in episode.steps]

    fig, axes = plt.subplots(6, 1, figsize=(9, 13), sharex=True)

    ax = axes[0]
    ax.step(steps, action_idx, where="post", color=_style.COLOR_EGO)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["KEEP", "FOLLOW", "MERGE", "STOP"])
    ax.set_ylabel("Action")
    for s, policy_step in zip(steps, is_policy):
        if not policy_step:
            ax.axvspan(s - 0.5, s + 0.5, color="gray", alpha=0.15)

    axes[1].plot(steps, speed, color=_style.COLOR_EGO)
    axes[1].set_ylabel("Ego speed (m/s)")

    axes[2].plot(steps, d_m, color=_style.COLOR_TARGET_LANE)
    axes[2].set_ylabel("d_m (m)")

    axes[3].plot(steps, front_ttc, color=_style.COLOR_TARGET_FRONT, label="Front TTC")
    axes[3].plot(steps, rear_ttc, color=_style.COLOR_TARGET_REAR, label="Rear TTC")
    axes[3].plot(steps, front_gap, color=_style.COLOR_TARGET_FRONT, linestyle="--", label="Front gap (m)")
    axes[3].plot(steps, rear_gap, color=_style.COLOR_TARGET_REAR, linestyle="--", label="Rear gap (m)")
    axes[3].set_ylabel("TTC (s) / Gap (m)")
    axes[3].legend(fontsize=6, ncol=2)

    axes[4].plot(steps, reward, color="black")
    axes[4].set_ylabel("Reward")
    axes[4].axhline(0.0, color="gray", linewidth=0.5)

    axes[5].step(steps, intervention, where="post", color=_style.COLOR_COLLISION)
    axes[5].set_ylabel("Intervention")
    axes[5].set_ylim(-0.1, 1.1)
    axes[5].set_xlabel("Physical step")

    if episode.merge_commit_step is not None:
        for ax in axes:
            ax.axvline(episode.merge_commit_step, color="red", linestyle=":", linewidth=1.0)

    fig.suptitle(f"{episode.maneuver_id} -- PPO trace (outcome={episode.outcome})", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pick_4panel_moments(episode: PPOEpisodeResult):
    """Picks four real moments from the recorded trajectory, matching
    the task's requested semantics, WITHOUT ever inventing a frame that
    does not exist in this episode's real trajectory:

      1. Decision Start        -- step 0
      2. Last WAIT/FOLLOW/KEEP-before-MERGE (or MERGE-1 if none) --
         the step immediately before commitment, if commitment
         happened; otherwise the temporal midpoint of the episode.
      3. MERGE Commit           -- ``episode.merge_commit_step`` if any;
         otherwise the same as moment 2 (no double-count of a
         non-existent commit event).
      4. Terminal Outcome       -- the last recorded step.
    """

    steps = episode.steps
    step0 = steps[0]
    terminal = steps[-1]

    if episode.merge_commit_step is not None:
        commit_idx = episode.merge_commit_step
        commit_record = next(s for s in steps if s.step_index == commit_idx)
        pre_commit_idx = max(step0.step_index, commit_idx - 1)
        pre_commit_record = next(s for s in steps if s.step_index == pre_commit_idx)
    else:
        # No MERGE commitment in this episode (e.g. a STOP/KEEP-driven
        # timeout) -- use the real temporal midpoint for BOTH "last
        # pre-merge" and "commit" moments' slots, rather than
        # fabricating a commitment event that never happened.
        mid_idx = steps[len(steps) // 2].step_index
        pre_commit_record = next(s for s in steps if s.step_index == mid_idx)
        commit_record = pre_commit_record

    return step0, pre_commit_record, commit_record, terminal


def render_rollout_4panel(episode: PPOEpisodeResult, env, output_path: str) -> None:
    """Paper/PPT-style 4-panel rollout figure, reusing
    ``scripts/figures/fig2_rollout_sequence.py``'s real-moment-picking
    + shared-window rendering structure, but driven by this episode's
    REAL PPO-selected actions (never a fixed repeated action) and by
    outcome-appropriate panel labels (Success/Collision/Timeout)."""

    if not episode.steps:
        return

    _style.apply_paper_style()

    moments = _pick_4panel_moments(episode)
    terminal_label = {
        "success": "4. Merge Success",
        "collision": "4. Collision",
        "timeout": "4. Horizon / Timeout",
        "offroad": "4. Offroad",
    }.get(episode.outcome, "4. Terminal Outcome")
    titles = ["1. Decision Start", "2. Pre-Merge / Last Decision", "3. Merge Commit", terminal_label]

    all_y = [s.ego_y for s in episode.steps]
    half_window_m = _HALF_WINDOW_M
    y_center_global = (min(all_y) + max(all_y)) / 2.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    for ax, moment, title in zip(axes, moments, titles):
        polylines = _polylines_near(env, moment.ego_x, moment.ego_y)
        for lane_id, xy in polylines:
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

        hist_x = [s.ego_x for s in episode.steps if s.step_index <= moment.step_index]
        hist_y = [s.ego_y for s in episode.steps if s.step_index <= moment.step_index]
        ax.plot(hist_x, hist_y, color=_style.COLOR_EXECUTED_TRAJ, linewidth=1.6,
                 linestyle="--", zorder=4, label="Ego history" if ax is axes[0] else None)

        _style.draw_oriented_box(ax, moment.ego_x, moment.ego_y, moment.ego_yaw,
                                  length=5.0, width=2.0, color=_style.COLOR_EGO, zorder=6)

        ax.set_xlim(moment.ego_x - half_window_m, moment.ego_x + half_window_m)
        ax.set_ylim(y_center_global - half_window_m, y_center_global + half_window_m)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nstep={moment.step_index} action={moment.selected_action}", fontsize=9)
        ax.set_xlabel("X (m)")
        if ax is axes[0]:
            ax.set_ylabel("Y (m)")

    fig.suptitle(
        f"{episode.maneuver_id} -- real closed-loop PPO rollout (outcome={episode.outcome}); "
        "POST-TRAINING FINAL-CHECKPOINT EVALUATION",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


__all__ = ["render_rollout_gif", "render_trace_png", "render_rollout_4panel"]
