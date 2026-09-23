"""Manual-review panel for interaction-aware v2 candidates."""

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.scenarios.merge_v2 import InteractionEvidence, TopologyEvidence


def render_v2_review_panel(
    *,
    topology: TopologyEvidence,
    interaction: InteractionEvidence,
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    ego_xy: np.ndarray,
    gap_timeseries: Iterable[dict],
    output_path: str,
    agent_tracks: Optional[Dict[int, np.ndarray]] = None,
    decision: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    """Render geometry, motion, topology, and temporal gaps in one image.

    ``decision``/``reason`` are purely for the figure title -- this
    function never filters by decision; the caller decides which
    decision categories (accept/reject/review) to render."""

    samples = list(gap_timeseries)
    tracks = agent_tracks or {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes[0]
    ax.plot(source_xy[:, 0], source_xy[:, 1], color="tab:blue", label=f"source {topology.source_lane_id}")
    ax.plot(target_xy[:, 0], target_xy[:, 1], color="tab:orange", label=f"target {topology.target_lane_id}")
    ax.plot(ego_xy[:, 0], ego_xy[:, 1], "k--", linewidth=2, label="ego")
    for agent_id, xy in tracks.items():
        ax.plot(xy[:, 0], xy[:, 1], linewidth=1, alpha=0.7, label=f"agent {agent_id}")
    ax.set_aspect("equal")
    ax.legend(fontsize=7)
    ax.set_title("Full topology and trajectories")

    ax = axes[1]
    ax.plot(source_xy[:, 0], source_xy[:, 1], color="tab:blue")
    ax.plot(target_xy[:, 0], target_xy[:, 1], color="tab:orange")
    ax.plot(ego_xy[:, 0], ego_xy[:, 1], "k--", linewidth=2)
    ax.scatter(
        ego_xy[[0, -1], 0], ego_xy[[0, -1], 1],
        c=["green", "red"], s=40, zorder=4,
    )
    ax.set_aspect("equal")
    ax.set_title(
        f"entry/exit; lateral shift={interaction.lateral_displacement_m:.2f} m\n"
        f"target entries={list(topology.target_entry_lane_ids)}"
    )

    ax = axes[2]
    frames = [s["frame"] for s in samples]
    front = [np.nan if s.get("front_gap_m") is None else s["front_gap_m"] for s in samples]
    rear = [np.nan if s.get("rear_gap_m") is None else s["rear_gap_m"] for s in samples]
    ax.plot(frames, front, label="front gap")
    ax.plot(frames, rear, label="rear gap")
    ax.axvline(interaction.commit_frame, color="red", linestyle=":", label="commit/onset")
    ax.axvline(interaction.completion_frame, color="green", linestyle=":", label="completion")
    ax.set_xlabel("logged frame")
    ax.set_ylabel("bumper gap (m)")
    ax.legend(fontsize=7)
    ax.set_title(
        f"interaction IDs={interaction.relevant_vehicle_ids}\n"
        f"gap-order persistence={interaction.gap_order_persistence_frames} frames"
    )

    title = "MERGE v2 manual review: topology + physical transition + interaction"
    if decision is not None:
        title += f"\ndecision={decision.upper()}"
        if reason:
            title += f" ({reason})"
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


__all__ = ["render_v2_review_panel"]
