"""Shared plotting style/color conventions for Phase 3 paper+PPT figures.

VISUALIZATION-ONLY module. Never imported by production code; only by
scripts under scripts/figures/. Centralizes one consistent color/label
convention so every figure (Fig 1-7 + failure diagnostics) uses the
same visual language for ego / lanes / agents / trajectories.

All figures generated from this style module must be built from real
data captured via scripts/figures/_rollout.py or by reading the
committed outputs/phase3/robustness/audit_168_maneuvers.{csv,json} --
see the individual figure scripts and FIGURE_INDEX.md for exact
provenance per figure.
"""

import matplotlib

matplotlib.use("Agg")  # headless; figures are always saved via savefig, never shown

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Color / style convention (reused across every figure)
# ---------------------------------------------------------------------------

COLOR_EGO = "#1a1a1a"  # near-black -- the ego vehicle box/history
COLOR_SOURCE_LANE = "#4C72B0"  # blue -- active source lane centerline
COLOR_TARGET_LANE = "#DD8452"  # orange -- active target lane centerline
COLOR_OTHER_LANE = "#B0B0B0"  # light gray -- context lanes, non-active
COLOR_SURROUNDING_AGENT = "#55A868"  # green -- other real agents
COLOR_TARGET_FRONT = "#C44E52"  # red -- labeled Target Front agent
COLOR_TARGET_REAR = "#8172B2"  # purple -- labeled Target Rear agent
COLOR_SOURCE_FRONT = "#CCB974"  # gold -- labeled Source Front agent
COLOR_PLANNED_TRAJ = "#DA8BC3"  # pink/magenta -- Frenet-planner planned trajectory
COLOR_EXECUTED_TRAJ = "#1a1a1a"  # matches ego -- actual closed-loop executed trajectory
COLOR_SUCCESS = "#4C72B0"
COLOR_COLLISION = "#C44E52"
COLOR_TRUNCATION = "#DD8452"
COLOR_OFFROAD = "#8172B2"

# Success/non-success distinction (Figure 5)
COLOR_SUCCESS_EPISODE = "#4C72B0"
COLOR_NONSUCCESS_EPISODE = "#C44E52"

FONT_PAPER = {
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 100,
}

FONT_PPT = {
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "lines.linewidth": 2.5,
    "figure.dpi": 100,
}


def apply_paper_style():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(FONT_PAPER)


def apply_ppt_style():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(FONT_PPT)


def savefig_paper(fig, path_no_ext):
    fig.savefig(f"{path_no_ext}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{path_no_ext}.pdf", bbox_inches="tight")


def savefig_ppt(fig, path_no_ext):
    fig.savefig(f"{path_no_ext}.png", dpi=200, bbox_inches="tight")


def draw_oriented_box(ax, x, y, yaw, length, width, color, alpha=1.0, zorder=5, label=None):
    """Draws an oriented rectangle (vehicle footprint) centered at
    (x, y) with heading yaw (radians), real length/width in meters."""
    import numpy as np
    from matplotlib.patches import Polygon

    hl, hw = length / 2.0, width / 2.0
    corners = np.array([
        [hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    world = corners @ rot.T + np.array([x, y])
    poly = Polygon(world, closed=True, facecolor=color, edgecolor="black",
                    linewidth=0.6, alpha=alpha, zorder=zorder, label=label)
    ax.add_patch(poly)
    # heading tick
    ax.plot([x, x + hl * c], [y, y + hl * s], color="white", linewidth=1.0,
             zorder=zorder + 1)
    return poly
