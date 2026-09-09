"""Phase 1 CLI: reconstruct lane polylines and summarize ego's surroundings.

Example:
    python scripts/inspect_lane_geometry.py --record-index 0
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios.lane_geometry import (
    extract_lane_polylines,
    nearest_lane_candidates,
    project_point_to_polyline,
)
from src.scenarios.scenario_loader import iter_scenarios, load_dataset_config

DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"
OUTPUT_PATH = Path("outputs/phase1/summaries/lane_geometry.png")


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct lane polylines for one scenario and summarize "
            "the lanes nearest to ego."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_DATASET_CONFIG,
        help="Path to dataset YAML config.",
    )

    parser.add_argument(
        "--record-index",
        type=int,
        default=0,
        help="0-based scenario index to inspect.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest lane candidates to report.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("=" * 70)
    print("ITS Merge Decision - Phase 1")
    print("Lane Geometry Inspection")
    print("=" * 70)

    dataset_config = load_dataset_config(args.config)

    print("\n[1] Loading scenario...")
    print(f"Record index: {args.record_index}")

    record = None

    for candidate in iter_scenarios(
        dataset_config, limit=args.record_index + 1
    ):
        record = candidate

    if record is None:
        raise RuntimeError(
            f"Could not load scenario at record index {args.record_index}."
        )

    print(f"Scene key   : {record.scene_key}")
    print(f"SDC index   : {record.sdc_index}")

    # ==================================================================
    # 2. Ego current position
    # ==================================================================

    current = record.state.current_sim_trajectory

    ego_x = float(np.asarray(current.x[record.sdc_index, 0]))
    ego_y = float(np.asarray(current.y[record.sdc_index, 0]))
    ego_yaw = float(np.asarray(current.yaw[record.sdc_index, 0]))

    print("\n[2] Ego current position")
    print(f"x={ego_x:.3f} m, y={ego_y:.3f} m, yaw={ego_yaw:.3f} rad")

    # ==================================================================
    # 3. Lane polyline reconstruction
    # ==================================================================

    print("\n" + "-" * 70)
    print("[3] Lane Polyline Reconstruction")
    print("-" * 70)

    polylines = extract_lane_polylines(record.state.roadgraph_points)

    print(f"Lane features reconstructed: {len(polylines)}")

    point_counts = [polyline.xy.shape[0] for polyline in polylines]

    if point_counts:
        print(
            f"Points per lane: min={min(point_counts)} "
            f"max={max(point_counts)} "
            f"mean={sum(point_counts) / len(point_counts):.1f}"
        )

    # ==================================================================
    # 4. Nearest lanes to ego
    # ==================================================================

    print("\n" + "-" * 70)
    print(f"[4] Nearest {args.top_k} Lanes to Ego")
    print("-" * 70)

    nearest = nearest_lane_candidates(
        polylines, ego_x, ego_y, k=args.top_k
    )

    lane_by_id = {polyline.lane_id: polyline for polyline in polylines}

    print(
        f"{'LANE_ID':>8} "
        f"{'TYPE':>6} "
        f"{'MIN_DIST':>10} "
        f"{'ARC_LEN':>10} "
        f"{'LATERAL':>10} "
        f"{'HEADING':>10} "
        f"{'NUM_PTS':>8}"
    )

    print("-" * 70)

    for lane_id, min_distance in nearest:

        polyline = lane_by_id[lane_id]
        projection = project_point_to_polyline(polyline, ego_x, ego_y)

        print(
            f"{lane_id:>8} "
            f"{polyline.lane_type:>6} "
            f"{min_distance:>10.2f} "
            f"{projection['arc_length_m']:>10.2f} "
            f"{projection['lateral_distance_m']:>10.2f} "
            f"{projection['heading_rad']:>10.3f} "
            f"{polyline.xy.shape[0]:>8}"
        )

    if not nearest:
        raise RuntimeError("No lane candidates found near ego.")

    print("\n" + "=" * 70)
    print("LANE GEOMETRY RECONSTRUCTION: PASS")
    print("=" * 70)

    # ==================================================================
    # 4-1. Ego heading vs. nearby-lane heading sanity check
    # ==================================================================
    #
    # This does not decide lane assignment (that is Commit C's job) --
    # a scene can have several nearby lanes with unrelated headings
    # (opposing traffic, a crossing street at an intersection), so the
    # *nearest* lane by distance alone is not necessarily ego's own
    # lane. This check instead looks for a heading-consistent lane
    # among the nearby candidates: if at least one nearby lane's
    # reconstructed travel direction matches ego's heading, that is
    # strong evidence direction alignment (Commit fix in this module)
    # is working -- a systematic ~180 degree flip would instead show
    # every nearby lane's heading mismatching ego.

    print("\n" + "-" * 70)
    print("[4-1] Ego Heading vs. Nearby-Lane Heading (sanity check)")
    print("-" * 70)

    print(f"Ego yaw : {ego_yaw:.3f} rad")
    print()

    best_lane_id, best_heading_diff = None, np.pi

    for lane_id, distance in nearest:

        polyline = lane_by_id[lane_id]
        projection = project_point_to_polyline(polyline, ego_x, ego_y)

        heading_diff = (
            projection["heading_rad"] - ego_yaw + np.pi
        ) % (2 * np.pi) - np.pi

        print(
            f"  lane={lane_id:>6} distance={distance:>6.2f} m "
            f"heading={projection['heading_rad']:>7.3f} rad "
            f"heading_diff={heading_diff:>7.3f} rad"
        )

        if abs(heading_diff) < abs(best_heading_diff):
            best_lane_id, best_heading_diff = lane_id, heading_diff

    print(
        f"\nClosest heading match: lane {best_lane_id} "
        f"(heading_diff={best_heading_diff:.3f} rad)"
    )

    if abs(best_heading_diff) > (np.pi / 2):
        print(
            "WARNING: even the best-heading-matching nearby lane "
            "differs from ego by more than 90 degrees -- reconstructed "
            "lane direction may be flipped relative to ego travel "
            "direction."
        )
    else:
        print(
            "Heading alignment looks consistent: at least one nearby "
            "lane's reconstructed direction matches ego (no systematic "
            "~180 degree flip)."
        )

    # ==================================================================
    # 5. Visualization
    # ==================================================================

    print("\n" + "-" * 70)
    print("[5] Visualization")
    print("-" * 70)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    for polyline in polylines:
        ax.plot(
            polyline.xy[:, 0],
            polyline.xy[:, 1],
            color="lightgray",
            linewidth=1,
        )

    nearest_ids = {lane_id for lane_id, _ in nearest}

    for lane_id in nearest_ids:
        polyline = lane_by_id[lane_id]
        ax.plot(
            polyline.xy[:, 0],
            polyline.xy[:, 1],
            linewidth=2,
        )
        ax.annotate(
            str(lane_id),
            (polyline.xy[0, 0], polyline.xy[0, 1]),
            fontsize=8,
        )

    ax.scatter([ego_x], [ego_y], color="red", marker="*", s=150, zorder=5)
    ax.set_aspect("equal")
    ax.set_title(f"Lane geometry near ego - {record.scene_key}")

    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")

    print("\n" + "=" * 70)
    print("PHASE 1 COMMIT B VALIDATION")
    print("=" * 70)
    print("[PASS] Lane polyline reconstruction")
    print("[PASS] Nearest-lane summary")
    print("[PASS] Visualization")
    print("=" * 70)


if __name__ == "__main__":
    main()
