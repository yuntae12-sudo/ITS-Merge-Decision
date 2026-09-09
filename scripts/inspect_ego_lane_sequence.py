"""Phase 1 CLI: assign ego's logged trajectory to a stable lane sequence.

Example:
    python scripts/inspect_ego_lane_sequence.py --record-index 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios.lane_assignment import (
    assign_ego_lane_sequence,
    compute_stable_lane_sequence,
    find_lane_transitions,
    load_lane_assignment_config,
)
from src.scenarios.lane_geometry import extract_lane_polylines
from src.scenarios.scenario_loader import iter_scenarios, load_dataset_config

DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"
DEFAULT_MERGE_CONFIG = "configs/phase1_merge.yaml"


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Assign ego's full logged trajectory to lanes, apply "
            "temporal filtering, and report stable lane transitions."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_DATASET_CONFIG,
        help="Path to dataset YAML config.",
    )

    parser.add_argument(
        "--merge-config",
        type=str,
        default=DEFAULT_MERGE_CONFIG,
        help="Path to Phase 1 merge/lane-assignment YAML config.",
    )

    parser.add_argument(
        "--record-index",
        type=int,
        default=0,
        help="0-based scenario index to inspect.",
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print every Nth frame (default 1: print all frames).",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("=" * 70)
    print("ITS Merge Decision - Phase 1")
    print("Ego Lane Sequence Inspection")
    print("=" * 70)

    dataset_config = load_dataset_config(args.config)
    lane_assignment_config = load_lane_assignment_config(args.merge_config)

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

    print("\n[2] Lane Assignment Config")
    print(lane_assignment_config)

    # ==================================================================
    # 3. Ego full logged trajectory
    # ==================================================================

    log_trajectory = record.state.log_trajectory
    sdc_index = record.sdc_index

    ego_x = np.asarray(log_trajectory.x[sdc_index])
    ego_y = np.asarray(log_trajectory.y[sdc_index])
    ego_yaw = np.asarray(log_trajectory.yaw[sdc_index])
    ego_valid = np.asarray(log_trajectory.valid[sdc_index]).astype(bool)

    num_frames = ego_x.shape[0]

    print("\n[3] Ego Logged Trajectory")
    print(f"Total frames: {num_frames}")
    print(f"Valid frames: {int(ego_valid.sum())}")

    # ==================================================================
    # 4. Lane geometry
    # ==================================================================

    polylines = extract_lane_polylines(record.state.roadgraph_points)

    print(f"Lane features reconstructed: {len(polylines)}")

    # ==================================================================
    # 5. Raw assignment -> stable sequence
    # ==================================================================

    print("\n" + "-" * 70)
    print("[4] Raw and Stable Lane Assignment")
    print("-" * 70)

    raw_assignments = assign_ego_lane_sequence(
        ego_x, ego_y, ego_yaw, ego_valid, polylines, lane_assignment_config
    )

    stable_sequence = compute_stable_lane_sequence(
        raw_assignments,
        persistence_frames=lane_assignment_config.persistence_frames,
        max_ambiguous_gap_frames=(
            lane_assignment_config.max_ambiguous_gap_frames
        ),
    )

    print(
        f"{'FRAME':>5} "
        f"{'X':>10} "
        f"{'Y':>10} "
        f"{'YAW':>8} "
        f"{'RAW_LANE':>9} "
        f"{'STABLE':>9} "
        f"{'LATERAL':>8} "
        f"{'HEAD_D':>8} "
        f"{'SCORE':>7}"
    )
    print("-" * 70)

    for frame_index in range(num_frames):

        if frame_index % args.print_every != 0:
            continue

        assignment = raw_assignments[frame_index]

        if not assignment.valid:
            print(f"{frame_index:>5} {'INVALID':>52}")
            continue

        lateral = (
            f"{assignment.lateral_distance_m:>8.2f}"
            if assignment.lateral_distance_m is not None
            else f"{'--':>8}"
        )
        heading_diff = (
            f"{np.degrees(assignment.heading_difference_rad):>8.2f}"
            if assignment.heading_difference_rad is not None
            else f"{'--':>8}"
        )
        score = (
            f"{assignment.score:>7.3f}"
            if assignment.score is not None
            else f"{'--':>7}"
        )
        raw_lane_str = (
            str(assignment.lane_id) if assignment.lane_id is not None
            else "None"
        )
        stable_lane_str = (
            str(stable_sequence[frame_index])
            if stable_sequence[frame_index] is not None
            else "None"
        )

        print(
            f"{frame_index:>5} "
            f"{ego_x[frame_index]:>10.2f} "
            f"{ego_y[frame_index]:>10.2f} "
            f"{ego_yaw[frame_index]:>8.3f} "
            f"{raw_lane_str:>9} "
            f"{stable_lane_str:>9} "
            f"{lateral} "
            f"{heading_diff} "
            f"{score}"
        )

    # ==================================================================
    # 6. Transition detection
    # ==================================================================

    print("\n" + "-" * 70)
    print("[5] Stable Lane Transitions")
    print("-" * 70)

    transitions = find_lane_transitions(
        stable_sequence,
        max_bridge_gap_frames=(
            lane_assignment_config.max_ambiguous_gap_frames
        ),
    )

    if not transitions:
        print("No stable lane transition detected")
    else:
        for transition in transitions:
            print(
                f"lane {transition.source_lane_id} -> "
                f"{transition.target_lane_id} "
                f"at frame {transition.transition_frame} "
                f"(source frames {transition.source_start_frame}-"
                f"{transition.source_end_frame}, "
                f"target frames {transition.target_start_frame}-"
                f"{transition.target_end_frame})"
            )

    print("\n" + "=" * 70)
    print("EGO LANE SEQUENCE: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
