"""Phase 1 CLI: run the merge detector over a scenario's stable
lane transitions and print detailed diagnostics.

For each stable A->B transition (Commit C), classifies it as ACCEPT
(confident merge candidate), REJECT (confidently not a merge), or
REVIEW (available geometry cannot confidently distinguish a true merge
from a serial map-segment continuation -- see merge_detector.py's
module docstring). ACCEPTed transitions get target-lane interaction
features (Front/Rear gap, relative speed, TTC, d_m, traffic density).

Example:
    python scripts/inspect_merge_candidate.py --record-index 0
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
from src.scenarios.lane_geometry import extract_lane_polylines, project_point_to_polyline
from src.scenarios.merge_detector import (
    MergeDecision,
    compute_merge_start_end_s,
    compute_remaining_merge_distance,
    detect_merge,
    load_merge_topology_config,
)
from src.scenarios.scenario_features import (
    extract_interaction_features,
    load_agent_selection_config,
)
from src.scenarios.scenario_loader import (
    build_waymax_config,
    iter_scenarios,
    load_dataset_config,
    select_single_shard_for_inspection,
)

DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"
DEFAULT_MERGE_CONFIG = "configs/phase1_merge.yaml"


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Classify a scenario's stable lane transitions as merge "
            "candidates and extract interaction features for accepted "
            "ones."
        )
    )

    parser.add_argument(
        "--config", type=str, default=DEFAULT_DATASET_CONFIG,
        help="Path to dataset YAML config.",
    )
    parser.add_argument(
        "--merge-config", type=str, default=DEFAULT_MERGE_CONFIG,
        help="Path to Phase 1 merge/lane-assignment YAML config.",
    )
    parser.add_argument(
        "--record-index", type=int, default=0,
        help="0-based scenario index to inspect.",
    )
    parser.add_argument(
        "--source-shard", type=str, default=None,
        help=(
            "Basename of the physical shard to inspect. Required when "
            "the dataset config resolves to multiple shards; optional "
            "(and unused) when it resolves to exactly one."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("=" * 70)
    print("ITS Merge Decision - Phase 1")
    print("Merge Candidate Inspection")
    print("=" * 70)

    expansion_config = load_dataset_config(args.config)
    lane_assignment_config = load_lane_assignment_config(args.merge_config)
    merge_topology_config = load_merge_topology_config(args.merge_config)
    agent_selection_config = load_agent_selection_config(args.merge_config)

    # Multi-shard scanning is handled by extract_merge_scenes.py's
    # iter_all_shards; this CLI inspects exactly one physical shard,
    # selected via the shared --source-shard/--record-index policy
    # (see select_single_shard_for_inspection).
    shard_path = select_single_shard_for_inspection(
        expansion_config,
        source_shard=args.source_shard,
        record_index=args.record_index,
    )
    dataset_config = build_waymax_config(expansion_config, shard_path)

    print(f"Source shard: {Path(shard_path).name}")

    record = None
    for candidate in iter_scenarios(
        dataset_config,
        limit=args.record_index + 1,
        source_dataset=expansion_config.dataset_name,
        source_split=expansion_config.split,
    ):
        record = candidate

    if record is None:
        raise RuntimeError(
            f"Could not load scenario at record index {args.record_index}."
        )

    print(f"\nScenario: {record.scene_key}")
    print(f"SDC index: {record.sdc_index}")

    log_trajectory = record.state.log_trajectory
    sdc_index = record.sdc_index

    ego_x = np.asarray(log_trajectory.x[sdc_index])
    ego_y = np.asarray(log_trajectory.y[sdc_index])
    ego_yaw = np.asarray(log_trajectory.yaw[sdc_index])
    ego_valid = np.asarray(log_trajectory.valid[sdc_index]).astype(bool)
    ego_vel_x = np.asarray(log_trajectory.vel_x[sdc_index])
    ego_vel_y = np.asarray(log_trajectory.vel_y[sdc_index])
    ego_length = np.asarray(log_trajectory.length[sdc_index])

    object_ids = np.asarray(record.state.object_metadata.ids)
    object_types = np.asarray(record.state.object_metadata.object_types)

    polylines = extract_lane_polylines(record.state.roadgraph_points)
    lane_by_id = {polyline.lane_id: polyline for polyline in polylines}

    raw_assignments = assign_ego_lane_sequence(
        ego_x,
        ego_y,
        ego_yaw,
        ego_valid,
        polylines,
        lane_assignment_config,
    )

    stable_sequence = compute_stable_lane_sequence(
        raw_assignments,
        persistence_frames=lane_assignment_config.persistence_frames,
        max_ambiguous_gap_frames=(
            lane_assignment_config.max_ambiguous_gap_frames
        ),
    )

    transitions = find_lane_transitions(
        stable_sequence,
        max_bridge_gap_frames=(
            lane_assignment_config.max_ambiguous_gap_frames
        ),
    )

    print(f"Stable transitions found: {len(transitions)}")

    if not transitions:
        print("\nNo stable lane transitions in this scenario.")
        print("\n" + "=" * 70)
        print("MERGE CANDIDATE INSPECTION: PASS (0 transitions)")
        print("=" * 70)
        return

    decision_counts = {
        MergeDecision.ACCEPT: 0,
        MergeDecision.REJECT: 0,
        MergeDecision.REVIEW: 0,
    }
    reason_counts = {}

    for index, transition in enumerate(transitions):

        print("\n" + "-" * 70)
        print(f"Transition {index + 1}")
        print("-" * 70)
        print(f"  source_lane       : {transition.source_lane_id}")
        print(f"  target_lane       : {transition.target_lane_id}")
        print(f"  transition_frame  : {transition.transition_frame}")

        source_polyline = lane_by_id.get(transition.source_lane_id)
        target_polyline = lane_by_id.get(transition.target_lane_id)

        ego_source_arc_length = None
        if source_polyline is not None:
            frame = transition.transition_frame
            projection = project_point_to_polyline(
                source_polyline,
                float(ego_x[frame]),
                float(ego_y[frame]),
            )
            ego_source_arc_length = projection["arc_length_m"]

        diagnostic = detect_merge(
            transition,
            source_polyline,
            target_polyline,
            ego_source_arc_length,
            merge_topology_config,
        )

        print(
            f"  source_remaining_at_transition: "
            f"{diagnostic.source_remaining_distance_m}"
        )
        print(
            f"  endpoint_target_distance : "
            f"{diagnostic.endpoint_target_distance_m}"
        )
        print(f"  heading_difference_deg   : {diagnostic.heading_difference_deg}")
        print(
            f"  separation_reduction_m   : "
            f"{diagnostic.separation_reduction_m}"
        )
        print(f"  decreasing_fraction      : {diagnostic.decreasing_fraction}")
        print(f"  lanes_converge           : {diagnostic.lanes_converge}")
        print(
            f"  max_collinear_offset_m   : "
            f"{diagnostic.max_collinear_offset_m} (diagnostic-only)"
        )
        print(
            f"  upstream_separation_m    : "
            f"{diagnostic.upstream_separation_m} (diagnostic-only)"
        )
        print(
            f"  parallel_continuation    : "
            f"{diagnostic.parallel_continuation}"
        )
        print(f"  source_lane_ends         : {diagnostic.source_lane_ends}")
        print(
            f"  target_persistent        : "
            f"{diagnostic.target_lane_persistent}"
        )

        decision_counts[diagnostic.decision] += 1
        reason_counts[diagnostic.reason] = (
            reason_counts.get(diagnostic.reason, 0) + 1
        )

        print(f"\n  RESULT: {diagnostic.decision.value.upper()}")

        if diagnostic.reason is not None:
            print(f"  reason: {diagnostic.reason}")

        if diagnostic.decision != MergeDecision.ACCEPT:
            continue

        merge_start_s, merge_end_s = compute_merge_start_end_s(
            source_polyline, target_polyline, merge_topology_config
        )

        d_m = compute_remaining_merge_distance(
            merge_end_s, ego_source_arc_length
        )

        frame = transition.transition_frame

        features = extract_interaction_features(
            frame_index=frame,
            target_polyline=target_polyline,
            merge_distance_m=d_m,
            ego_id=record.sdc_id,
            ego_x=float(ego_x[frame]),
            ego_y=float(ego_y[frame]),
            ego_vel_x=float(ego_vel_x[frame]),
            ego_vel_y=float(ego_vel_y[frame]),
            ego_length_m=float(ego_length[frame]),
            object_ids=object_ids,
            object_types=object_types,
            valid=np.asarray(
                record.state.log_trajectory.valid[:, frame]
            ).astype(bool),
            x=np.asarray(record.state.log_trajectory.x[:, frame]),
            y=np.asarray(record.state.log_trajectory.y[:, frame]),
            yaw=np.asarray(record.state.log_trajectory.yaw[:, frame]),
            vel_x=np.asarray(record.state.log_trajectory.vel_x[:, frame]),
            vel_y=np.asarray(record.state.log_trajectory.vel_y[:, frame]),
            length=np.asarray(record.state.log_trajectory.length[:, frame]),
            config=agent_selection_config,
        )

        print(f"\n  merge_start_s : {merge_start_s:.2f}")
        print(f"  merge_end_s   : {merge_end_s:.2f}")
        print(f"  d_m           : {d_m:.2f}")
        print(
            f"  ego_longitudinal_speed: "
            f"{features.ego_longitudinal_speed_mps:.2f}"
        )
        print(f"  front_id      : {features.front_vehicle_id}")
        print(f"  front_gap     : {features.front_gap_m}")
        print(f"  front_rel_speed(dv_f): {features.front_relative_speed_mps}")
        print(f"  front_ttc     : {features.front_ttc_s}")
        print(f"  rear_id       : {features.rear_vehicle_id}")
        print(f"  rear_gap      : {features.rear_gap_m}")
        print(f"  rear_rel_speed(dv_r): {features.rear_relative_speed_mps}")
        print(f"  rear_ttc      : {features.rear_ttc_s}")
        print(f"  traffic_density: {features.traffic_density}")

    print("\n" + "=" * 70)
    print("Decision Summary")
    print("=" * 70)
    print(f"  ACCEPT: {decision_counts[MergeDecision.ACCEPT]}")
    print(f"  REJECT: {decision_counts[MergeDecision.REJECT]}")
    print(f"  REVIEW: {decision_counts[MergeDecision.REVIEW]}")
    print(f"  TOTAL : {len(transitions)}")

    assert sum(decision_counts.values()) == len(transitions), (
        "decision counts must sum to total transitions inspected"
    )

    print("\n  Reason histogram:")
    for reason, count in sorted(
        reason_counts.items(), key=lambda item: (-item[1], str(item[0]))
    ):
        print(f"    {reason}: {count}")

    print("\n" + "=" * 70)
    print(
        f"MERGE CANDIDATE INSPECTION: PASS "
        f"({decision_counts[MergeDecision.ACCEPT]} accepted / "
        f"{len(transitions)} transitions)"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
