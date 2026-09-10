"""Phase 1 CLI: render manual-validation PNGs for merge candidates.

Reads ``merge_candidates.csv``, filters to the requested decisions (and
optionally one record_index / source_shard / a per-decision cap),
groups by (source_split, source_shard, record_index) so each PHYSICAL
scene is only loaded once (and never conflated with a same-numbered
record_index from a different shard), recomputes the live
transition/diagnostic objects needed for rendering (the CSV only
carries flattened diagnostic scalars, not ``LanePolyline`` objects) and
asserts the recomputed decision matches the CSV row's decision (catches
any drift between the CSV and a re-run of the exact same detector
code).

Shard-aware selection (``--record-index``/``--source-shard``): if
``--record-index`` is given and the candidates CSV contains multiple
distinct ``source_shard`` values with that record_index, and
``--source-shard`` was NOT also given, this is treated as an ambiguous
selection -- rather than silently picking one shard, ALL matching
shards are rendered, and the shards included are printed so the
ambiguity is visible. Passing ``--source-shard`` disambiguates by
filtering to exactly that one shard.

Example:
    python scripts/render_merge_validation.py --decisions accept,review
    python scripts/render_merge_validation.py --record-index 28
    python scripts/render_merge_validation.py --record-index 28 --source-shard validation_tfexample.tfrecord-00000-of-00150
    python scripts/render_merge_validation.py --decisions reject --limit-per-decision 3
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.scenarios.dataset_builder import (
    read_candidates_csv,
    sanitize_candidate_id_for_filename,
)
from src.scenarios.lane_assignment import (
    assign_ego_lane_sequence,
    compute_stable_lane_sequence,
    find_lane_transitions,
    load_lane_assignment_config,
)
from src.scenarios.lane_geometry import (
    ALL_LANE_TYPE_IDS,
    extract_lane_polylines,
    project_point_to_polyline,
)
from src.scenarios.merge_detector import (
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
    resolve_physical_shard,
)
from src.scenarios.validation_viz import render_candidate_figure

DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"
DEFAULT_PHASE1_CONFIG = "configs/phase1_merge.yaml"
DEFAULT_CANDIDATES = "data/manifests/merge_candidates.csv"
DEFAULT_OUTPUT_DIR = "outputs/phase1/merge_validation"


def parse_args():

    parser = argparse.ArgumentParser(
        description="Render manual-validation PNGs for merge candidates."
    )

    parser.add_argument(
        "--dataset-config", type=str, default=DEFAULT_DATASET_CONFIG
    )
    parser.add_argument(
        "--phase1-config", type=str, default=DEFAULT_PHASE1_CONFIG
    )
    parser.add_argument("--candidates", type=str, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--decisions", type=str, default="accept,review")
    parser.add_argument("--limit-per-decision", type=int, default=None)
    parser.add_argument("--record-index", type=int, default=None)
    parser.add_argument(
        "--source-shard", type=str, default=None,
        help=(
            "Disambiguates --record-index when multiple shards share "
            "that record_index. Without it in an ambiguous case, ALL "
            "matching shards are rendered (see module docstring)."
        ),
    )
    parser.add_argument("--viewport-radius", type=float, default=75.0)

    return parser.parse_args()


def select_candidate_rows(rows, record_index=None, source_shard=None):
    """Applies the --record-index/--source-shard filtering rules.

    Returns:
        (filtered_rows, ambiguous_shards): ``ambiguous_shards`` is a
        non-empty sorted list of distinct source_shard values only
        when ``record_index`` was given, ``source_shard`` was NOT
        given, and more than one distinct source_shard matched that
        record_index (i.e. the caller should print a disambiguation
        notice) -- otherwise an empty list.
    """

    if record_index is None:
        return rows, []

    matching = [r for r in rows if int(r["record_index"]) == record_index]

    if source_shard is not None:
        return [r for r in matching if r["source_shard"] == source_shard], []

    distinct_shards = sorted({r["source_shard"] for r in matching})
    ambiguous_shards = distinct_shards if len(distinct_shards) > 1 else []
    return matching, ambiguous_shards


def _row_to_types(row):
    """Casts a raw CSV row's numeric fields to int for matching."""

    return {
        "record_index": int(row["record_index"]),
        "transition_frame": int(row["transition_frame"]),
        "source_lane_id": int(row["source_lane_id"]),
        "target_lane_id": int(row["target_lane_id"]),
        "decision": row["decision"],
    }


def main():

    args = parse_args()
    decisions = {d.strip().lower() for d in args.decisions.split(",") if d.strip()}

    print("=" * 70)
    print("ITS Merge Decision - Phase 1")
    print("Render Merge Validation Figures")
    print("=" * 70)

    rows = read_candidates_csv(args.candidates)

    decision_filtered = [row for row in rows if row["decision"] in decisions]

    filtered, ambiguous_shards = select_candidate_rows(
        decision_filtered,
        record_index=args.record_index,
        source_shard=args.source_shard,
    )

    if ambiguous_shards:
        print(
            f"\nWARN: --record-index={args.record_index} matches "
            f"{len(ambiguous_shards)} distinct source_shard values and "
            "--source-shard was not given -- rendering ALL matching "
            "shards (pass --source-shard to disambiguate):"
        )
        for shard in ambiguous_shards:
            print(f"  {shard}")

    if args.limit_per_decision is not None:
        capped = []
        counts = defaultdict(int)
        for row in filtered:
            if counts[row["decision"]] >= args.limit_per_decision:
                continue
            counts[row["decision"]] += 1
            capped.append(row)
        filtered = capped

    print(f"\nCandidates matched   : {len(filtered)}")

    # Grouped by (source_split, source_shard, record_index) -- NEVER by
    # record_index alone, since two different physical shards can
    # legitimately share the same record_index.
    by_shard_record = defaultdict(list)
    for row in filtered:
        key = (row["source_split"], row["source_shard"], int(row["record_index"]))
        by_shard_record[key].append(row)

    by_shard = defaultdict(dict)
    for (source_split, source_shard, record_index), record_rows in by_shard_record.items():
        by_shard[(source_split, source_shard)][record_index] = record_rows

    expansion_config = load_dataset_config(args.dataset_config)
    lane_assignment_config = load_lane_assignment_config(args.phase1_config)
    merge_topology_config = load_merge_topology_config(args.phase1_config)
    agent_selection_config = load_agent_selection_config(args.phase1_config)

    output_dir = Path(args.output_dir)
    rendered_counts = defaultdict(int)

    for (source_split, source_shard), rows_by_record_index in by_shard.items():

        # Split-mismatch check fires FIRST, before any Waymax config is
        # built or any file is touched -- rendering a row whose
        # source_split disagrees with the supplied dataset config's
        # split is a hard failure, not a skip-with-warning.
        shard_path = resolve_physical_shard(
            expansion_config, source_shard=source_shard, source_split=source_split
        )

        shard_waymax_config = build_waymax_config(expansion_config, shard_path)
        max_record_index = max(rows_by_record_index.keys())

        for record in iter_scenarios(
            shard_waymax_config,
            limit=max_record_index + 1,
            source_dataset=expansion_config.dataset_name,
            source_split=source_split,
        ):

            record_rows = rows_by_record_index.get(record.record_index)
            if not record_rows:
                continue

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

            vehicle_polylines = extract_lane_polylines(record.state.roadgraph_points)
            all_polylines = extract_lane_polylines(
                record.state.roadgraph_points, lane_type_ids=ALL_LANE_TYPE_IDS
            )
            lane_by_id = {p.lane_id: p for p in vehicle_polylines}

            raw_assignments = assign_ego_lane_sequence(
                ego_x, ego_y, ego_yaw, ego_valid, vehicle_polylines, lane_assignment_config
            )
            stable_sequence = compute_stable_lane_sequence(
                raw_assignments,
                persistence_frames=lane_assignment_config.persistence_frames,
                max_ambiguous_gap_frames=lane_assignment_config.max_ambiguous_gap_frames,
            )
            transitions = find_lane_transitions(
                stable_sequence,
                max_bridge_gap_frames=lane_assignment_config.max_ambiguous_gap_frames,
            )

            transitions_by_key = {
                (t.transition_frame, t.source_lane_id, t.target_lane_id): t
                for t in transitions
            }

            for row in record_rows:

                key = (
                    int(row["transition_frame"]),
                    int(row["source_lane_id"]),
                    int(row["target_lane_id"]),
                )
                transition = transitions_by_key.get(key)
                if transition is None:
                    print(
                        f"  WARN: could not re-locate transition for "
                        f"candidate_id={row['candidate_id']} -- skipping."
                    )
                    continue

                source_polyline = lane_by_id.get(transition.source_lane_id)
                target_polyline = lane_by_id.get(transition.target_lane_id)

                frame = transition.transition_frame
                ego_source_arc_length = None
                if source_polyline is not None:
                    projection = project_point_to_polyline(
                        source_polyline, float(ego_x[frame]), float(ego_y[frame])
                    )
                    ego_source_arc_length = projection["arc_length_m"]

                diagnostic = detect_merge(
                    transition,
                    source_polyline,
                    target_polyline,
                    ego_source_arc_length,
                    merge_topology_config,
                )

                assert diagnostic.decision.value == row["decision"], (
                    f"Recomputed decision {diagnostic.decision.value!r} does not "
                    f"match CSV row decision {row['decision']!r} for "
                    f"candidate_id={row['candidate_id']} -- detector drift "
                    "detected."
                )

                # Build a lightweight namespace matching what
                # render_candidate_figure expects from a CandidateRecord
                # (only the fields it actually reads).
                from types import SimpleNamespace

                merge_start_s = merge_end_s = None
                front_id = front_gap = front_ttc = None
                rear_id = rear_gap = rear_ttc = None

                if diagnostic.decision.value == "accept":
                    merge_start_s, merge_end_s = compute_merge_start_end_s(
                        source_polyline, target_polyline, merge_topology_config
                    )
                    d_m = compute_remaining_merge_distance(
                        merge_end_s, ego_source_arc_length
                    )
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
                        vel_x=np.asarray(
                            record.state.log_trajectory.vel_x[:, frame]
                        ),
                        vel_y=np.asarray(
                            record.state.log_trajectory.vel_y[:, frame]
                        ),
                        length=np.asarray(
                            record.state.log_trajectory.length[:, frame]
                        ),
                        config=agent_selection_config,
                    )
                    front_id = features.front_vehicle_id
                    front_gap = features.front_gap_m
                    front_ttc = features.front_ttc_s
                    rear_id = features.rear_vehicle_id
                    rear_gap = features.rear_gap_m
                    rear_ttc = features.rear_ttc_s

                candidate_ns = SimpleNamespace(
                    candidate_id=row["candidate_id"],
                    scene_key=row["scene_key"],
                    transition_frame=transition.transition_frame,
                    source_lane_id=transition.source_lane_id,
                    target_lane_id=transition.target_lane_id,
                    decision=diagnostic.decision.value,
                    reason=diagnostic.reason,
                    endpoint_target_distance_m=diagnostic.endpoint_target_distance_m,
                    heading_difference_deg=diagnostic.heading_difference_deg,
                    separation_reduction_m=diagnostic.separation_reduction_m,
                    decreasing_fraction=diagnostic.decreasing_fraction,
                    source_remaining_distance_m=diagnostic.source_remaining_distance_m,
                    target_lane_persistent=diagnostic.target_lane_persistent,
                    merge_start_s=merge_start_s,
                    merge_end_s=merge_end_s,
                    front_vehicle_id=front_id,
                    front_gap_m=front_gap,
                    front_ttc_s=front_ttc,
                    rear_vehicle_id=rear_id,
                    rear_gap_m=rear_gap,
                    rear_ttc_s=rear_ttc,
                )

                decision_dir = output_dir / diagnostic.decision.value
                filename = sanitize_candidate_id_for_filename(row["candidate_id"]) + ".png"
                output_path = decision_dir / filename

                render_candidate_figure(
                    record,
                    candidate_ns,
                    source_polyline,
                    target_polyline,
                    all_polylines,
                    None,
                    output_path,
                    viewport_radius_m=args.viewport_radius,
                )

                rendered_counts[diagnostic.decision.value] += 1

    print("\nRendered counts per decision:")
    for decision, count in sorted(rendered_counts.items()):
        print(f"  {decision}: {count}  -> {output_dir / decision}")

    print("\n" + "=" * 70)
    print("RENDER MERGE VALIDATION: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
