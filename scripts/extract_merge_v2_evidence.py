#!/usr/bin/env python3
"""Join broad trajectory transitions with WOMD protobuf topology.

Output is an evidence JSONL review queue. It is not canonical until each
automatic ACCEPT has been rendered/reviewed and marked CONFIRMED_MERGE.
"""

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios.dataset_builder import reconstruct_transition
from src.scenarios.interaction_evidence import extract_temporal_interaction_evidence
from src.scenarios.lane_assignment import load_lane_assignment_config
from src.scenarios.lane_geometry import project_point_to_polyline
from src.scenarios.merge_v2 import classify_v2_merge
from src.scenarios.scenario_features import load_agent_selection_config
from src.scenarios.scenario_loader import (
    build_waymax_config,
    iter_scenarios,
    load_dataset_config,
    resolve_physical_shard,
)
from src.scenarios.scenario_proto_loader import (
    index_scenario_topologies_for_ids,
    iter_tfexample_scenario_ids,
)
from src.scenarios.v2_review_viz import render_v2_review_panel
import numpy as np


class _NumpyJSONEncoder(json.JSONEncoder):
    """Evidence values pass through numpy-backed geometry/interaction
    computations (e.g. ``project_point_to_polyline``, ``extract_
    interaction_features``) and can carry numpy scalar/bool types that
    the stdlib ``json`` module rejects; convert them to native Python
    types without changing any numeric value."""

    def default(self, o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-config", required=True)
    p.add_argument("--scenario-proto", nargs="+", required=True)
    p.add_argument("--transitions-csv", required=True)
    p.add_argument("--phase1-config", default="configs/phase1_merge.yaml")
    p.add_argument("--manual-labels", default=None)
    p.add_argument("--output", default="data/manifests/v2/evidence_review_queue.jsonl")
    p.add_argument("--review-dir", default="outputs/merge_v2_review")
    return p.parse_args(argv)


def _labels(path):
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["candidate_id"]: r for r in csv.DictReader(f)}


def _build_scenario_id_lookup(transition_rows, expansion):
    """One sequential pass per distinct tf_example shard, instead of
    rescanning from record 0 for every candidate row (was O(n^2) in the
    number of transition rows per shard). Returns
    ``{shard_path: {record_index: scenario_id}}`` -- cheap: only reads
    the ``scenario/id`` feature, never the full trajectory/map payload,
    and only from the local tf_example shards this run actually needs
    (never touches ``--scenario-proto``)."""

    wanted_by_shard = {}
    for row in transition_rows:
        shard_path = resolve_physical_shard(
            expansion, row["source_shard"], source_split=row.get("source_split")
        )
        wanted_by_shard.setdefault(shard_path, set()).add(int(row["record_index"]))

    resolved_by_shard = {}
    for shard_path, wanted_indices in wanted_by_shard.items():
        resolved_by_shard[shard_path] = _scenario_ids_for_shard(shard_path, wanted_indices)
    return resolved_by_shard


def _scenario_ids_for_shard(shard_path, wanted_indices):
    """Single sequential pass over one tf_example shard, reading only
    the ``scenario/id`` feature for the requested record indices."""

    resolved = {}
    max_wanted = max(wanted_indices)
    for record_index, scenario_id in iter_tfexample_scenario_ids(shard_path):
        if record_index in wanted_indices:
            resolved[record_index] = scenario_id
        if record_index >= max_wanted:
            break
    return resolved


def _float(row, key, default):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def main(argv=None):
    args = parse_args(argv)
    expansion = load_dataset_config(args.dataset_config)
    lane_config = load_lane_assignment_config(args.phase1_config)
    agent_config = load_agent_selection_config(args.phase1_config)
    labels = _labels(args.manual_labels)

    with open(args.transitions_csv, newline="", encoding="utf-8") as f:
        transition_rows = list(csv.DictReader(f))

    # Cheap pass first: read only ``scenario/id`` from the local
    # tf_example shards this run needs (never the full trajectory/map
    # payload, never --scenario-proto). This bounds the expensive
    # protobuf-topology scan to only the ids actually wanted, instead of
    # loading every one of the (potentially ~1000-shard) --scenario-proto
    # split's topologies into memory at once.
    scenario_id_by_shard = _build_scenario_id_lookup(transition_rows, expansion)
    wanted_scenario_ids = {
        scenario_id
        for by_record in scenario_id_by_shard.values()
        for scenario_id in by_record.values()
    }
    topology_index = index_scenario_topologies_for_ids(args.scenario_proto, wanted_scenario_ids)

    skipped_no_topology = []
    for shard_path, by_record in scenario_id_by_shard.items():
        for record_index, scenario_id in by_record.items():
            if scenario_id not in topology_index:
                skipped_no_topology.append((shard_path, record_index, scenario_id))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as out:
        for row in transition_rows:
            shard_path = resolve_physical_shard(
                expansion, row["source_shard"], source_split=row.get("source_split")
            )
            record_index = int(row["record_index"])
            scenario_id = scenario_id_by_shard[shard_path][record_index]
            topology = topology_index.get(scenario_id)
            if topology is None:
                continue

            dataset_config = build_waymax_config(expansion, shard_path)
            records = list(iter_scenarios(
                dataset_config, start_index=record_index, limit=1,
                source_dataset=expansion.dataset_name, source_split=expansion.split,
            ))
            if len(records) != 1:
                raise RuntimeError(f"Could not load {row['candidate_id']}")
            record = records[0]
            transition, source, target, _ = reconstruct_transition(
                record, lane_config,
                int(row["transition_frame"]), int(row["source_lane_id"]),
                int(row["target_lane_id"]), row["candidate_id"],
            )

            source_start_on_target = project_point_to_polyline(
                target, float(source.xy[0, 0]), float(source.xy[0, 1])
            )["arc_length_m"]
            source_end_on_target = project_point_to_polyline(
                target, float(source.xy[-1, 0]), float(source.xy[-1, 1])
            )["arc_length_m"]
            overlaps = bool(
                source_start_on_target > 0.5
                and source_end_on_target < target.arc_length[-1] - 0.5
            )
            topology_evidence = topology.evidence(
                transition.source_lane_id,
                transition.target_lane_id,
                lanes_overlap_longitudinally=overlaps,
                terminal_heading_difference_deg=_float(row, "heading_difference_deg", 180.0),
                terminal_collinear_offset_m=_float(row, "max_collinear_offset_m", float("inf")),
                endpoint_gap_m=_float(row, "endpoint_target_distance_m", float("inf")),
            )

            decision_start = int(float(row.get("merge_start_frame") or transition.source_start_frame))
            completion = int(float(row.get("merge_complete_frame") or transition.target_start_frame))
            interaction, gap_samples = extract_temporal_interaction_evidence(
                record=record,
                source_polyline=source,
                target_polyline=target,
                decision_start_frame=decision_start,
                transition_frame=transition.transition_frame,
                completion_frame=completion,
                target_end_frame=transition.target_end_frame,
                source_occupancy_frames=(transition.source_end_frame - transition.source_start_frame + 1),
                target_stable_frames=(transition.target_end_frame - transition.target_start_frame + 1),
                agent_config=agent_config,
            )
            diagnostic = classify_v2_merge(topology_evidence, interaction)
            label = labels.get(row["candidate_id"], {})
            evidence_record = {
                "candidate_id": row["candidate_id"],
                "maneuver_id": row.get("maneuver_id") or row["candidate_id"],
                "scenario_id": scenario_id,
                "scene_key": row["scene_key"],
                "source_dataset": row.get("source_dataset", "WOMD"),
                "source_split": row.get("source_split", expansion.split),
                "source_shard": row["source_shard"],
                "record_index": record_index,
                "topology_evidence": dataclasses.asdict(topology_evidence),
                "interaction_evidence": dataclasses.asdict(interaction),
                "gap_timeseries": gap_samples,
                "automatic_decision": diagnostic.decision.value,
                "automatic_reason": diagnostic.reason,
                "manual_validation": label.get("manual_validation", "UNREVIEWED"),
                "manual_note": label.get("manual_note", ""),
            }
            out.write(
                json.dumps(evidence_record, allow_nan=True, cls=_NumpyJSONEncoder) + "\n"
            )

            # Render every decision category (accept/reject/review) for
            # human audit, not only automatic ACCEPTs -- a reject or an
            # ambiguous review call needs the same visual evidence to be
            # checked for correctness.
            traj = record.state.log_trajectory
            frames = slice(decision_start, min(transition.target_end_frame + 1, np.asarray(traj.x).shape[1]))
            ego_xy = np.column_stack((
                np.asarray(traj.x)[record.sdc_index, frames],
                np.asarray(traj.y)[record.sdc_index, frames],
            ))
            relevant_ids = set(interaction.relevant_vehicle_ids)
            ids = np.asarray(record.state.object_metadata.ids)
            tracks = {
                int(ids[i]): np.column_stack((
                    np.asarray(traj.x)[i, frames], np.asarray(traj.y)[i, frames]
                ))
                for i in range(len(ids)) if int(ids[i]) in relevant_ids
            }
            safe_id = row["candidate_id"].replace("/", "_").replace("#", "_")
            decision_subdir = diagnostic.decision.value  # "accept" | "reject" | "review"
            render_v2_review_panel(
                topology=topology_evidence,
                interaction=interaction,
                source_xy=source.xy,
                target_xy=target.xy,
                ego_xy=ego_xy,
                gap_timeseries=gap_samples,
                agent_tracks=tracks,
                decision=diagnostic.decision.value,
                reason=diagnostic.reason,
                output_path=str(Path(args.review_dir) / decision_subdir / f"{safe_id}.png"),
            )
            written += 1
    print(f"wrote {written} evidence records to {output}")
    if skipped_no_topology:
        print(
            f"skipped {len(skipped_no_topology)} candidate row(s) with no "
            "matching Scenario protobuf topology (scenario_id not found in "
            "--scenario-proto shards):"
        )
        for shard_path, idx, sid in skipped_no_topology[:20]:
            print(f"  {shard_path} record {idx}: scenario_id={sid!r}")
        if len(skipped_no_topology) > 20:
            print(f"  ... and {len(skipped_no_topology) - 20} more")


if __name__ == "__main__":
    main()
