"""Serialization and split utilities for the interaction-aware v2 dataset."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, List

import yaml

from src.scenarios.merge_v2 import (
    DATASET_SCHEMA_V2,
    InteractionEvidence,
    TopologyEvidence,
    classify_v2_merge,
)


V2_MANIFEST_FIELDS = [
    "schema_version", "candidate_id", "maneuver_id", "scenario_id",
    "scene_key", "source_dataset", "source_split", "source_shard",
    "record_index", "maneuver_type", "source_lane_id", "target_lane_id",
    "lane_chain", "candidate_ids", "merge_start_frame", "commit_frame",
    "merge_complete_frame", "decision_start_frame", "source_occupancy_frames",
    "target_stable_frames", "longitudinal_progress_m", "lateral_displacement_m",
    "front_vehicle_id", "rear_vehicle_id", "conflict_vehicle_ids",
    "gap_order_valid_at_entry", "gap_order_persistence_frames",
    "interactive", "detector_decision", "detector_reason",
    "manual_validation", "manual_note", "topology_evidence",
    "interaction_evidence", "gap_timeseries",
]


def _tuple_int(values):
    return tuple(int(x) for x in (values or ()))


def load_calibrated_v2_thresholds(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if raw.get("calibration_status") != "calibrated":
        raise ValueError(
            f"{path} is not calibrated; review the protobuf-backed candidate "
            "distribution and manual labels before building a canonical manifest"
        )
    values = dict(raw.get("classification") or {})
    values.pop("required_gap_order_persistence_frames", None)
    return values


def classify_evidence_record(record: dict, classifier_kwargs=None) -> dict:
    """Validate and classify one JSON-compatible evidence record."""

    top_raw = dict(record["topology_evidence"])
    top_raw["source_exit_lane_ids"] = _tuple_int(top_raw.get("source_exit_lane_ids"))
    top_raw["target_entry_lane_ids"] = _tuple_int(top_raw.get("target_entry_lane_ids"))
    topology = TopologyEvidence(**top_raw)

    int_raw = dict(record["interaction_evidence"])
    int_raw["conflict_vehicle_ids"] = _tuple_int(int_raw.get("conflict_vehicle_ids"))
    interaction = InteractionEvidence(**int_raw)
    diagnostic = classify_v2_merge(
        topology, interaction, **(classifier_kwargs or {})
    )

    candidate_id = str(record["candidate_id"])
    maneuver_id = str(record.get("maneuver_id") or candidate_id)
    manual_validation = str(record.get("manual_validation", "UNREVIEWED"))
    row = {
        "schema_version": DATASET_SCHEMA_V2,
        "candidate_id": candidate_id,
        "maneuver_id": maneuver_id,
        "scenario_id": str(record["scenario_id"]),
        "scene_key": str(record.get("scene_key", record["scenario_id"])),
        "source_dataset": str(record.get("source_dataset", "WOMD")),
        "source_split": str(record.get("source_split", "")),
        "source_shard": str(record["source_shard"]),
        "record_index": int(record["record_index"]),
        "maneuver_type": diagnostic.maneuver_type.value if diagnostic.maneuver_type else "",
        "source_lane_id": topology.source_lane_id,
        "target_lane_id": topology.target_lane_id,
        "lane_chain": f"{topology.source_lane_id}->{topology.target_lane_id}",
        "candidate_ids": candidate_id,
        "merge_start_frame": interaction.decision_start_frame,
        "commit_frame": interaction.commit_frame,
        "merge_complete_frame": interaction.completion_frame,
        "decision_start_frame": interaction.decision_start_frame,
        "source_occupancy_frames": interaction.source_occupancy_frames,
        "target_stable_frames": interaction.target_stable_frames,
        "longitudinal_progress_m": interaction.longitudinal_progress_m,
        "lateral_displacement_m": interaction.lateral_displacement_m,
        "front_vehicle_id": interaction.front_vehicle_id,
        "rear_vehicle_id": interaction.rear_vehicle_id,
        "conflict_vehicle_ids": json.dumps(interaction.conflict_vehicle_ids),
        "gap_order_valid_at_entry": interaction.gap_order_valid_at_entry,
        "gap_order_persistence_frames": interaction.gap_order_persistence_frames,
        "interactive": diagnostic.interactive,
        "detector_decision": diagnostic.decision.value,
        "detector_reason": diagnostic.reason or "",
        "manual_validation": manual_validation,
        "manual_note": str(record.get("manual_note", "")),
        "topology_evidence": json.dumps(top_raw, sort_keys=True),
        "interaction_evidence": json.dumps(int_raw, sort_keys=True),
        "gap_timeseries": json.dumps(record.get("gap_timeseries", []), sort_keys=True),
    }
    row["canonical"] = bool(
        diagnostic.canonical and manual_validation == "CONFIRMED_MERGE"
    )
    return row


def assign_scenario_grouped_splits(rows: Iterable[dict], seed: int = 0) -> List[dict]:
    """Deterministic 70/15/15 split keyed only by scenario_id."""

    result = []
    scenario_split = {}
    for row in rows:
        scenario_id = row["scenario_id"]
        if scenario_id not in scenario_split:
            digest = hashlib.sha256(f"{seed}:{scenario_id}".encode()).digest()
            bucket = int.from_bytes(digest[:8], "big") % 100
            scenario_split[scenario_id] = (
                "train" if bucket < 70 else "tune" if bucket < 85 else "validation"
            )
        copied = dict(row)
        copied["split"] = scenario_split[scenario_id]
        result.append(copied)
    return result


def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "V2_MANIFEST_FIELDS",
    "classify_evidence_record",
    "assign_scenario_grouped_splits",
    "write_csv",
    "load_calibrated_v2_thresholds",
]
