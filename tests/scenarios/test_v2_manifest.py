import json

from src.scenarios.merge_v2 import DATASET_SCHEMA_V2
from src.scenarios.v2_manifest import (
    assign_scenario_grouped_splits,
    classify_evidence_record,
    load_calibrated_v2_thresholds,
)
import pytest


def record(candidate_id="c1", scenario_id="s1", manual="CONFIRMED_MERGE"):
    return {
        "candidate_id": candidate_id,
        "maneuver_id": candidate_id,
        "scenario_id": scenario_id,
        "scene_key": scenario_id,
        "source_shard": "scenario.tfrecord",
        "record_index": 0,
        "manual_validation": manual,
        "topology_evidence": {
            "source_lane_id": 1,
            "target_lane_id": 2,
            "source_exit_lane_ids": [2],
            "target_entry_lane_ids": [1, 3],
            "source_exits_to_target": True,
            "terminal_heading_difference_deg": 10.0,
            "terminal_collinear_offset_m": 4.0,
            "endpoint_gap_m": 1.0,
        },
        "interaction_evidence": {
            "decision_start_frame": 10,
            "commit_frame": 20,
            "completion_frame": 35,
            "source_occupancy_frames": 10,
            "target_stable_frames": 8,
            "longitudinal_progress_m": 20.0,
            "lateral_displacement_m": 3.0,
            "front_vehicle_id": 7,
        },
    }


def test_manifest_requires_auto_accept_and_manual_confirmation():
    row = classify_evidence_record(record())
    assert row["schema_version"] == DATASET_SCHEMA_V2
    assert row["canonical"] is True
    assert json.loads(row["topology_evidence"])["target_entry_lane_ids"] == [1, 3]

    unreviewed = classify_evidence_record(record(manual="UNREVIEWED"))
    assert unreviewed["detector_decision"] == "accept"
    assert unreviewed["canonical"] is False


def test_split_never_leaks_one_scenario():
    rows = [
        classify_evidence_record(record("c1", "shared")),
        classify_evidence_record(record("c2", "shared")),
        classify_evidence_record(record("c3", "other")),
    ]
    split = assign_scenario_grouped_splits(rows, seed=123)
    shared = {r["split"] for r in split if r["scenario_id"] == "shared"}
    assert len(shared) == 1


def test_canonical_build_is_blocked_until_thresholds_are_calibrated(tmp_path):
    config = tmp_path / "v2.yaml"
    config.write_text("calibration_status: pending\nclassification: {}\n")
    with pytest.raises(ValueError, match="not calibrated"):
        load_calibrated_v2_thresholds(str(config))

    config.write_text(
        "calibration_status: calibrated\n"
        "classification:\n"
        "  min_source_occupancy_frames: 7\n"
        "  required_gap_order_persistence_frames: 5\n"
    )
    assert load_calibrated_v2_thresholds(str(config)) == {
        "min_source_occupancy_frames": 7
    }
