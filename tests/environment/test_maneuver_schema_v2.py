import json

import pytest

from src.environment.merge_environment import ManeuverSpec
from src.scenarios.merge_v2 import DATASET_SCHEMA_V2, LEGACY_DATASET_SCHEMA


def test_legacy_maneuver_defaults_to_explicit_legacy_schema():
    spec = ManeuverSpec("M", "s", 0, [1, 2], ["c"], 10)
    assert spec.schema_version == LEGACY_DATASET_SCHEMA
    with pytest.raises(ValueError, match="expected"):
        spec.require_schema(DATASET_SCHEMA_V2)


def test_v2_csv_row_requires_manual_and_evidence():
    row = {
        "maneuver_id": "M2", "source_shard": "s", "record_index": "0",
        "lane_chain": "1->2", "candidate_ids": "c2",
        "schema_version": DATASET_SCHEMA_V2,
        "maneuver_type": "topological_merge",
        "manual_validation": "CONFIRMED_MERGE",
        "topology_evidence": json.dumps({"source_exits_to_target": True}),
        "interaction_evidence": json.dumps({"front_vehicle_id": 7, "conflict_vehicle_ids": []}),
    }
    spec = ManeuverSpec.from_csv_row(
        row, {"c2": {"merge_start_frame": "10", "schema_version": DATASET_SCHEMA_V2}}
    )
    spec.require_schema(DATASET_SCHEMA_V2)
    assert spec.topology_evidence["source_exits_to_target"] is True


def test_unreviewed_v2_maneuver_fails_closed():
    spec = ManeuverSpec(
        "M", "s", 0, [1, 2], ["c"], 10,
        schema_version=DATASET_SCHEMA_V2,
        maneuver_type="topological_merge",
        manual_validation="UNREVIEWED",
        topology_evidence={"source_exits_to_target": True},
        interaction_evidence={"front_vehicle_id": 7},
    )
    with pytest.raises(ValueError, match="not manually confirmed"):
        spec.require_schema(DATASET_SCHEMA_V2)
