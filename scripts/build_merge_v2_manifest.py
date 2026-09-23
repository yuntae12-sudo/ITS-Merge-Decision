#!/usr/bin/env python3
"""Build schema-v2 candidate/canonical manifests from audited evidence JSONL.

Each input row must carry authoritative ``topology_evidence`` produced from a
WOMD Scenario protobuf and temporal ``interaction_evidence``.  The script
fails closed: only automatic ACCEPT + ``CONFIRMED_MERGE`` rows enter the
canonical manifest; non-interactive events remain in the candidate audit.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios.v2_manifest import (
    V2_MANIFEST_FIELDS,
    assign_scenario_grouped_splits,
    classify_evidence_record,
    load_calibrated_v2_thresholds,
    write_csv,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-jsonl", required=True)
    parser.add_argument("--output-dir", default="data/manifests/v2")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--v2-config", default="configs/merge_v2.yaml")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.evidence_jsonl)
    classifier_kwargs = load_calibrated_v2_thresholds(args.v2_config)
    rows = []
    with input_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(classify_evidence_record(json.loads(line), classifier_kwargs))
            except Exception as exc:
                raise ValueError(f"Invalid evidence at {input_path}:{line_number}: {exc}") from exc

    canonical = [row for row in rows if row.pop("canonical")]
    canonical = assign_scenario_grouped_splits(canonical, seed=args.split_seed)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "merge_candidates_v2.csv", rows, V2_MANIFEST_FIELDS)
    write_csv(
        output_dir / "merge_manifest_v2.csv",
        canonical,
        V2_MANIFEST_FIELDS + ["split"],
    )
    write_csv(
        output_dir / "merge_maneuvers_v2.csv",
        canonical,
        V2_MANIFEST_FIELDS + ["split"],
    )
    write_csv(
        output_dir / "dataset_split_v2.csv",
        [{"maneuver_id": r["maneuver_id"], "scene_key": r["scene_key"], "scenario_id": r["scenario_id"], "split": r["split"]} for r in canonical],
        ["maneuver_id", "scene_key", "scenario_id", "split"],
    )
    print(f"candidates={len(rows)} canonical={len(canonical)}")
    print("types", dict(Counter(r["maneuver_type"] for r in canonical)))
    print("splits", dict(Counter(r["split"] for r in canonical)))


if __name__ == "__main__":
    main()
