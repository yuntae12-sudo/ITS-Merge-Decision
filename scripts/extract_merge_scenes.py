"""Phase 1 CLI: batch-scan WOMD scenes and build the flat merge
-candidate dataset (Compressed Commit E).

For each scenario in the configured dataset shard, extracts all stable
lane-transition candidates (Commit C), classifies them (Commit D), and
extracts interaction features for ACCEPTed candidates -- flattening
everything into one row per transition via
``src.scenarios.dataset_builder.build_candidate_records``. Writes:

    - ``--output`` (default data/manifests/merge_candidates.csv): the
      full candidate manifest.
    - ``outputs/phase1/summaries/merge_scan_summary.json``: aggregate
      counts (scenes scanned, ACCEPT/REJECT/REVIEW counts, review
      ratios, reason histogram).
    - ``outputs/phase1/summaries/merge_scan_failures.json``: only
      written if one or more scenes raised an exception during
      candidate-building (isolated per-scene, never crashes the batch).

Example:
    python scripts/extract_merge_scenes.py --limit-scenes 30 --overwrite
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios.dataset_builder import (
    build_candidate_records,
    compute_summary_statistics,
    parse_source_split,
    write_candidates_csv,
    write_summary_json,
)
from src.scenarios.lane_assignment import load_lane_assignment_config
from src.scenarios.merge_detector import load_merge_topology_config
from src.scenarios.scenario_features import load_agent_selection_config
from src.scenarios.scenario_loader import iter_scenarios, load_dataset_config

DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"
DEFAULT_PHASE1_CONFIG = "configs/phase1_merge.yaml"
DEFAULT_OUTPUT = "data/manifests/merge_candidates.csv"
DEFAULT_SUMMARY_DIR = "outputs/phase1/summaries"


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Batch-scan a WOMD dataset shard and build the flat merge "
            "-candidate CSV manifest + scan summary."
        )
    )

    parser.add_argument(
        "--dataset-config", type=str, default=DEFAULT_DATASET_CONFIG
    )
    parser.add_argument(
        "--phase1-config", type=str, default=DEFAULT_PHASE1_CONFIG
    )
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--start-record", type=int, default=0)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main():

    args = parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise RuntimeError(
            f"Output file already exists: {output_path}. "
            "Pass --overwrite to replace it."
        )

    print("=" * 70)
    print("ITS Merge Decision - Phase 1")
    print("Extract Merge Scenes (Compressed Commit E)")
    print("=" * 70)

    dataset_config = load_dataset_config(args.dataset_config)
    lane_assignment_config = load_lane_assignment_config(args.phase1_config)
    merge_topology_config = load_merge_topology_config(args.phase1_config)
    agent_selection_config = load_agent_selection_config(args.phase1_config)

    source_split = parse_source_split(dataset_config.path)

    all_candidates = []
    failures = []
    scenes_scanned = 0

    for record in iter_scenarios(
        dataset_config, limit=args.limit_scenes, start_index=args.start_record
    ):
        scenes_scanned += 1

        records, error_info = build_candidate_records(
            record,
            lane_assignment_config,
            merge_topology_config,
            agent_selection_config,
            source_split=source_split,
        )

        if error_info is not None:
            failures.append(error_info)
            continue

        all_candidates.extend(records)

    print(f"\nScenes scanned      : {scenes_scanned}")
    print(f"Scenes failed        : {len(failures)}")
    print(f"Total transitions    : {len(all_candidates)}")

    write_candidates_csv(all_candidates, output_path)
    print(f"\nWrote candidates CSV : {output_path}")

    summary = compute_summary_statistics(all_candidates)
    summary["scenes_scanned"] = scenes_scanned
    summary["scenes_failed"] = len(failures)

    summary_path = Path(DEFAULT_SUMMARY_DIR) / "merge_scan_summary.json"
    write_summary_json(summary, summary_path)
    print(f"Wrote scan summary   : {summary_path}")

    if failures:
        failures_path = Path(DEFAULT_SUMMARY_DIR) / "merge_scan_failures.json"
        write_summary_json({"failures": failures}, failures_path)
        print(f"Wrote failures       : {failures_path}")

        print("\nWARN: some scenes failed during candidate extraction:")
        for failure in failures:
            print(
                f"  scene_key={failure['scene_key']} "
                f"record_index={failure['record_index']} "
                f"{failure['exception_type']}: "
                f"{failure['exception_message']}"
            )

    print("\n" + "=" * 70)
    print("Scan Summary")
    print("=" * 70)
    print(f"  scenes_scanned                     : {scenes_scanned}")
    print(f"  scenes_with_transitions            : {summary['scenes_with_transitions']}")
    print(f"  total_stable_transitions           : {summary['total_stable_transitions']}")
    print(f"  ACCEPT                              : {summary['accept_count']}")
    print(f"  REJECT                              : {summary['reject_count']}")
    print(f"  REVIEW                              : {summary['review_count']}")
    print(f"  review_ratio                         : {summary['review_ratio']:.4f}")
    print(
        "  review_ratio_among_accept_review     : "
        f"{summary['review_ratio_among_accept_review']:.4f}"
    )
    print("\n  Reason histogram:")
    for reason, count in sorted(
        summary["reason_histogram"].items(), key=lambda item: (-item[1], item[0])
    ):
        print(f"    {reason}: {count}")

    if failures and scenes_scanned == len(failures):
        print("\nALL scenes failed -- exiting non-zero.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("EXTRACT MERGE SCENES: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
