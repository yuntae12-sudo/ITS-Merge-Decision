"""Phase 1 CLI: iterate WOMD scenarios and print basic scan metadata.

Example:
    python scripts/inspect_scenarios.py --limit 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios.scenario_loader import iter_all_shards, load_dataset_config

DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"


def parse_args():

    parser = argparse.ArgumentParser(
        description="Iterate WOMD scenarios and print scan metadata."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_DATASET_CONFIG,
        help="Path to dataset YAML config.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of scenarios to iterate.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("=" * 70)
    print("ITS Merge Decision - Phase 1")
    print("Scenario Iterator Inspection")
    print("=" * 70)

    expansion_config = load_dataset_config(args.config)

    print("\n[1] Dataset Config")
    print(f"Config file    : {args.config}")
    print(f"Split          : {expansion_config.split}")
    print(f"Physical shards: {len(expansion_config.shard_paths)}")
    for shard_path in expansion_config.shard_paths:
        print(f"  - {shard_path}")
    print(f"Max objects    : {expansion_config.max_num_objects}")
    print(f"Limit          : {args.limit} (total across all shards)")

    print("\n" + "-" * 70)
    print("[2] Scenario Scan")
    print("-" * 70)

    header = (
        f"{'SPLIT':>10} "
        f"{'SOURCE_SHARD':<45} "
        f"{'IDX':>4} "
        f"{'SCENE_KEY':<50} "
        f"{'NUM_OBJ':>8} "
        f"{'SDC_IDX':>8} "
        f"{'SDC_ID':>10} "
        f"{'VALID_LEN':>10} "
        f"{'RG_POINTS':>10}"
    )

    print(header)
    print("-" * 70)

    scanned = 0

    for record in iter_all_shards(expansion_config, limit=args.limit):

        print(
            f"{record.source_split:>10} "
            f"{record.source_shard:<45} "
            f"{record.record_index:>4} "
            f"{record.scene_key:<50} "
            f"{record.num_objects:>8} "
            f"{record.sdc_index:>8} "
            f"{record.sdc_id:>10} "
            f"{record.valid_trajectory_length:>10} "
            f"{record.roadgraph_point_count:>10}"
        )

        scanned += 1

    print("\n" + "=" * 70)
    print(f"Scenarios scanned: {scanned}")

    if scanned == args.limit:
        print("SCENARIO ITERATION: PASS")
    else:
        print(
            f"SCENARIO ITERATION: WARNING "
            f"(expected {args.limit}, got {scanned})"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
