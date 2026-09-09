"""Phase 1 CLI: batch-scan WOMD scenes and build the flat merge
-candidate dataset (Compressed Commit E; multi-shard dataset expansion).

For each scenario across every physical shard in the configured
dataset expansion config (see ``src.scenarios.scenario_loader``),
extracts all stable lane-transition candidates (Commit C), classifies
them (Commit D), and extracts interaction features for ACCEPTed
candidates -- flattening everything into one row per transition via
``src.scenarios.dataset_builder.build_candidate_records``. Writes:

    - ``--output`` (default data/manifests/merge_candidates.csv): the
      full candidate manifest, across all scanned shards.
    - ``outputs/phase1/summaries/merge_scan_summary.json``: shard-aware
      summary -- a ``global`` section (aggregate counts across all
      shards) and a ``per_shard`` section (one entry per physical
      shard scanned).
    - ``outputs/phase1/summaries/merge_scan_failures.json``: only
      written if one or more scenes raised an exception during
      candidate-building (isolated per-scene, never crashes the batch)
      or one or more physical shards could not be opened/parsed at all
      (isolated per-shard, never crashes the whole multi-shard scan).

``--limit-scenes``/``--start-record`` apply to the TOTAL scan across
all shards in this invocation, not per-shard: ``--limit-scenes N``
stops after N scenes total, scanning shard by shard in the dataset
config's sorted shard order (see ``iter_all_shards``).

Example:
    python scripts/extract_merge_scenes.py --limit-scenes 30 --overwrite
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios.dataset_builder import (
    build_candidate_records,
    compute_multi_shard_summary,
    write_candidates_csv,
    write_summary_json,
)
from src.scenarios.lane_assignment import load_lane_assignment_config
from src.scenarios.merge_detector import load_merge_topology_config
from src.scenarios.scenario_features import load_agent_selection_config
from src.scenarios.scenario_loader import (
    build_waymax_config,
    iter_scenarios,
    load_dataset_config,
)

DEFAULT_DATASET_CONFIG = "configs/dataset.yaml"
DEFAULT_PHASE1_CONFIG = "configs/phase1_merge.yaml"
DEFAULT_OUTPUT = "data/manifests/merge_candidates.csv"
DEFAULT_SUMMARY_DIR = "outputs/phase1/summaries"


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Batch-scan a WOMD dataset expansion config (one or more "
            "physical shards, one split) and build the flat merge"
            "-candidate CSV manifest + shard-aware scan summary. "
            "--limit-scenes/--start-record apply to the TOTAL scan "
            "across all shards, not per-shard."
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


def _scan_all_shards(
    expansion_config,
    lane_assignment_config,
    merge_topology_config,
    agent_selection_config,
    limit_scenes,
    start_record,
):
    """Scans every physical shard in ``expansion_config``, isolating
    both scene-level failures (existing per-scene try/except inside
    ``build_candidate_records``) and shard-level LOADING failures
    (the physical file itself can't be opened/parsed -- wrapped here
    around the per-shard ``iter_scenarios`` call so one bad shard does
    not abort the whole multi-shard scan).

    ``limit_scenes``/``start_record`` apply to the TOTAL scan across
    all shards (see module docstring), implemented the same way
    ``iter_all_shards`` does it, but kept inline here (rather than
    calling ``iter_all_shards`` directly) so shard-level load failures
    can be caught per shard without losing already-scanned shards'
    results.

    Returns:
        (all_candidates, scene_failures, shard_failures,
         scenes_scanned, per_shard_scan_counts) where
         per_shard_scan_counts maps (source_split, source_shard) ->
         {"scenes_scanned": int, "scenes_failed": int}.
    """

    all_candidates = []
    scene_failures = []
    shard_failures = []
    per_shard_scan_counts = {}

    stop_index = None if limit_scenes is None else start_record + limit_scenes
    scenes_seen_total = 0

    for shard_path in expansion_config.shard_paths:

        if stop_index is not None and scenes_seen_total >= stop_index:
            break

        shard_name = Path(shard_path).name
        shard_key = (expansion_config.split, shard_name)
        per_shard_scan_counts.setdefault(
            shard_key, {"scenes_scanned": 0, "scenes_failed": 0}
        )

        try:
            waymax_cfg = build_waymax_config(expansion_config, shard_path)
            shard_iterator = iter_scenarios(
                waymax_cfg,
                source_dataset=expansion_config.dataset_name,
                source_split=expansion_config.split,
            )

            for record in shard_iterator:

                if scenes_seen_total < start_record:
                    scenes_seen_total += 1
                    continue

                if stop_index is not None and scenes_seen_total >= stop_index:
                    break

                scenes_seen_total += 1
                per_shard_scan_counts[shard_key]["scenes_scanned"] += 1

                records, error_info = build_candidate_records(
                    record,
                    lane_assignment_config,
                    merge_topology_config,
                    agent_selection_config,
                )

                if error_info is not None:
                    scene_failures.append(error_info)
                    per_shard_scan_counts[shard_key]["scenes_failed"] += 1
                    continue

                all_candidates.extend(records)

        except Exception as exc:  # noqa: BLE001 - intentional broad catch
            # for shard-level isolation; caller inspects shard_failures.
            # A shard-loading failure (the physical file can't be
            # opened/parsed at all) must not abort the rest of the
            # multi-shard scan.
            shard_failures.append(
                {
                    "source_split": expansion_config.split,
                    "source_shard": shard_name,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            )

    return (
        all_candidates,
        scene_failures,
        shard_failures,
        scenes_seen_total - start_record if scenes_seen_total >= start_record else 0,
        per_shard_scan_counts,
    )


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
    print("Extract Merge Scenes (multi-shard dataset expansion)")
    print("=" * 70)

    expansion_config = load_dataset_config(args.dataset_config)
    lane_assignment_config = load_lane_assignment_config(args.phase1_config)
    merge_topology_config = load_merge_topology_config(args.phase1_config)
    agent_selection_config = load_agent_selection_config(args.phase1_config)

    print(f"\nDataset split        : {expansion_config.split}")
    print(f"Physical shards       : {len(expansion_config.shard_paths)}")
    for shard_path in expansion_config.shard_paths:
        print(f"  - {shard_path}")

    (
        all_candidates,
        scene_failures,
        shard_failures,
        scenes_scanned,
        per_shard_scan_counts,
    ) = _scan_all_shards(
        expansion_config,
        lane_assignment_config,
        merge_topology_config,
        agent_selection_config,
        args.limit_scenes,
        args.start_record,
    )

    print(f"\nScenes scanned       : {scenes_scanned}")
    print(f"Scenes failed         : {len(scene_failures)}")
    print(f"Shards failed         : {len(shard_failures)}")
    print(f"Total transitions     : {len(all_candidates)}")

    write_candidates_csv(all_candidates, output_path)
    print(f"\nWrote candidates CSV : {output_path}")

    summary = compute_multi_shard_summary(
        all_candidates,
        physical_shards_scanned=len(expansion_config.shard_paths),
        scenes_scanned=scenes_scanned,
        scenes_failed=len(scene_failures),
        per_shard_scan_counts=per_shard_scan_counts,
    )

    global_summary = summary["global"]
    # review_ratio/review_ratio_among_accept_review/reason_histogram
    # already computed inside global_summary by
    # compute_summary_statistics (reused, not recomputed here).

    summary_path = Path(DEFAULT_SUMMARY_DIR) / "merge_scan_summary.json"
    write_summary_json(summary, summary_path)
    print(f"Wrote scan summary   : {summary_path}")

    if scene_failures or shard_failures:
        failures_path = Path(DEFAULT_SUMMARY_DIR) / "merge_scan_failures.json"
        write_summary_json(
            {"scene_failures": scene_failures, "shard_failures": shard_failures},
            failures_path,
        )
        print(f"Wrote failures       : {failures_path}")

        if scene_failures:
            print("\nWARN: some scenes failed during candidate extraction:")
            for failure in scene_failures:
                print(
                    f"  scene_key={failure['scene_key']} "
                    f"record_index={failure['record_index']} "
                    f"{failure['exception_type']}: "
                    f"{failure['exception_message']}"
                )

        if shard_failures:
            print("\nWARN: some shards failed to load/parse:")
            for failure in shard_failures:
                print(
                    f"  source_split={failure['source_split']} "
                    f"source_shard={failure['source_shard']} "
                    f"{failure['exception_type']}: "
                    f"{failure['exception_message']}"
                )

    print("\n" + "=" * 70)
    print("Scan Summary (global)")
    print("=" * 70)
    print(f"  physical_shards_scanned             : {global_summary['physical_shards_scanned']}")
    print(f"  scenes_scanned                      : {global_summary['scenes_scanned']}")
    print(f"  scenes_with_transitions             : {global_summary['scenes_with_transitions']}")
    print(f"  total_stable_transitions            : {global_summary['total_stable_transitions']}")
    print(f"  ACCEPT                               : {global_summary['accept_count']}")
    print(f"  REJECT                               : {global_summary['reject_count']}")
    print(f"  REVIEW                               : {global_summary['review_count']}")
    print(f"  review_ratio                          : {global_summary['review_ratio']:.4f}")
    print(
        "  review_ratio_among_accept_review      : "
        f"{global_summary['review_ratio_among_accept_review']:.4f}"
    )
    print("\n  Reason histogram:")
    for reason, count in sorted(
        global_summary["reason_histogram"].items(), key=lambda item: (-item[1], item[0])
    ):
        print(f"    {reason}: {count}")

    print("\n  Per-shard summary:")
    for shard_summary in summary["per_shard"]:
        print(
            f"    {shard_summary['source_split']}/{shard_summary['source_shard']}: "
            f"scanned={shard_summary['scenes_scanned']} "
            f"failed={shard_summary['scenes_failed']} "
            f"transitions={shard_summary['total_stable_transitions']} "
            f"ACCEPT={shard_summary['accept_count']} "
            f"REJECT={shard_summary['reject_count']} "
            f"REVIEW={shard_summary['review_count']}"
        )

    all_scenes_failed = (
        scenes_scanned > 0 and len(scene_failures) == scenes_scanned
    )
    all_shards_failed = (
        len(expansion_config.shard_paths) > 0
        and len(shard_failures) == len(expansion_config.shard_paths)
    )
    if all_scenes_failed or all_shards_failed:
        print("\nALL scenes/shards failed -- exiting non-zero.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("EXTRACT MERGE SCENES: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
