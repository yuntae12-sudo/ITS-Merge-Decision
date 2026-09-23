#!/usr/bin/env python3
"""Evaluate FSM and trivial sanity policies on the canonical v2 split."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment.fsm_policy import FsmPolicy
from src.environment.full_split_evaluator import load_maneuver_specs, run_full_split
from src.environment.merge_environment import MergeEnvironment
from src.environment.trivial_policies import AlwaysKeepPolicy, AlwaysMergePolicy, AlwaysStopPolicy
from src.scenarios.merge_v2 import DATASET_SCHEMA_V2


POLICIES = {
    "fsm": FsmPolicy,
    "always_keep": AlwaysKeepPolicy,
    "always_merge": AlwaysMergePolicy,
    "always_stop": AlwaysStopPolicy,
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("train", "tune", "validation"), default="validation")
    p.add_argument("--dataset-config", required=True)
    p.add_argument("--manifest-root", default="data/manifests/v2")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--output", default="outputs/v2_baselines.json")
    return p.parse_args(argv)


def summarize(results):
    by_type = defaultdict(Counter)
    for result in results:
        by_type[result.maneuver_type or "unknown"][result.outcome] += 1
    return {
        "n": len(results),
        "outcomes": dict(Counter(r.outcome for r in results)),
        "by_maneuver_type": {k: dict(v) for k, v in sorted(by_type.items())},
        "immediate_merge_rate": (
            sum(r.immediate_merge for r in results) / len(results) if results else 0.0
        ),
        "fallback_rate": (
            sum(r.fallback_count for r in results)
            / max(sum(r.steps_elapsed for r in results), 1)
        ),
    }


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.manifest_root)
    specs = load_maneuver_specs(
        args.split,
        split_manifest_path=str(root / "dataset_split_v2.csv"),
        maneuver_table_path=str(root / "merge_maneuvers_v2.csv"),
        candidate_manifest_path=str(root / "merge_manifest_v2.csv"),
        required_schema_version=DATASET_SCHEMA_V2,
    )
    env = MergeEnvironment(
        args.dataset_config,
        downstream_mode="frenet_mpc",
        required_dataset_schema_version=DATASET_SCHEMA_V2,
    )
    report = {
        name: summarize(run_full_split(env, factory(), specs, args.max_steps))
        for name, factory in POLICIES.items()
    }
    always_keep_successes = report["always_keep"]["outcomes"].get("success", 0)
    if always_keep_successes:
        raise RuntimeError(
            f"v2 validity failure: AlwaysKeep produced {always_keep_successes} success(es)"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
