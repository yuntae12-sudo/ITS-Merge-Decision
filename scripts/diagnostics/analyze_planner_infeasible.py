#!/usr/bin/env python3
"""PLANNER_INFEASIBLE root-cause diagnostic (read-only analysis only).

Runs real, deterministic, closed-loop MergeEnvironment(downstream_mode=
"frenet_mpc") rollouts for a fixed maneuver_id list under a restored PPO
checkpoint, and summarizes -- per feasibility check
(longitudinal_accel/forward_progress/curvature/longitudinal_jerk) -- how
often and how severely each check fails whenever a step's
downstream_status is PLANNER_INFEASIBLE.

This script does not change training, reward, planner thresholds, MPC,
candidate generation, or dataset behavior in any way -- it only reads
the `info["downstream_diagnostics"]` field each PPOStepRecord already
carries verbatim (see src/environment/merge_environment.py's
`downstream_diagnostics` info key and src/planning/candidate_evaluator.py's
`EvaluationResult.feasibility_values`).

Usage:

    PYTHONPATH=. python scripts/diagnostics/analyze_planner_infeasible.py \\
        --checkpoint outputs/checkpoints/full_seed0_step000036.pkl \\
        --maneuver-id-file outputs/diagnostics/train_diag64_seed20260928.txt \\
        --dataset-config-path configs/dataset.yaml \\
        --max-episode-steps 100 \\
        --output-dir outputs/diagnostics/planner_infeasible_step36
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.environment.full_split_evaluator import load_decision_dataset_maneuver_specs
from src.environment.merge_environment import MergeEnvironment
from src.scenarios.merge_v2 import MERGE_DATASET_SCHEMA
from src.training.config import load_reward_config
from src.visualization.ppo_checkpoint_policy import restore_ppo_checkpoint
from src.visualization.ppo_rollout import PPOStepRecord, run_ppo_episode

FEASIBILITY_CHECKS = (
    "longitudinal_accel",
    "forward_progress",
    "curvature",
    "longitudinal_jerk",
)
FEASIBILITY_VALUE_KEYS = {
    "longitudinal_accel": "max_abs_longitudinal_accel_mps2",
    "forward_progress": "min_forward_progress_s_dot_mps",
    "curvature": "max_abs_curvature_per_m",
    "longitudinal_jerk": "max_abs_longitudinal_jerk_mps3",
}
PHASE_PRE_COMMIT = "PRE_COMMIT_POLICY"
PHASE_POST_COMMIT = "POST_COMMIT_AUTO_MERGE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a PPO checkpoint .pkl.")
    parser.add_argument("--maneuver-id-file", required=True,
                         help="Text file, one maneuver_id per line (e.g. Fixed TRAIN64).")
    parser.add_argument("--dataset-config-path", default="configs/dataset.yaml")
    parser.add_argument("--downstream-mode", default="frenet_mpc")
    parser.add_argument("--reward-config-path", default=None,
                         help="Defaults to the checkpoint's own recorded reward_config_path.")
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--policy-mode", choices=("deterministic", "stochastic"), default="deterministic")
    parser.add_argument("--output-dir", required=True)
    return parser


def _load_maneuver_ids(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _planner_diagnostics(record: PPOStepRecord) -> Optional[dict]:
    downstream_diagnostics = record.info.get("downstream_diagnostics")
    if not downstream_diagnostics:
        return None
    return downstream_diagnostics.get("planner")


def _phase(record: PPOStepRecord) -> str:
    return PHASE_POST_COMMIT if record.merge_committed_before else PHASE_PRE_COMMIT


def _action_label(record: PPOStepRecord) -> str:
    return record.selected_action if record.is_policy_step else "POST_COMMIT_AUTO_MERGE"


def analyze(episode_results, feasibility_checks=FEASIBILITY_CHECKS) -> dict:
    """Pure, read-only aggregation over already-collected episode
    results. Kept separate from I/O/CLI so it is directly unit
    -testable without running a real environment."""

    infeasible_records = []
    for episode in episode_results:
        for record in episode.steps:
            if record.downstream_status != "PLANNER_INFEASIBLE":
                continue
            planner = _planner_diagnostics(record)
            if planner is None:
                continue
            infeasible_records.append((episode, record, planner))

    total_infeasible = len(infeasible_records)

    check_counts = Counter()
    combination_counts = Counter()
    phase_check_counts = defaultdict(Counter)
    phase_totals = Counter()
    outcome_phase_totals = defaultdict(Counter)
    outcome_phase_check_counts = defaultdict(lambda: defaultdict(Counter))
    action_check_counts = defaultdict(Counter)
    violating_values = defaultdict(list)  # check -> list of measured values (only for FAILED checks)
    episode_rows = []

    for episode, record, planner in infeasible_records:
        checks_failed = tuple(planner.get("checks_failed", ()))
        feasibility_values = planner.get("feasibility_values", {}) or {}
        phase = _phase(record)
        outcome = episode.outcome
        action_label = _action_label(record)

        for check in checks_failed:
            if check not in feasibility_checks:
                continue
            check_counts[check] += 1
            phase_check_counts[phase][check] += 1
            outcome_phase_check_counts[outcome][phase][check] += 1
            action_check_counts[action_label][check] += 1
            value_key = FEASIBILITY_VALUE_KEYS.get(check)
            if value_key is not None and value_key in feasibility_values:
                violating_values[check].append(feasibility_values[value_key])

        relevant_combo = tuple(sorted(c for c in checks_failed if c in feasibility_checks))
        if relevant_combo:
            combination_counts[relevant_combo] += 1

        phase_totals[phase] += 1
        outcome_phase_totals[outcome][phase] += 1

        episode_rows.append({
            "maneuver_id": episode.maneuver_id,
            "outcome": episode.outcome,
            "phase": phase,
            "step_index": record.step_index,
            "action": action_label,
            "downstream_status": record.downstream_status,
            "checks_failed": ";".join(checks_failed),
            **{
                FEASIBILITY_VALUE_KEYS[c]: feasibility_values.get(FEASIBILITY_VALUE_KEYS[c])
                for c in feasibility_checks
            },
        })

    return {
        "total_infeasible_steps": total_infeasible,
        "check_counts": dict(check_counts),
        "combination_counts": {",".join(k) if k else "(none)": v for k, v in combination_counts.items()},
        "phase_totals": dict(phase_totals),
        "phase_check_counts": {p: dict(c) for p, c in phase_check_counts.items()},
        "outcome_phase_totals": {o: dict(p) for o, p in outcome_phase_totals.items()},
        "outcome_phase_check_counts": {
            o: {p: dict(c) for p, c in phases.items()}
            for o, phases in outcome_phase_check_counts.items()
        },
        "action_check_counts": {a: dict(c) for a, c in action_check_counts.items()},
        "violating_values": {k: list(v) for k, v in violating_values.items()},
        "episode_rows": episode_rows,
    }


def _fmt_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.1f}%"


def print_summary(result: dict) -> None:
    total = result["total_infeasible_steps"]

    print("-" * 60)
    print("A. Overall PLANNER_INFEASIBLE Failure Checks")
    print("-" * 60)
    print(f"{'Check':<24}{'Count':>8}{'Rate among infeasible':>26}")
    print("-" * 60)
    for check in FEASIBILITY_CHECKS:
        count = result["check_counts"].get(check, 0)
        print(f"{check:<24}{count:>8}{_fmt_pct(count, total):>26}")
    print(
        "(NOTE: multiple checks can fail on the same step, so counts "
        f"can sum to more than total PLANNER_INFEASIBLE steps={total}.)"
    )

    print()
    print("Exact failure combinations:")
    for combo, count in sorted(result["combination_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  [{combo}]  {count}")

    print()
    print("-" * 60)
    print("B. PRE vs POST COMMIT")
    print("-" * 60)
    for phase in (PHASE_PRE_COMMIT, PHASE_POST_COMMIT):
        phase_total = result["phase_totals"].get(phase, 0)
        print(f"{phase}: {phase_total} PLANNER_INFEASIBLE step(s)")
        for check in FEASIBILITY_CHECKS:
            count = result["phase_check_counts"].get(phase, {}).get(check, 0)
            print(f"    {check:<24}{count:>6}{_fmt_pct(count, phase_total):>10}")

    print()
    print("-" * 60)
    print("C. OUTCOME x PHASE")
    print("-" * 60)
    for outcome, phases in result["outcome_phase_totals"].items():
        for phase, phase_total in phases.items():
            print(f"{outcome} / {phase}: {phase_total} PLANNER_INFEASIBLE step(s)")
            breakdown = result["outcome_phase_check_counts"].get(outcome, {}).get(phase, {})
            for check in FEASIBILITY_CHECKS:
                count = breakdown.get(check, 0)
                print(f"    {check:<24}{count:>6}{_fmt_pct(count, phase_total):>10}")

    print()
    print("-" * 60)
    print("D. Actual Value vs Threshold")
    print("-" * 60)
    for check in FEASIBILITY_CHECKS:
        values = result["violating_values"].get(check, [])
        if not values:
            print(f"{check}: no violating steps")
            continue
        print(f"{check}")
        print(f"  n            : {len(values)}")
        print(f"  median fail  : {statistics.median(values):.3f}")
        print(f"  mean fail    : {statistics.mean(values):.3f}")
        print(f"  max fail     : {max(values):.3f}")
        print(f"  min fail     : {min(values):.3f}")

    print()
    print("-" * 60)
    print("E. Action Breakdown")
    print("-" * 60)
    for action, counts in result["action_check_counts"].items():
        print(f"{action}:")
        for check in FEASIBILITY_CHECKS:
            print(f"    {check:<24}{counts.get(check, 0):>6}")


def _write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(result: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    _write_csv(
        os.path.join(output_dir, "failure_check_summary.csv"),
        [{"check": c, "count": result["check_counts"].get(c, 0)} for c in FEASIBILITY_CHECKS],
        ["check", "count"],
    )
    _write_csv(
        os.path.join(output_dir, "failure_combination_summary.csv"),
        [{"combination": combo, "count": count} for combo, count in result["combination_counts"].items()],
        ["combination", "count"],
    )

    phase_outcome_rows = []
    for outcome, phases in result["outcome_phase_totals"].items():
        for phase, total in phases.items():
            row = {"outcome": outcome, "phase": phase, "total": total}
            breakdown = result["outcome_phase_check_counts"].get(outcome, {}).get(phase, {})
            for check in FEASIBILITY_CHECKS:
                row[check] = breakdown.get(check, 0)
            phase_outcome_rows.append(row)
    _write_csv(
        os.path.join(output_dir, "phase_outcome_summary.csv"),
        phase_outcome_rows,
        ["outcome", "phase", "total", *FEASIBILITY_CHECKS],
    )

    violating_rows = []
    for check, values in result["violating_values"].items():
        for value in values:
            violating_rows.append({"check": check, "value": value})
    _write_csv(
        os.path.join(output_dir, "violating_values.csv"),
        violating_rows,
        ["check", "value"],
    )

    episode_fieldnames = [
        "maneuver_id", "outcome", "phase", "step_index", "action",
        "downstream_status", "checks_failed", *FEASIBILITY_VALUE_KEYS.values(),
    ]
    _write_csv(
        os.path.join(output_dir, "episode_summary.csv"),
        result["episode_rows"],
        episode_fieldnames,
    )


def main() -> None:
    args = build_parser().parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    restored = restore_ppo_checkpoint(
        args.checkpoint, expected_dataset_schema_version=MERGE_DATASET_SCHEMA,
    )
    reward_config_path = args.reward_config_path or restored.reward_config_path or "configs/reward.yaml"
    reward_config = load_reward_config(reward_config_path)
    print(f"Reward config: {reward_config_path} (reward_version={reward_config.reward_version})")

    maneuver_ids = _load_maneuver_ids(args.maneuver_id_file)
    print(f"Loaded {len(maneuver_ids)} maneuver_id(s) from {args.maneuver_id_file}")

    train_specs = {s.maneuver_id: s for s in load_decision_dataset_maneuver_specs("train")}
    missing = [m for m in maneuver_ids if m not in train_specs]
    if missing:
        raise ValueError(f"maneuver_id(s) not found in canonical TRAIN: {missing}")
    maneuvers = [train_specs[m] for m in maneuver_ids]

    env = MergeEnvironment(
        dataset_config_path=args.dataset_config_path,
        downstream_mode=args.downstream_mode,
        required_dataset_schema_version=MERGE_DATASET_SCHEMA,
    )

    print(f"Running {len(maneuvers)} deterministic episode(s) ...")
    episode_results = []
    for maneuver in maneuvers:
        result = run_ppo_episode(
            env=env,
            maneuver=maneuver,
            policy=restored.policy,
            value_network=restored.value_network,
            value_params=restored.value_params,
            reward_config=reward_config,
            max_steps=args.max_episode_steps,
            policy_mode=args.policy_mode,
        )
        episode_results.append(result)
        print(f"  {maneuver.maneuver_id}: outcome={result.outcome} steps={result.physical_step_count}")

    analysis = analyze(episode_results)
    print()
    print_summary(analysis)
    write_outputs(analysis, args.output_dir)
    print(f"\nWrote diagnostic outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
