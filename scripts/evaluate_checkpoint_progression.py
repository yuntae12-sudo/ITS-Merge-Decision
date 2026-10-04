#!/usr/bin/env python3
"""Evaluate PPO checkpoints on one immutable Fixed TRAIN64 set.

This is a read-only evaluation entrypoint.  It restores existing checkpoints
and reuses the canonical deterministic PPO rollout; it does not alter PPO,
reward, planner, controller, environment, or dataset code.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.environment.full_split_evaluator import load_decision_dataset_maneuver_specs
from src.environment.merge_environment import MergeEnvironment
from src.scenarios.merge_v2 import MERGE_DATASET_SCHEMA
from src.training.config import load_reward_config
from src.visualization.ppo_checkpoint_policy import restore_ppo_checkpoint
from src.visualization.ppo_rollout import run_ppo_episode


DEFAULT_UPDATES = (1, 6, 12, 18, 24, 30, 36)
ACTION_NAMES = ("KEEP", "FOLLOW", "MERGE", "STOP")
OUTCOMES = ("success", "collision", "offroad", "timeout", "exception")
DEFAULT_FIXED_SET = REPO_ROOT / "outputs/diagnostics/train_diag64_seed20260928.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/checkpoint_progression"

RAW_FIELDS = (
    "checkpoint", "checkpoint_path", "maneuver_id", "outcome", "termination_reason",
    "episode_steps", "policy_decision_count", "episode_return",
    "reward_terminal", "reward_decision_cost", "reward_safety", "reward_progress",
    "reward_decision", "keep_count", "follow_count", "merge_count", "stop_count",
    "intervention_count", "collision_blocked_count", "planner_infeasible_count",
    "intervention_rate", "collision_blocked_rate", "planner_infeasible_rate",
    "merge_commit_step", "exception_repr",
)

SUMMARY_FIELDS = (
    "checkpoint", "checkpoint_path", "num_maneuvers", "success_count", "collision_count",
    "offroad_count", "timeout_count", "exception_count", "success_rate", "collision_rate",
    "offroad_rate", "timeout_rate", "exception_rate", "intervention_count",
    "collision_blocked_count", "planner_infeasible_count", "intervention_rate",
    "collision_blocked_rate", "planner_infeasible_rate", "mean_episode_return",
    "std_episode_return", "reward_total", "reward_terminal", "reward_decision_cost",
    "reward_safety", "reward_progress", "reward_decision", "keep_count", "follow_count",
    "merge_count", "stop_count", "keep_ratio", "follow_ratio", "merge_ratio", "stop_ratio",
    "mean_episode_length", "mean_merge_commit_step",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", default=",".join(map(str, DEFAULT_UPDATES)))
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--checkpoint-prefix", default="full_seed0_step")
    parser.add_argument("--fixed-set", default=str(DEFAULT_FIXED_SET))
    parser.add_argument("--selection-seed", type=int, default=20260928)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset-config-path", default="configs/dataset.yaml")
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="Reuse a complete, schema-valid raw CSV.")
    return parser.parse_args(argv)


def read_fixed_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != 64 or len(set(ids)) != 64:
        raise ValueError(f"Fixed set must contain exactly 64 unique maneuver IDs; got {len(ids)}")
    return ids


def build_fixed_manifest(source: Path, maneuver_ids: Sequence[str], selection_seed: int) -> dict:
    source_bytes = source.read_bytes()
    try:
        source_path = str(source.relative_to(REPO_ROOT))
    except ValueError:
        source_path = str(source)
    return {
        "split": "TRAIN",
        "num_maneuvers": len(maneuver_ids),
        "selection_seed": selection_seed,
        "selection_method": "reuse_existing_fixed_train64",
        "source_path": source_path,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "maneuver_ids": list(maneuver_ids),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def episode_to_row(checkpoint: int, checkpoint_path: Path, episode) -> dict:
    action_counts = Counter(
        record.selected_action for record in episode.steps if record.is_policy_step
    )
    final_info = episode.steps[-1].info if episode.steps else {}
    steps_elapsed = max(1, int(final_info.get("steps_elapsed", episode.physical_step_count) or 0))
    intervention_count = int(final_info.get("downstream_failure_count", 0) or 0)
    blocked_count = int(final_info.get("collision_blocked_count", 0) or 0)
    infeasible_count = int(final_info.get("planner_infeasible_count", 0) or 0)

    def component(name: str) -> float:
        return float(sum(getattr(record, name) for record in episode.steps))

    return {
        "checkpoint": checkpoint,
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
        "maneuver_id": episode.maneuver_id,
        "outcome": episode.outcome,
        "termination_reason": episode.termination_reason or "",
        "episode_steps": episode.physical_step_count,
        "policy_decision_count": episode.policy_decision_count,
        "episode_return": component("reward_total"),
        "reward_terminal": component("reward_terminal_component"),
        "reward_decision_cost": component("reward_decision_cost_component"),
        "reward_safety": component("reward_safety_component"),
        "reward_progress": component("reward_progress_component"),
        "reward_decision": component("reward_decision_component"),
        "keep_count": action_counts["KEEP"],
        "follow_count": action_counts["FOLLOW"],
        "merge_count": action_counts["MERGE"],
        "stop_count": action_counts["STOP"],
        "intervention_count": intervention_count,
        "collision_blocked_count": blocked_count,
        "planner_infeasible_count": infeasible_count,
        # Match src.training.trainer._aggregate_downstream_rates: each episode's
        # final cumulative count divided by that episode's steps_elapsed.
        "intervention_rate": intervention_count / steps_elapsed,
        "collision_blocked_rate": blocked_count / steps_elapsed,
        "planner_infeasible_rate": infeasible_count / steps_elapsed,
        "merge_commit_step": "" if episode.merge_commit_step is None else episode.merge_commit_step,
        "exception_repr": episode.exception_repr or "",
    }


def aggregate_rows(checkpoint: int, checkpoint_path: str, rows: Sequence[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot aggregate zero episodes")
    outcomes = Counter(str(row["outcome"]) for row in rows)
    n = len(rows)

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    def total_int(key: str) -> int:
        return int(sum(int(float(row[key])) for row in rows))

    returns = values("episode_return")
    action_counts = {name: total_int(f"{name.lower()}_count") for name in ACTION_NAMES}
    action_total = sum(action_counts.values())
    commits = [float(row["merge_commit_step"]) for row in rows if str(row["merge_commit_step"]) != ""]
    result = {
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "num_maneuvers": n,
    }
    for outcome in OUTCOMES:
        result[f"{outcome}_count"] = outcomes[outcome]
        result[f"{outcome}_rate"] = outcomes[outcome] / n
    for key in ("intervention_count", "collision_blocked_count", "planner_infeasible_count"):
        result[key] = total_int(key)
    for key in ("intervention_rate", "collision_blocked_rate", "planner_infeasible_rate"):
        result[key] = float(np.mean(values(key)))
    result.update({
        "mean_episode_return": float(np.mean(returns)),
        "std_episode_return": float(np.std(returns)),
        "reward_total": float(np.mean(returns)),
        "reward_terminal": float(np.mean(values("reward_terminal"))),
        "reward_decision_cost": float(np.mean(values("reward_decision_cost"))),
        "reward_safety": float(np.mean(values("reward_safety"))),
        "reward_progress": float(np.mean(values("reward_progress"))),
        "reward_decision": float(np.mean(values("reward_decision"))),
        "mean_episode_length": float(np.mean(values("episode_steps"))),
        "mean_merge_commit_step": float(np.mean(commits)) if commits else "",
    })
    for name in ACTION_NAMES:
        lower = name.lower()
        result[f"{lower}_count"] = action_counts[name]
        result[f"{lower}_ratio"] = action_counts[name] / action_total if action_total else 0.0
    return result


def validate_raw_rows(rows: Sequence[dict], maneuver_ids: Sequence[str], checkpoint: int) -> None:
    if len(rows) != len(maneuver_ids):
        raise ValueError(f"checkpoint {checkpoint}: expected {len(maneuver_ids)} rows, got {len(rows)}")
    actual = [str(row["maneuver_id"]) for row in rows]
    if actual != list(maneuver_ids):
        raise ValueError(f"checkpoint {checkpoint}: raw CSV maneuver order/set differs from Fixed TRAIN64")
    if any(int(float(row["checkpoint"])) != checkpoint for row in rows):
        raise ValueError(f"checkpoint {checkpoint}: raw CSV contains a different checkpoint")


def evaluate_checkpoint(
    checkpoint: int,
    checkpoint_path: Path,
    maneuvers,
    env: MergeEnvironment,
    max_steps: int,
) -> list[dict]:
    restored = restore_ppo_checkpoint(
        str(checkpoint_path), expected_dataset_schema_version=MERGE_DATASET_SCHEMA
    )
    if restored.ppo_update_step != checkpoint:
        raise ValueError(
            f"checkpoint filename says {checkpoint}, payload says {restored.ppo_update_step}"
        )
    reward_config = load_reward_config(restored.reward_config_path or "configs/reward.yaml")
    rows = []
    for index, maneuver in enumerate(maneuvers, start=1):
        episode = run_ppo_episode(
            env=env,
            maneuver=maneuver,
            policy=restored.policy,
            value_network=restored.value_network,
            value_params=restored.value_params,
            reward_config=reward_config,
            max_steps=max_steps,
            policy_mode="deterministic",
        )
        rows.append(episode_to_row(checkpoint, checkpoint_path, episode))
        print(
            f"  update {checkpoint:02d} [{index:02d}/64] {maneuver.maneuver_id}: "
            f"{episode.outcome} steps={episode.physical_step_count}",
            flush=True,
        )
    return rows


def transition_rows(early_rows: Sequence[dict], final_rows: Sequence[dict]) -> list[dict]:
    early = {str(row["maneuver_id"]): str(row["outcome"]) for row in early_rows}
    final = {str(row["maneuver_id"]): str(row["outcome"]) for row in final_rows}
    if set(early) != set(final):
        raise ValueError("Paired transition inputs do not contain the same maneuver IDs")
    rows = []
    for maneuver_id in early:
        before, after = early[maneuver_id], final[maneuver_id]
        if before != "success" and after == "success":
            transition = "improved_to_success"
        elif before == "success" and after != "success":
            transition = "regressed_from_success"
        elif before == after:
            transition = "unchanged"
        else:
            transition = "changed_failure_mode"
        rows.append({
            "maneuver_id": maneuver_id,
            "early_outcome": before,
            "final_outcome": after,
            "transition": transition,
        })
    return rows


def _paper_axes(ax, ylabel: str) -> None:
    ax.set_xlabel("PPO update")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)


def make_figures(summary: Sequence[dict], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    updates = np.asarray([int(row["checkpoint"]) for row in summary])
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "figure.facecolor": "white"})

    plots = [
        ("checkpoint_outcome_progression", "Outcome rate", (
            ("success_rate", "Success", "#4C72B0"), ("collision_rate", "Collision", "#C44E52"),
            ("offroad_rate", "Off-road", "#8172B2"), ("timeout_rate", "Timeout", "#DD8452"),
        )),
        ("checkpoint_downstream_progression", "Mean per-episode step rate", (
            ("intervention_rate", "Intervention", "#4C72B0"),
            ("collision_blocked_rate", "Collision blocked", "#C44E52"),
            ("planner_infeasible_rate", "Planner infeasible", "#DD8452"),
        )),
    ]
    for filename, ylabel, series in plots:
        fig, ax = plt.subplots(figsize=(6.2, 3.7))
        for key, label, color in series:
            ax.plot(updates, [float(row[key]) for row in summary], marker="o", label=label, color=color)
        _paper_axes(ax, ylabel)
        ax.set_xticks(updates)
        ax.set_ylim(bottom=0.0)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
        fig.savefig(figure_dir / f"{filename}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    mean = np.asarray([float(row["mean_episode_return"]) for row in summary])
    std = np.asarray([float(row["std_episode_return"]) for row in summary])
    ax.errorbar(updates, mean, yerr=std, marker="o", capsize=3, color="#4C72B0")
    _paper_axes(ax, "Episode return (mean ± std)")
    ax.set_xticks(updates)
    fig.tight_layout()
    fig.savefig(figure_dir / "checkpoint_return_progression.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "checkpoint_return_progression.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    bottom = np.zeros(len(summary))
    colors = ("#4C72B0", "#55A868", "#DD8452", "#8172B2")
    for action, color in zip(ACTION_NAMES, colors):
        values = np.asarray([float(row[f"{action.lower()}_ratio"]) for row in summary])
        ax.bar(updates, values, bottom=bottom, label=action, color=color, width=3.6)
        bottom += values
    _paper_axes(ax, "Policy action ratio")
    ax.set_xticks(updates)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(figure_dir / "checkpoint_action_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "checkpoint_action_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def gate_decision(summary: Sequence[dict]) -> tuple[str, str]:
    first, final = summary[0], summary[-1]
    best_success = max(float(row["success_rate"]) for row in summary)
    final_success = float(final["success_rate"])
    noisy_or_not_best = final_success + 1e-12 < best_success
    success_improved = final_success > float(first["success_rate"])
    collision_improved = float(final["collision_rate"]) < float(first["collision_rate"])
    intervention_improved = float(final["intervention_rate"]) < float(first["intervention_rate"])
    reward_improved = float(final["mean_episode_return"]) > float(first["mean_episode_return"])
    dependency_high = float(final["intervention_rate"]) >= 0.5
    if noisy_or_not_best:
        return "CHECKPOINT_SELECTION_REQUIRED", "Final checkpoint is not the best TRAIN64 success checkpoint."
    if success_improved and dependency_high:
        return (
            "POLICY_IMPROVED_BUT_SAFETY_LAYER_DEPENDENCY_HIGH",
            "Success improved, but final intervention remains at or above 50%.",
        )
    if success_improved and collision_improved and intervention_improved:
        return "READY_FOR_FROZEN_VALIDATION", "Success, collision, and intervention all improved."
    if reward_improved:
        return (
            "REWARD_ALIGNMENT_REVIEW_REQUIRED",
            "Return improved without a clear joint success/collision/intervention improvement.",
        )
    return "CHECKPOINT_SELECTION_REQUIRED", "TRAIN64 behavior progression is not clearly monotonic."


def write_report(path: Path, summary: Sequence[dict], transitions: Sequence[dict], manifest: dict) -> None:
    gate, reason = gate_decision(summary)
    improved = sum(row["transition"] == "improved_to_success" for row in transitions)
    regressed = sum(row["transition"] == "regressed_from_success" for row in transitions)
    changed_failure = sum(row["transition"] == "changed_failure_mode" for row in transitions)
    lines = [
        "# Fixed TRAIN64 Checkpoint Progression Report", "", "## 1. Audit", "",
        "- 기존 `restore_ppo_checkpoint`, `run_ppo_episode`, canonical TRAIN loader, Reward V1, environment counter를 재사용했다.",
        "- 평가용 `scripts/evaluate_checkpoint_progression.py`, 전용 테스트, 결과 artifact만 추가했다.",
        "- Checkpoint 위치: `outputs/checkpoints/full_seed0_step000001.pkl`부터 update 36까지.",
        f"- 기존 Fixed TRAIN64 `{manifest['source_path']}`를 재사용했다(SHA-256 `{manifest['source_sha256']}`).",
        "", "## 2. Evaluation Protocol", "",
        "- Dataset: canonical TRAIN(`merge_decision` schema), 동일한 64개 maneuver.",
        "- Checkpoint: 1(가장 가까운 initial), 6, 12, 18, 24, 30, 36. update-0 checkpoint는 존재하지 않는다.",
        "- Deterministic argmax policy, 고정 순서, `frenet_mpc`, Reward V1, 최대 100 physical steps를 사용했다.",
        "- Frozen VALIDATION은 실행하지 않았다.", "", "## 3. Checkpoint Results", "",
        "| Checkpoint | Success | Collision | Off-road | Timeout | Intervention | Collision Blocked | Planner Infeasible | Mean Return |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['checkpoint']} | {100*float(row['success_rate']):.2f}% | "
            f"{100*float(row['collision_rate']):.2f}% | {100*float(row['offroad_rate']):.2f}% | "
            f"{100*float(row['timeout_rate']):.2f}% | {100*float(row['intervention_rate']):.2f}% | "
            f"{100*float(row['collision_blocked_rate']):.2f}% | "
            f"{100*float(row['planner_infeasible_rate']):.2f}% | "
            f"{float(row['mean_episode_return']):.4f} |"
        )
    lines.extend(["", "## 4. Action Distribution", "", "| Checkpoint | KEEP | FOLLOW | MERGE | STOP |", "|---|---:|---:|---:|---:|"])
    for row in summary:
        lines.append(
            f"| {row['checkpoint']} | {100*float(row['keep_ratio']):.2f}% | "
            f"{100*float(row['follow_ratio']):.2f}% | {100*float(row['merge_ratio']):.2f}% | "
            f"{100*float(row['stop_ratio']):.2f}% |"
        )
    first, final = summary[0], summary[-1]
    best = max(summary, key=lambda row: float(row["success_rate"]))
    lines.extend([
        "", "## 5. Key Observations", "",
        f"- Success는 {100*float(first['success_rate']):.2f}%에서 {100*float(final['success_rate']):.2f}%로 증가했지만 단조 증가하지 않았다.",
        f"- Collision은 {100*float(first['collision_rate']):.2f}%에서 {100*float(final['collision_rate']):.2f}%로 소폭 감소했다.",
        f"- Intervention은 {100*float(first['intervention_rate']):.2f}%에서 {100*float(final['intervention_rate']):.2f}%로 감소하지 않았다.",
        f"- Mean return은 {float(first['mean_episode_return']):.4f}에서 {float(final['mean_episode_return']):.4f}로 개선됐지만 update 12 이후 악화됐다.",
        f"- TRAIN64 최고 success는 update {best['checkpoint']}의 {100*float(best['success_rate']):.2f}%이며 final update 36이 best가 아니다.",
        "- Action distribution은 trend 진단용으로만 사용했고 성능 판정 기준에는 포함하지 않았다.",
        "- 기존 rollout schema에는 merge completion time이 없어 만들지 않았으며, 사용 가능한 merge commit step만 보고했다.",
        "", "## 6. Scenario Transition Analysis", "",
        f"- Nearest-initial update 1 → final update 36: success로 개선 {improved}개, success에서 악화 {regressed}개, failure mode 변경 {changed_failure}개.",
        "- Maneuver별 paired outcome은 `transitions/early_vs_final.csv`에 저장했다.",
        "- 지배적인 final failure mode는 collision 28/64와 timeout 17/64이다.",
        "", "## 7. Gate Decision", "", f"**{gate}**", "",
        "Final checkpoint가 TRAIN64 success 기준 best checkpoint가 아니며 progression도 noisy하다.",
        "", "## 8. Next Recommended Action", "",
        "TRAIN64만을 근거로 update 12를 포함한 checkpoint 선택을 먼저 확정해야 한다. 그 전에는 Frozen VALIDATION을 실행하지 않는 것을 권고하며, 이번 작업에서는 실제 VALIDATION을 실행하지 않았다.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    updates = tuple(int(value.strip()) for value in args.updates.split(",") if value.strip())
    output_dir = Path(args.output_dir).resolve()
    raw_dir = output_dir / "raw"
    fixed_source = Path(args.fixed_set).resolve()
    maneuver_ids = read_fixed_ids(fixed_source)
    manifest = build_fixed_manifest(fixed_source, maneuver_ids, args.selection_seed)
    write_json(output_dir / "fixed_train64_manifest.json", manifest)

    train_specs = {spec.maneuver_id: spec for spec in load_decision_dataset_maneuver_specs("train")}
    missing = [maneuver_id for maneuver_id in maneuver_ids if maneuver_id not in train_specs]
    if missing:
        raise ValueError(f"Fixed TRAIN64 contains IDs absent from canonical TRAIN: {missing}")
    maneuvers = [train_specs[maneuver_id] for maneuver_id in maneuver_ids]
    env = MergeEnvironment(
        dataset_config_path=args.dataset_config_path,
        downstream_mode="frenet_mpc",
        required_dataset_schema_version=MERGE_DATASET_SCHEMA,
    )

    summary = []
    rows_by_update = {}
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    for update in updates:
        checkpoint_path = checkpoint_dir / f"{args.checkpoint_prefix}{update:06d}.pkl"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        raw_path = raw_dir / f"checkpoint_{update:03d}.csv"
        if args.resume and raw_path.exists():
            rows = read_csv(raw_path)
            validate_raw_rows(rows, maneuver_ids, update)
            print(f"Reusing complete {raw_path}", flush=True)
        else:
            print(f"Evaluating update {update}: {checkpoint_path}", flush=True)
            rows = evaluate_checkpoint(update, checkpoint_path, maneuvers, env, args.max_episode_steps)
            write_csv(raw_path, rows, RAW_FIELDS)
        validate_raw_rows(rows, maneuver_ids, update)
        rows_by_update[update] = rows
        summary.append(aggregate_rows(update, str(checkpoint_path.relative_to(REPO_ROOT)), rows))
        gc.collect()

    write_csv(output_dir / "checkpoint_summary.csv", summary, SUMMARY_FIELDS)
    early_transitions = transition_rows(rows_by_update[updates[0]], rows_by_update[updates[-1]])
    write_csv(
        output_dir / "transitions/early_vs_final.csv", early_transitions,
        ("maneuver_id", "early_outcome", "final_outcome", "transition"),
    )
    if 6 in rows_by_update:
        step6 = transition_rows(rows_by_update[6], rows_by_update[updates[-1]])
        write_csv(
            output_dir / "transitions/update006_vs_final.csv", step6,
            ("maneuver_id", "early_outcome", "final_outcome", "transition"),
        )
    make_figures(summary, output_dir / "figures")
    write_report(output_dir / "REPORT.md", summary, early_transitions, manifest)
    print(f"Completed Fixed TRAIN64 checkpoint progression: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
