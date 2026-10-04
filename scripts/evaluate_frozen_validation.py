#!/usr/bin/env python3
"""Freeze PPO update 12 and evaluate it once on canonical VALIDATION 388.

This entrypoint is evaluation-only.  It references (but never rewrites) the
selected checkpoint and frozen dataset, uses deterministic PPO rollout, and
persists each episode immediately so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.evaluate_checkpoint_progression import RAW_FIELDS, episode_to_row
from src.environment.full_split_evaluator import (
    DECISION_EVIDENCE_TRAINING,
    DECISION_EVIDENCE_VALIDATION,
    DECISION_MANEUVER_TABLE,
    DECISION_SPLIT_MANIFEST,
    load_decision_dataset_maneuver_specs,
)
from src.environment.merge_environment import MergeEnvironment
from src.scenarios.merge_v2 import MERGE_DATASET_SCHEMA
from src.training.config import load_reward_config
from src.visualization.outcome_scan import select_representative_episodes
from src.visualization.ppo_checkpoint_policy import restore_ppo_checkpoint
from src.visualization.ppo_rollout import run_ppo_episode


SELECTED_UPDATE = 12
EXPECTED_VALIDATION_COUNT = 388
ACTION_NAMES = ("KEEP", "FOLLOW", "MERGE", "STOP")
OUTCOMES = ("success", "collision", "offroad", "timeout", "exception")
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs/checkpoints/full_seed0_step000012.pkl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/frozen_validation"
DEFAULT_VALIDATION_DATASET_CONFIG = "configs/dataset_validation.yaml"
TRAIN64_REFERENCE = {
    "success_rate": 0.3125,
    "collision_rate": 0.4375,
    "offroad_rate": 0.03125,
    "timeout_rate": 0.21875,
    "intervention_rate": 0.5889048706113006,
    "collision_blocked_rate": 0.5161551522007777,
    "planner_infeasible_rate": 0.07274971841052288,
    "mean_episode_return": -0.09137217333664463,
}

SUMMARY_FIELDS = (
    "checkpoint", "checkpoint_path", "num_episodes", "success_count",
    "collision_count", "offroad_count", "timeout_count", "exception_count",
    "success_rate", "collision_rate", "offroad_rate", "timeout_rate",
    "exception_rate", "intervention_count", "collision_blocked_count",
    "planner_infeasible_count", "intervention_rate", "collision_blocked_rate",
    "planner_infeasible_rate", "mean_episode_return", "std_episode_return",
    "median_episode_return", "reward_total", "reward_terminal",
    "reward_decision_cost", "reward_safety", "reward_progress", "reward_decision",
    "keep_count", "follow_count", "merge_count", "stop_count", "keep_ratio",
    "follow_ratio", "merge_ratio", "stop_ratio", "mean_episode_length",
    "mean_merge_commit_step",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset-config-path", default=DEFAULT_VALIDATION_DATASET_CONFIG)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build_validation_manifest(validation_specs, train_specs) -> dict:
    ids = [spec.maneuver_id for spec in validation_specs]
    train_ids = {spec.maneuver_id for spec in train_specs}
    sources = [
        DECISION_MANEUVER_TABLE,
        DECISION_SPLIT_MANIFEST,
        DECISION_EVIDENCE_VALIDATION,
    ]
    payload = {
        "split": "VALIDATION",
        "dataset_schema_version": MERGE_DATASET_SCHEMA,
        "ordering": "lexicographic_maneuver_id",
        "num_maneuvers": len(ids),
        "train_validation_maneuver_id_overlap": len(train_ids.intersection(ids)),
        "sources": [
            {"path": source, "sha256": sha256_file(REPO_ROOT / source)}
            for source in sources
        ],
        "maneuver_ids": ids,
    }
    encoded_ids = "\n".join(ids).encode("utf-8") + b"\n"
    payload["ordered_maneuver_ids_sha256"] = hashlib.sha256(encoded_ids).hexdigest()
    return payload


def validate_resume_prefix(rows: Sequence[dict], maneuver_ids: Sequence[str]) -> None:
    if len(rows) > len(maneuver_ids):
        raise ValueError("Raw CSV has more rows than Frozen VALIDATION")
    actual_ids = [str(row["maneuver_id"]) for row in rows]
    if actual_ids != list(maneuver_ids[:len(rows)]):
        raise ValueError("Raw CSV is not an exact ordered prefix of Frozen VALIDATION")
    if any(int(float(row["checkpoint"])) != SELECTED_UPDATE for row in rows):
        raise ValueError("Raw CSV contains a checkpoint other than update 12")


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def aggregate_rows(rows: Sequence[dict], checkpoint_path: str) -> dict:
    if len(rows) != EXPECTED_VALIDATION_COUNT:
        raise ValueError(f"Expected 388 rows, got {len(rows)}")
    outcomes = Counter(str(row["outcome"]) for row in rows)
    unknown = set(outcomes).difference(OUTCOMES)
    if unknown:
        raise ValueError(f"Unknown outcome(s): {sorted(unknown)}")
    n = len(rows)

    def values(key):
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    def total_int(key):
        return int(sum(int(float(row[key])) for row in rows))

    result = {
        "checkpoint": SELECTED_UPDATE,
        "checkpoint_path": checkpoint_path,
        "num_episodes": n,
    }
    for outcome in OUTCOMES:
        result[f"{outcome}_count"] = outcomes[outcome]
        result[f"{outcome}_rate"] = outcomes[outcome] / n
    for key in ("intervention_count", "collision_blocked_count", "planner_infeasible_count"):
        result[key] = total_int(key)
    for key in ("intervention_rate", "collision_blocked_rate", "planner_infeasible_rate"):
        result[key] = float(np.mean(values(key)))
    returns = values("episode_return")
    result.update({
        "mean_episode_return": float(np.mean(returns)),
        "std_episode_return": float(np.std(returns)),
        "median_episode_return": float(np.median(returns)),
        "reward_total": float(np.mean(returns)),
        "reward_terminal": float(np.mean(values("reward_terminal"))),
        "reward_decision_cost": float(np.mean(values("reward_decision_cost"))),
        "reward_safety": float(np.mean(values("reward_safety"))),
        "reward_progress": float(np.mean(values("reward_progress"))),
        "reward_decision": float(np.mean(values("reward_decision"))),
        "mean_episode_length": float(np.mean(values("episode_steps"))),
    })
    commits = [float(row["merge_commit_step"]) for row in rows if str(row["merge_commit_step"]) != ""]
    result["mean_merge_commit_step"] = float(np.mean(commits)) if commits else ""
    action_total = sum(total_int(f"{name.lower()}_count") for name in ACTION_NAMES)
    for name in ACTION_NAMES:
        lower = name.lower()
        count = total_int(f"{lower}_count")
        result[f"{lower}_count"] = count
        result[f"{lower}_ratio"] = count / action_total if action_total else 0.0
    return result


def representative_selection(rows: Sequence[dict]) -> dict:
    minimal_results = [
        SimpleNamespace(maneuver_id=str(row["maneuver_id"]), outcome=str(row["outcome"]))
        for row in rows
    ]
    return {
        "selector": "src.visualization.outcome_scan.select_representative_episodes",
        "ordering": "lexicographic_maneuver_id",
        "max_per_outcome": 5,
        "selection": select_representative_episodes(minimal_results, max_per_outcome=5),
    }


def _style(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7, zorder=0)


def make_figures(summary: dict, rows: Sequence[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "figure.facecolor": "white"})

    def bar_figure(filename, labels, counts, rates, ylabel):
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        bars = ax.bar(labels, rates, color=("#4C72B0", "#C44E52", "#8172B2", "#DD8452")[:len(labels)], zorder=2)
        _style(ax, ylabel)
        ax.set_ylim(0, max(rates) * 1.22 if max(rates) else 1)
        for bar, count, rate in zip(bars, counts, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{count}\n({100*rate:.1f}%)", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")
        plt.close(fig)

    outcome_keys = ("success", "collision", "offroad", "timeout")
    bar_figure("validation_outcome_distribution", ("Success", "Collision", "Off-road", "Timeout"),
               [summary[f"{k}_count"] for k in outcome_keys], [summary[f"{k}_rate"] for k in outcome_keys], "Episode rate")
    downstream_keys = ("intervention", "collision_blocked", "planner_infeasible")
    bar_figure("validation_downstream_metrics", ("Intervention", "Collision\nblocked", "Planner\ninfeasible"),
               [summary[f"{k}_count"] for k in downstream_keys], [summary[f"{k}_rate"] for k in downstream_keys], "Mean per-episode step rate")
    action_keys = ("keep", "follow", "merge", "stop")
    bar_figure("validation_action_distribution", ACTION_NAMES,
               [summary[f"{k}_count"] for k in action_keys], [summary[f"{k}_ratio"] for k in action_keys], "Policy decision ratio")

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.hist([float(row["episode_return"]) for row in rows], bins=24, color="#4C72B0", edgecolor="white")
    _style(ax, "Episodes")
    ax.set_xlabel("Episode return")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_return_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "validation_return_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def gate_decisions(summary: dict) -> list[tuple[str, str]]:
    gates = []
    outcome_sum = sum(int(summary[f"{name}_count"]) for name in OUTCOMES)
    action_sum = sum(float(summary[f"{name.lower()}_ratio"]) for name in ACTION_NAMES)
    if summary["exception_count"] or outcome_sum != EXPECTED_VALIDATION_COUNT or abs(action_sum - 1.0) > 1e-9:
        gates.append(("VALIDATION_PIPELINE_REVIEW_REQUIRED", "exception 또는 집계 consistency anomaly가 존재한다."))
    if float(summary["intervention_rate"]) >= 0.5 or float(summary["collision_rate"]) >= 0.4:
        gates.append(("HIGH_SAFETY_LAYER_DEPENDENCY", "intervention ≥ 50% 또는 collision ≥ 40%이다."))
    if float(summary["success_rate"]) - TRAIN64_REFERENCE["success_rate"] <= -0.10:
        gates.append(("GENERALIZATION_GAP_REVIEW_REQUIRED", "VALIDATION success가 TRAIN64보다 10%p 이상 낮다."))
    if not gates:
        gates.append(("READY_FOR_FSM_BASELINE_COMPARISON", "runtime anomaly와 큰 success 붕괴가 없고 safety threshold 미만이다."))
    return gates


def write_report(path: Path, summary: dict, freeze: dict, representatives: dict) -> None:
    labels = {"success": "Success", "collision": "Collision", "offroad": "Off-road", "timeout": "Timeout"}
    lines = [
        "# Frozen VALIDATION 388 Report — PPO Update 12", "", "## 1. Frozen Checkpoint", "",
        f"- Selected update: {SELECTED_UPDATE}", f"- Checkpoint path: `{freeze['checkpoint_path']}`",
        f"- Checkpoint SHA-256: `{freeze['checkpoint_sha256']}`",
        "- TRAIN64 success 최고, timeout 최저, mean return 최고를 근거로 선택했다.",
        "- Validation은 checkpoint 선택에 사용하지 않았다.", "", "## 2. Validation Protocol", "",
        f"- Dataset: canonical Frozen VALIDATION (`{MERGE_DATASET_SCHEMA}`), {summary['num_episodes']} maneuvers",
        "- Deterministic argmax, current frozen `frenet_mpc`, Reward V1, 최대 100 physical steps",
        f"- Exception count: {summary['exception_count']}", "", "## 3. Outcome Results", "",
        "| Metric | Count | Rate |", "|---|---:|---:|",
    ]
    for key in labels:
        lines.append(f"| {labels[key]} | {summary[f'{key}_count']} | {100*float(summary[f'{key}_rate']):.2f}% |")
    lines += ["", "## 4. Downstream Results", "", "| Metric | Count | Rate |", "|---|---:|---:|"]
    for key, label in (("intervention", "Intervention"), ("collision_blocked", "Collision Blocked"), ("planner_infeasible", "Planner Infeasible")):
        lines.append(f"| {label} | {summary[f'{key}_count']} | {100*float(summary[f'{key}_rate']):.2f}% |")
    lines += ["", "Rate는 기존 trainer와 동일하게 각 episode의 final cumulative count / physical steps를 계산한 뒤 episode 평균한 값이다.",
              "", "## 5. Reward Results", "",
              f"- Mean / std / median return: {float(summary['mean_episode_return']):.4f} / {float(summary['std_episode_return']):.4f} / {float(summary['median_episode_return']):.4f}",
              f"- Mean components — terminal {float(summary['reward_terminal']):.4f}, safety {float(summary['reward_safety']):.4f}, progress {float(summary['reward_progress']):.4f}, decision {float(summary['reward_decision']):.4f}, decision-cost {float(summary['reward_decision_cost']):.4f}",
              "", "## 6. Action Distribution", "", "| Action | Count | Ratio |", "|---|---:|---:|"]
    for action in ACTION_NAMES:
        key = action.lower()
        lines.append(f"| {action} | {summary[f'{key}_count']} | {100*float(summary[f'{key}_ratio']):.2f}% |")
    lines += ["", "## 7. TRAIN64 vs VALIDATION", "", "| Metric | TRAIN64 U12 | VALIDATION | Gap |", "|---|---:|---:|---:|"]
    comparison = (("success_rate", "Success", True), ("collision_rate", "Collision", True), ("offroad_rate", "Off-road", True), ("timeout_rate", "Timeout", True), ("intervention_rate", "Intervention", True), ("collision_blocked_rate", "Collision Blocked", True), ("planner_infeasible_rate", "Planner Infeasible", True), ("mean_episode_return", "Mean Return", False))
    for key, label, percent in comparison:
        train, val = TRAIN64_REFERENCE[key], float(summary[key])
        if percent:
            lines.append(f"| {label} | {100*train:.2f}% | {100*val:.2f}% | {100*(val-train):+.2f}%p |")
        else:
            lines.append(f"| {label} | {train:.4f} | {val:.4f} | {val-train:+.4f} |")
    selection = representatives["selection"]
    lines += ["", "## 8. Representative Episodes", ""]
    for key in ("success", "collision", "offroad", "timeout"):
        lines.append(f"- {labels[key]}: {', '.join(selection[key]['maneuver_ids']) or '(none)'}")
    dominant = max(("collision", "offroad", "timeout"), key=lambda key: int(summary[f"{key}_count"]))
    lines += ["", "## 9. Key Observations", "",
              f"- Success gap은 {100*(float(summary['success_rate'])-TRAIN64_REFERENCE['success_rate']):+.2f}%p이다.",
              f"- Dominant failure mode는 {dominant} ({summary[f'{dominant}_count']} episodes)이다.",
              f"- Intervention rate는 {100*float(summary['intervention_rate']):.2f}%이다.",
              "- Return/action distribution은 descriptive diagnostic이며 checkpoint 재선택에 사용하지 않는다.",
              "", "## 10. Gate Decision", ""]
    for gate, reason in gate_decisions(summary):
        lines += [f"**{gate}** — {reason}", ""]
    lines += ["## 11. Next Recommended Action", "",
              "동일 Frozen VALIDATION에서 FSM baseline comparison으로 진행하거나, 위 gate가 요구한 generalization/safety/pipeline 분석을 먼저 수행한다. Checkpoint 재선택은 하지 않는다."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    raw_path = output_dir / "validation_episode_results.csv"

    validation_specs = sorted(load_decision_dataset_maneuver_specs("validation"), key=lambda spec: spec.maneuver_id)
    train_specs = load_decision_dataset_maneuver_specs("train")
    manifest = build_validation_manifest(validation_specs, train_specs)
    if manifest["num_maneuvers"] != EXPECTED_VALIDATION_COUNT:
        raise ValueError(f"Canonical VALIDATION count is {manifest['num_maneuvers']}, expected 388")
    if manifest["train_validation_maneuver_id_overlap"] != 0:
        raise ValueError("TRAIN/VALIDATION maneuver overlap is nonzero")

    restored = restore_ppo_checkpoint(str(checkpoint_path), expected_dataset_schema_version=MERGE_DATASET_SCHEMA)
    if restored.ppo_update_step != SELECTED_UPDATE:
        raise ValueError(f"Only update 12 is authorized; checkpoint payload is update {restored.ppo_update_step}")
    freeze = {
        "selection_source": "fixed_train64_checkpoint_progression",
        "selected_update": SELECTED_UPDATE,
        "selection_metric_primary": "success_rate",
        "train64_success_rate": TRAIN64_REFERENCE["success_rate"],
        "train64_timeout_rate": TRAIN64_REFERENCE["timeout_rate"],
        "train64_mean_return": TRAIN64_REFERENCE["mean_episode_return"],
        "checkpoint_path": relative(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_seed": restored.seed,
        "checkpoint_reward_version": restored.reward_version,
        "checkpoint_dataset_schema_version": restored.dataset_schema_version,
        "checkpoint_git_sha": restored.checkpoint_git_sha,
        "validation_used_for_selection": False,
    }
    write_json(output_dir / "selected_checkpoint.json", freeze)
    write_json(output_dir / "frozen_validation_manifest.json", manifest)

    ids = manifest["maneuver_ids"]
    if raw_path.exists():
        if not args.resume:
            raise FileExistsError(f"{raw_path} exists; use --resume to validate and continue")
        rows = read_csv(raw_path)
        validate_resume_prefix(rows, ids)
    else:
        rows = []

    if len(rows) < EXPECTED_VALIDATION_COUNT:
        env = MergeEnvironment(dataset_config_path=args.dataset_config_path, downstream_mode="frenet_mpc", required_dataset_schema_version=MERGE_DATASET_SCHEMA)
        reward_config = load_reward_config(restored.reward_config_path or "configs/reward.yaml")
        for index in range(len(rows), len(validation_specs)):
            maneuver = validation_specs[index]
            episode = run_ppo_episode(env=env, maneuver=maneuver, policy=restored.policy,
                                      value_network=restored.value_network, value_params=restored.value_params,
                                      reward_config=reward_config, max_steps=args.max_episode_steps,
                                      policy_mode="deterministic")
            row = episode_to_row(SELECTED_UPDATE, checkpoint_path, episode)
            append_row(raw_path, row)
            rows.append(row)
            print(f"[{index+1:03d}/{EXPECTED_VALIDATION_COUNT}] {maneuver.maneuver_id}: {episode.outcome} steps={episode.physical_step_count}", flush=True)

    validate_resume_prefix(rows, ids)
    summary = aggregate_rows(rows, relative(checkpoint_path))
    write_csv(output_dir / "validation_summary.csv", [summary], SUMMARY_FIELDS)
    representatives = representative_selection(rows)
    write_json(output_dir / "representative_episodes.json", representatives)
    make_figures(summary, rows, output_dir / "figures")
    write_report(output_dir / "REPORT.md", summary, freeze, representatives)
    print(f"Completed Frozen VALIDATION 388: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
