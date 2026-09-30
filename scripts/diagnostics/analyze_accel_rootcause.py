#!/usr/bin/env python3
"""Root-cause diagnostic for longitudinal_accel PLANNER_INFEASIBLE
violations (read-only, diagnostic-only analysis).

Tests 3 hypotheses using per-step diagnostics already threaded through
info["downstream_diagnostics"] (src/environment/merge_environment.py,
src/planning/frenet_planner.py, src/environment/common_downstream.py):

  A. target-frame s_dot: does target-reference projection shrink
     ego_frenet.s_d well below the Cartesian ego_speed_mps during MERGE?
  B. reference_speed_mps: is delta_v (reference_speed - s_d) large
     because of a fixed/aggressive reference speed choice?
  C. fallback failure loop: does a PLANNER_INFEASIBLE step's fallback
     braking increase delta_v on the following step, causing repeated
     PLANNER_INFEASIBLE streaks?

Never modifies planner/controller/PPO/reward/dataset behavior.

Usage:

    PYTHONPATH=. python scripts/diagnostics/analyze_accel_rootcause.py \\
        --checkpoint outputs/checkpoints/full_seed0_step000036.pkl \\
        --maneuver-id-file outputs/diagnostics/train_diag64_seed20260928.txt \\
        --dataset-config-path configs/dataset.yaml \\
        --max-episode-steps 100 \\
        --output-dir outputs/diagnostics/accel_rootcause_step36
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from src.environment.full_split_evaluator import load_decision_dataset_maneuver_specs
from src.environment.merge_environment import MergeEnvironment
from src.scenarios.merge_v2 import MERGE_DATASET_SCHEMA
from src.training.config import load_reward_config
from src.visualization.ppo_checkpoint_policy import restore_ppo_checkpoint
from src.visualization.ppo_rollout import PPOStepRecord, run_ppo_episode

PHASE_PRE_COMMIT = "PRE_COMMIT_POLICY"
PHASE_POST_COMMIT = "POST_COMMIT_AUTO_MERGE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--maneuver-id-file", required=True)
    parser.add_argument("--dataset-config-path", default="configs/dataset.yaml")
    parser.add_argument("--downstream-mode", default="frenet_mpc")
    parser.add_argument("--reward-config-path", default=None)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--policy-mode", choices=("deterministic", "stochastic"), default="deterministic")
    parser.add_argument("--output-dir", required=True)
    return parser


def _load_maneuver_ids(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _phase(record: PPOStepRecord) -> str:
    return PHASE_POST_COMMIT if record.merge_committed_before else PHASE_PRE_COMMIT


def _action_label(record: PPOStepRecord) -> str:
    return record.selected_action if record.is_policy_step else "POST_COMMIT_AUTO_MERGE"


def _planner_diagnostics(record: PPOStepRecord) -> Optional[dict]:
    dd = record.info.get("downstream_diagnostics")
    if not dd:
        return None
    return dd.get("planner")


def build_step_rows(episode_results) -> List[dict]:
    """One row per physical step, across all episodes -- the raw
    dataset every hypothesis analysis below is computed from."""

    rows = []
    for episode in episode_results:
        for record in episode.steps:
            planner = _planner_diagnostics(record)
            ego_frenet = (planner or {}).get("ego_frenet_state") or {}
            profile = (planner or {}).get("generated_longitudinal_profile") or {}
            reference_lane = (planner or {}).get("reference_lane")
            reference_speed = (planner or {}).get("reference_speed_mps")
            delta_v = (planner or {}).get("delta_v_mps")
            checks_failed = tuple((planner or {}).get("checks_failed", ()))
            info = record.info
            follow_inputs = (info.get("downstream_diagnostics") or {}).get("follow_inputs") or {}
            target_front_gap_m = follow_inputs.get("target_front_gap_m")

            rows.append({
                "maneuver_id": episode.maneuver_id,
                "outcome": episode.outcome,
                "step_index": record.step_index,
                "phase": _phase(record),
                "selected_action": _action_label(record),
                "is_policy_step": record.is_policy_step,
                "ego_speed_mps": record.ego_speed_mps,
                "ego_x": record.ego_x,
                "ego_y": record.ego_y,
                "ego_yaw": record.ego_yaw,
                "reference_lane": reference_lane,
                "ego_frenet_s": ego_frenet.get("s"),
                "ego_frenet_s_d": ego_frenet.get("s_d"),
                "ego_frenet_s_dd": ego_frenet.get("s_dd"),
                "ego_frenet_d": ego_frenet.get("d"),
                "ego_frenet_d_d": ego_frenet.get("d_d"),
                "ego_frenet_d_dd": ego_frenet.get("d_dd"),
                "reference_speed_mps": reference_speed,
                "delta_v_mps": delta_v,
                "target_front_present": target_front_gap_m is not None,
                "target_front_gap_m": target_front_gap_m,
                "target_front_relative_speed_mps": follow_inputs.get("target_front_relative_speed_mps"),
                "min_s_dd_mps2": profile.get("min_s_dd_mps2"),
                "max_s_dd_mps2": profile.get("max_s_dd_mps2"),
                "max_abs_s_dd_mps2": profile.get("max_abs_s_dd_mps2"),
                "time_of_max_abs_s_dd_s": profile.get("time_of_max_abs_s_dd_s"),
                "min_s_d_mps": profile.get("min_s_d_mps"),
                "max_s_d_mps": profile.get("max_s_d_mps"),
                "terminal_s_d_mps": profile.get("terminal_s_d_mps"),
                "downstream_status": record.downstream_status,
                "checks_failed": ";".join(checks_failed),
                "longitudinal_accel_failed": "longitudinal_accel" in checks_failed,
                "fallback_applied": record.downstream_status not in (None, "OK"),
                "applied_acceleration_mps2": info.get("controller_acceleration_mps2"),
                "previous_downstream_status": info.get("previous_downstream_status"),
                "consecutive_planner_infeasible_steps": info.get("consecutive_planner_infeasible_steps"),
                "consecutive_fallback_steps": info.get("consecutive_fallback_steps"),
                "first_planner_infeasible_step": info.get("first_planner_infeasible_step"),
                "steps_since_first_planner_infeasible": info.get("steps_since_first_planner_infeasible"),
            })
    return rows


def _quantiles(values):
    if not values:
        return {"p10": None, "p50": None, "p90": None, "median": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
    }


def hypothesis_a_target_frame(rows: List[dict]) -> dict:
    """MERGE-only (reference_lane == "target") rows: compares
    ego_speed_mps (Cartesian) vs ego_frenet_s_d (target-frame
    projected) and their ratio, split OK vs PLANNER_INFEASIBLE and by
    outcome."""

    merge_rows = [r for r in rows if r["reference_lane"] == "target" and r["ego_frenet_s_d"] is not None]

    def _group(pred):
        g = [r for r in merge_rows if pred(r)]
        speeds = [r["ego_speed_mps"] for r in g]
        s_dots = [r["ego_frenet_s_d"] for r in g]
        ratios = [
            (r["ego_frenet_s_d"] / r["ego_speed_mps"])
            for r in g if r["ego_speed_mps"] and abs(r["ego_speed_mps"]) > 1e-6
        ]
        ref_speeds = [r["reference_speed_mps"] for r in g if r["reference_speed_mps"] is not None]
        delta_vs = [r["delta_v_mps"] for r in g if r["delta_v_mps"] is not None]
        return {
            "n": len(g),
            "ego_speed_mps": _quantiles(speeds),
            "target_frame_s_d_mps": _quantiles(s_dots),
            "s_d_over_speed_ratio": _quantiles(ratios),
            "reference_speed_mps": _quantiles(ref_speeds),
            "delta_v_mps": _quantiles(delta_vs),
        }

    result = {
        "OK": _group(lambda r: not r["longitudinal_accel_failed"]),
        "INFEASIBLE": _group(lambda r: r["longitudinal_accel_failed"]),
    }
    for outcome in ("OUTCOME_SUCCESS", "OUTCOME_COLLISION", "OUTCOME_TIMEOUT"):
        result[outcome] = _group(lambda r, o=outcome: r["outcome"] == o)
    return result


def hypothesis_b_reference_speed(rows: List[dict]) -> dict:
    """delta_v distribution, correlation(delta_v, max_abs_s_dd), and
    infeasible rate by delta_v bucket; target-front present vs absent
    comparison for MERGE rows."""

    valid = [r for r in rows if r["delta_v_mps"] is not None and r["max_abs_s_dd_mps2"] is not None]
    delta_vs = np.array([r["delta_v_mps"] for r in valid])
    max_abs_accels = np.array([r["max_abs_s_dd_mps2"] for r in valid])

    pearson = float(np.corrcoef(delta_vs, max_abs_accels)[0, 1]) if len(valid) > 1 else None
    try:
        from scipy.stats import spearmanr
        spearman = float(spearmanr(delta_vs, max_abs_accels).correlation) if len(valid) > 1 else None
    except ImportError:
        spearman = None

    buckets = [(0, 3), (3, 6), (6, 9), (9, 12), (12, float("inf"))]
    bucket_stats = []
    for lo, hi in buckets:
        in_bucket = [r for r in valid if lo <= abs(r["delta_v_mps"]) < hi]
        infeasible = [r for r in in_bucket if r["longitudinal_accel_failed"]]
        bucket_stats.append({
            "range": f"{lo}-{hi if hi != float('inf') else 'inf'}",
            "count": len(in_bucket),
            "infeasible_rate": (len(infeasible) / len(in_bucket)) if in_bucket else None,
        })

    merge_rows = [r for r in rows if r["reference_lane"] == "target"]

    def _merge_group(present: bool):
        g = [r for r in merge_rows if bool(r["target_front_present"]) == present]
        ref_speeds = [r["reference_speed_mps"] for r in g if r["reference_speed_mps"] is not None]
        delta_v_g = [r["delta_v_mps"] for r in g if r["delta_v_mps"] is not None]
        infeasible = [r for r in g if r["longitudinal_accel_failed"]]
        return {
            "n": len(g),
            "reference_speed_mps": _quantiles(ref_speeds),
            "delta_v_mps": _quantiles(delta_v_g),
            "infeasible_rate": (len(infeasible) / len(g)) if g else None,
        }

    return {
        "pearson_delta_v_vs_max_abs_accel": pearson,
        "spearman_delta_v_vs_max_abs_accel": spearman,
        "delta_v_buckets": bucket_stats,
        "merge_target_front_present": _merge_group(True),
        "merge_target_front_absent": _merge_group(False),
    }


def hypothesis_c_failure_streak(rows: List[dict], episode_results) -> dict:
    """Streak/transition analysis: P(next infeasible | current
    infeasible) vs P(next infeasible | current OK), before/after
    first-infeasible windows, per-episode max streak by outcome."""

    by_episode = defaultdict(list)
    for r in rows:
        by_episode[r["maneuver_id"]].append(r)
    for maneuver_id in by_episode:
        by_episode[maneuver_id].sort(key=lambda r: r["step_index"])

    transitions_from_infeasible = {"next_infeasible": 0, "next_ok": 0}
    transitions_from_ok = {"next_infeasible": 0, "next_ok": 0}
    episode_max_streak = {}
    episode_outcome = {}
    first_infeasible_windows = []

    for maneuver_id, steps in by_episode.items():
        episode_outcome[maneuver_id] = steps[0]["outcome"] if steps else None
        max_streak = 0
        current_streak = 0
        first_infeasible_idx = None
        for i, r in enumerate(steps):
            is_infeasible = r["longitudinal_accel_failed"]
            if is_infeasible:
                current_streak += 1
                if first_infeasible_idx is None:
                    first_infeasible_idx = i
            else:
                max_streak = max(max_streak, current_streak)
                current_streak = 0
            if i + 1 < len(steps):
                next_infeasible = steps[i + 1]["longitudinal_accel_failed"]
                bucket = transitions_from_infeasible if is_infeasible else transitions_from_ok
                bucket["next_infeasible" if next_infeasible else "next_ok"] += 1
        max_streak = max(max_streak, current_streak)
        episode_max_streak[maneuver_id] = max_streak

        if first_infeasible_idx is not None:
            window = {"maneuver_id": maneuver_id, "outcome": episode_outcome[maneuver_id]}
            pre = steps[first_infeasible_idx - 1] if first_infeasible_idx > 0 else None
            window["pre_ego_speed_mps"] = pre["ego_speed_mps"] if pre else None
            window["pre_s_d_mps"] = pre["ego_frenet_s_d"] if pre else None
            window["pre_reference_speed_mps"] = pre["reference_speed_mps"] if pre else None
            window["pre_delta_v_mps"] = pre["delta_v_mps"] if pre else None
            for offset in range(1, 6):
                idx = first_infeasible_idx + offset - 1
                if idx < len(steps):
                    s = steps[idx]
                    window[f"post{offset}_fallback_applied"] = s["fallback_applied"]
                    window[f"post{offset}_ego_speed_mps"] = s["ego_speed_mps"]
                    window[f"post{offset}_s_d_mps"] = s["ego_frenet_s_d"]
                    window[f"post{offset}_reference_speed_mps"] = s["reference_speed_mps"]
                    window[f"post{offset}_delta_v_mps"] = s["delta_v_mps"]
                    window[f"post{offset}_downstream_status"] = s["downstream_status"]
            first_infeasible_windows.append(window)

    def _p(bucket):
        total = bucket["next_infeasible"] + bucket["next_ok"]
        return (bucket["next_infeasible"] / total) if total else None

    streaks_by_outcome = defaultdict(list)
    for maneuver_id, streak in episode_max_streak.items():
        streaks_by_outcome[episode_outcome[maneuver_id]].append(streak)

    episode_streak_summary = []
    for outcome, streaks in streaks_by_outcome.items():
        n_with_infeasible = sum(1 for s in streaks if s > 0)
        episode_streak_summary.append({
            "outcome": outcome,
            "episodes_with_infeasible": n_with_infeasible,
            "total_episodes": len(streaks),
            "median_max_streak": statistics.median(streaks) if streaks else None,
            "max_streak_overall": max(streaks) if streaks else None,
        })

    streak_buckets = {"2_step": 0, "3_step": 0, "5plus_step": 0}
    for streak in episode_max_streak.values():
        if streak >= 5:
            streak_buckets["5plus_step"] += 1
        elif streak >= 3:
            streak_buckets["3_step"] += 1
        elif streak >= 2:
            streak_buckets["2_step"] += 1

    return {
        "p_next_infeasible_given_current_infeasible": _p(transitions_from_infeasible),
        "p_next_infeasible_given_current_ok": _p(transitions_from_ok),
        "streak_length_buckets": streak_buckets,
        "episode_streak_summary": episode_streak_summary,
        "first_infeasible_windows": first_infeasible_windows,
        "median_streak_all_episodes": (
            statistics.median(list(episode_max_streak.values())) if episode_max_streak else None
        ),
        "max_streak_all_episodes": (
            max(episode_max_streak.values()) if episode_max_streak else None
        ),
    }


def quartic_sanity_check(horizon_s: float = 3.0) -> List[dict]:
    """Numerically verifies the theoretical a_max ~ 1.5*|delta_v|/T
    approximation against the REAL QuarticPolynomial.solve used by
    candidate_generator.py, for a range of delta_v magnitudes."""

    from src.planning.polynomial import QuarticPolynomial

    results = []
    for delta_v in (3.0, 6.0, 9.0, 12.0, 15.0):
        poly = QuarticPolynomial.solve(
            p0=0.0, v0=0.0, a0_acc=0.0, vT=delta_v, aT=0.0, T=horizon_s,
        )
        t = np.linspace(0.0, horizon_s, 301)
        accel = poly.acceleration(t)
        max_abs_accel = float(np.max(np.abs(accel)))
        theoretical = 1.5 * abs(delta_v) / horizon_s
        results.append({
            "delta_v_mps": delta_v,
            "horizon_s": horizon_s,
            "numerical_max_abs_accel_mps2": max_abs_accel,
            "theoretical_approx_1_5_dv_over_T": theoretical,
        })
    return results


def print_summary(hyp_a, hyp_b, hyp_c, sanity) -> None:
    print("-" * 60)
    print("Hypothesis A: target-frame s_dot (MERGE rows only)")
    print("-" * 60)
    for key in ("OK", "INFEASIBLE"):
        g = hyp_a[key]
        print(f"{key} (n={g['n']}): ego_speed p50={g['ego_speed_mps']['p50']}, "
              f"target_s_d p50={g['target_frame_s_d_mps']['p50']}, "
              f"ratio p50={g['s_d_over_speed_ratio']['p50']}, "
              f"ref_speed p50={g['reference_speed_mps']['p50']}, "
              f"delta_v p50={g['delta_v_mps']['p50']}")

    print()
    print("-" * 60)
    print("Hypothesis B: reference_speed / delta_v")
    print("-" * 60)
    print(f"Pearson(delta_v, max_abs_accel)  = {hyp_b['pearson_delta_v_vs_max_abs_accel']}")
    print(f"Spearman(delta_v, max_abs_accel) = {hyp_b['spearman_delta_v_vs_max_abs_accel']}")
    print("delta_v range     count   infeasible_rate")
    for b in hyp_b["delta_v_buckets"]:
        print(f"  {b['range']:<14}{b['count']:>6}   {b['infeasible_rate']}")
    print(f"MERGE target-front present:  {hyp_b['merge_target_front_present']}")
    print(f"MERGE target-front absent:   {hyp_b['merge_target_front_absent']}")

    print()
    print("-" * 60)
    print("Hypothesis C: fallback failure loop")
    print("-" * 60)
    print(f"P(next infeasible | current infeasible) = {hyp_c['p_next_infeasible_given_current_infeasible']}")
    print(f"P(next infeasible | current OK)         = {hyp_c['p_next_infeasible_given_current_ok']}")
    print(f"Streak buckets: {hyp_c['streak_length_buckets']}")
    print("Outcome    episodes_w_infeasible/total   median_max_streak   max_streak")
    for s in hyp_c["episode_streak_summary"]:
        print(f"  {s['outcome']:<20}{s['episodes_with_infeasible']}/{s['total_episodes']}"
              f"    {s['median_max_streak']}    {s['max_streak_overall']}")

    print()
    print("-" * 60)
    print("Quartic sanity check (delta_v -> max |accel|)")
    print("-" * 60)
    for r in sanity:
        print(f"  delta_v={r['delta_v_mps']:>5.1f} m/s  numerical={r['numerical_max_abs_accel_mps2']:.3f} m/s^2  "
              f"theoretical(1.5*dv/T)={r['theoretical_approx_1_5_dv_over_T']:.3f} m/s^2")


def _write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(step_rows, hyp_a, hyp_b, hyp_c, sanity, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump({
            "hypothesis_a_target_frame": hyp_a,
            "hypothesis_b_reference_speed": hyp_b,
            "hypothesis_c_failure_streak": {
                k: v for k, v in hyp_c.items() if k != "first_infeasible_windows"
            },
            "quartic_sanity_check": sanity,
        }, f, indent=2, default=str)

    if step_rows:
        _write_csv(
            os.path.join(output_dir, "step_diagnostics.csv"),
            step_rows, list(step_rows[0].keys()),
        )

    with open(os.path.join(output_dir, "hypothesis_a_target_frame.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "n", "ego_speed_p50", "target_s_d_p50", "ratio_p50", "ref_speed_p50", "delta_v_p50"])
        for k, v in hyp_a.items():
            writer.writerow([
                k, v["n"], v["ego_speed_mps"]["p50"], v["target_frame_s_d_mps"]["p50"],
                v["s_d_over_speed_ratio"]["p50"], v["reference_speed_mps"]["p50"], v["delta_v_mps"]["p50"],
            ])

    _write_csv(
        os.path.join(output_dir, "hypothesis_b_reference_speed.csv"),
        hyp_b["delta_v_buckets"], ["range", "count", "infeasible_rate"],
    )

    _write_csv(
        os.path.join(output_dir, "hypothesis_c_failure_streak.csv"),
        hyp_c["first_infeasible_windows"],
        list(hyp_c["first_infeasible_windows"][0].keys()) if hyp_c["first_infeasible_windows"] else
        ["maneuver_id", "outcome"],
    )

    _write_csv(
        os.path.join(output_dir, "episode_streak_summary.csv"),
        hyp_c["episode_streak_summary"],
        ["outcome", "episodes_with_infeasible", "total_episodes", "median_max_streak", "max_streak_overall"],
    )


def main() -> None:
    args = build_parser().parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    restored = restore_ppo_checkpoint(
        args.checkpoint, expected_dataset_schema_version=MERGE_DATASET_SCHEMA,
    )
    reward_config_path = args.reward_config_path or restored.reward_config_path or "configs/reward.yaml"
    reward_config = load_reward_config(reward_config_path)

    maneuver_ids = _load_maneuver_ids(args.maneuver_id_file)
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
            env=env, maneuver=maneuver, policy=restored.policy,
            value_network=restored.value_network, value_params=restored.value_params,
            reward_config=reward_config, max_steps=args.max_episode_steps,
            policy_mode=args.policy_mode,
        )
        episode_results.append(result)
        print(f"  {maneuver.maneuver_id}: outcome={result.outcome} steps={result.physical_step_count}")

    step_rows = build_step_rows(episode_results)
    hyp_a = hypothesis_a_target_frame(step_rows)
    hyp_b = hypothesis_b_reference_speed(step_rows)
    hyp_c = hypothesis_c_failure_streak(step_rows, episode_results)
    sanity = quartic_sanity_check()

    print()
    print_summary(hyp_a, hyp_b, hyp_c, sanity)
    write_outputs(step_rows, hyp_a, hyp_b, hyp_c, sanity, args.output_dir)
    print(f"\nWrote diagnostic outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
