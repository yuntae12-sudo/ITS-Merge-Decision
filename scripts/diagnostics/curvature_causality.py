#!/usr/bin/env python3
"""Curvature causality diagnostic (read-only, diagnostic-only).

Determines whether NEW curvature failures introduced after the
Planner-side acceleration-feasibility fix (commit 7f2cf16) are a
DIRECT_REGRESSION of the feasibility shaping itself, or an
EXISTING_GEOMETRY_FAILURE that was merely exposed because episodes no
longer terminate early via the (now-eliminated) acceleration-driven
collision loop.

Method: targeted replay of ONLY the maneuvers containing a NEW
curvature failure (identified by set-difference against the
pre-Planner-Fix episode_summary.csv), extracting the full per-step
planner state at each NEW-curvature step. For each such step, this
script performs a SAME-STATE counterfactual using the already
-recorded (ego_frenet, reference_speed_mps, horizon) as boundary
conditions:

  Counterfactual A (SHAPING OFF): generate the quartic candidate
    targeting the REQUESTED reference_speed_mps directly (bypassing
    _solve_feasible_terminal_speed).
  Counterfactual B (SHAPING ON): generate the quartic candidate via
    the current production _solve_feasible_terminal_speed (identical
    to what the real rollout used).

Both candidates are evaluated with the SAME exact
_exact_max_abs_longitudinal_accel/curvature-equivalent evaluation
(reusing candidate_evaluator.evaluate_candidate) -- never re-stepping
the environment, never changing simulated dynamics.

This never modifies any threshold, planner algorithm, or runtime
behavior -- purely a read-only counterfactual analysis tool.
"""

import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from src.environment.full_split_evaluator import load_decision_dataset_maneuver_specs
from src.environment.merge_environment import MergeEnvironment
from src.planning.candidate_evaluator import CollisionLimits, FeasibilityLimits, evaluate_candidate
from src.planning.candidate_generator import (
    _solve_feasible_terminal_speed,
    generate_follow_or_merge_candidate,
    project_cartesian_to_frame,
)
from src.planning.frenet_planner import load_planner_config
from src.scenarios.merge_v2 import MERGE_DATASET_SCHEMA
from src.training.config import load_reward_config
from src.visualization.ppo_checkpoint_policy import restore_ppo_checkpoint
from src.visualization.ppo_rollout import run_ppo_episode

OUTPUT_DIR = "outputs/diagnostics/curvature_causality"
CURVATURE_THRESHOLD = 0.3
ACCEL_THRESHOLD = 6.0


def identify_new_curvature_failures():
    with open("outputs/diagnostics/planner_infeasible_step36/episode_summary.csv") as f:
        before_rows = list(csv.DictReader(f))
    with open("outputs/diagnostics/planner_infeasible_step36_plannerfix/episode_summary.csv") as f:
        after_rows = list(csv.DictReader(f))

    before_curv = {(r["maneuver_id"], int(r["step_index"])) for r in before_rows if "curvature" in r["checks_failed"].split(";")}
    after_curv_rows = [r for r in after_rows if "curvature" in r["checks_failed"].split(";")]
    after_curv_by_key = {(r["maneuver_id"], int(r["step_index"])): r for r in after_curv_rows}
    after_curv_keys = set(after_curv_by_key.keys())

    new_keys = after_curv_keys - before_curv
    retained_keys = after_curv_keys & before_curv
    return new_keys, retained_keys, after_curv_by_key


def replay_and_capture(maneuver_ids, checkpoint_path, dataset_config_path, max_episode_steps):
    restored = restore_ppo_checkpoint(checkpoint_path, expected_dataset_schema_version=MERGE_DATASET_SCHEMA)
    reward_config = load_reward_config(restored.reward_config_path or "configs/reward.yaml")

    train_specs = {s.maneuver_id: s for s in load_decision_dataset_maneuver_specs("train")}
    maneuvers = [train_specs[m] for m in maneuver_ids]

    env = MergeEnvironment(
        dataset_config_path=dataset_config_path, downstream_mode="frenet_mpc",
        required_dataset_schema_version=MERGE_DATASET_SCHEMA,
    )

    captured = {}
    for maneuver in maneuvers:
        result = run_ppo_episode(
            env=env, maneuver=maneuver, policy=restored.policy,
            value_network=restored.value_network, value_params=restored.value_params,
            reward_config=reward_config, max_steps=max_episode_steps, policy_mode="deterministic",
        )
        for record in result.steps:
            planner = (record.info.get("downstream_diagnostics") or {}).get("planner")
            if planner is None:
                continue
            captured[(maneuver.maneuver_id, record.step_index)] = {
                "outcome": result.outcome,
                "phase": "POST_COMMIT_AUTO_MERGE" if record.merge_committed_before else "PRE_COMMIT_POLICY",
                "action": record.selected_action if record.is_policy_step else "POST_COMMIT_AUTO_MERGE",
                "ego_x": record.ego_x, "ego_y": record.ego_y, "ego_yaw": record.ego_yaw,
                "ego_speed_mps": record.ego_speed_mps,
                "ego_frenet_state": planner.get("ego_frenet_state"),
                "reference_lane": planner.get("reference_lane"),
                "requested_reference_speed_mps": planner.get("reference_speed_mps"),
                "terminal_speed_feasibility": planner.get("terminal_speed_feasibility"),
            }
    return captured, env


def _feasibility_limits():
    planner_config = load_planner_config()
    return planner_config, planner_config.feasibility_limits, planner_config.collision_limits


def counterfactual_for_step(state, planner_config, feasibility_limits, collision_limits):
    """Pure candidate-generation + evaluation counterfactual: OFF
    (requested speed directly) vs ON (current production shaping),
    from the SAME captured ego_frenet/reference boundary condition.
    Never touches the real environment."""

    from src.planning.frenet_types import FrenetState
    from src.planning.candidate_generator import _longitudinal_quartic_velocity_keeping, _build_frenet_path, _lateral_quintic

    ego_frenet_dict = state["ego_frenet_state"]
    if ego_frenet_dict is None:
        return None
    ego_frenet = FrenetState(
        s=ego_frenet_dict["s"], s_d=ego_frenet_dict["s_d"], s_dd=ego_frenet_dict["s_dd"],
        d=ego_frenet_dict["d"], d_d=ego_frenet_dict["d_d"], d_dd=ego_frenet_dict["d_dd"],
    )
    requested_speed = state["requested_reference_speed_mps"]
    horizon_s = planner_config.trajectory_horizon_s
    dt_s = planner_config.dt_s

    # OFF: quartic targeting requested speed directly (bypass shaping).
    longitudinal_off, _ = _longitudinal_quartic_velocity_keeping(
        ego_frenet.s, ego_frenet.s_d, ego_frenet.s_dd, requested_speed, horizon_s,
        max_longitudinal_accel_mps2=None,
    )
    lateral_off = _lateral_quintic(ego_frenet.d, ego_frenet.d_d, ego_frenet.d_dd, horizon_s)
    path_off = _build_frenet_path(lateral_off, longitudinal_off, horizon_s, dt_s)

    # ON: production shaping (identical to what the real rollout used).
    feasibility = _solve_feasible_terminal_speed(
        ego_frenet.s_d, ego_frenet.s_dd, requested_speed, horizon_s,
        feasibility_limits.max_longitudinal_accel_mps2,
    )
    longitudinal_on, _ = _longitudinal_quartic_velocity_keeping(
        ego_frenet.s, ego_frenet.s_d, ego_frenet.s_dd, requested_speed, horizon_s,
        max_longitudinal_accel_mps2=feasibility_limits.max_longitudinal_accel_mps2,
    )
    lateral_on = _lateral_quintic(ego_frenet.d, ego_frenet.d_d, ego_frenet.d_dd, horizon_s)
    path_on = _build_frenet_path(lateral_on, longitudinal_on, horizon_s, dt_s)

    # Need a ReferenceLine for curvature evaluation -- reuse the env's
    # own active reference (source/target per reference_lane), passed
    # in by the caller via state["_reference"].
    reference = state["_reference"]
    eval_off = evaluate_candidate(path_off, reference, feasibility_limits, collision_limits)
    eval_on = evaluate_candidate(path_on, reference, feasibility_limits, collision_limits)

    return {
        "requested_reference_speed_mps": requested_speed,
        "feasible_terminal_speed_mps": feasibility.feasible_terminal_speed_mps,
        "terminal_speed_was_limited": feasibility.terminal_speed_was_limited,
        "delta_terminal_speed": requested_speed - feasibility.feasible_terminal_speed_mps,
        "max_abs_curvature_off": eval_off.feasibility_values.get("max_abs_curvature_per_m"),
        "max_abs_accel_off": eval_off.feasibility_values.get("max_abs_longitudinal_accel_mps2"),
        "max_abs_curvature_on": eval_on.feasibility_values.get("max_abs_curvature_per_m"),
        "max_abs_accel_on": eval_on.feasibility_values.get("max_abs_longitudinal_accel_mps2"),
        "ego_frenet": ego_frenet_dict,
    }


def classify(cf, curvature_threshold=CURVATURE_THRESHOLD):
    off_fail = cf["max_abs_curvature_off"] > curvature_threshold
    on_fail = cf["max_abs_curvature_on"] > curvature_threshold
    if not off_fail and on_fail:
        return "DIRECT_REGRESSION"
    if off_fail and on_fail:
        return "EXISTING_GEOMETRY_FAILURE"
    if off_fail and not on_fail:
        return "SHAPING_IMPROVED"
    # Neither fails -- inconsistent with this being a curvature-failure
    # step in the real rollout; flag as numerical edge for inspection.
    return "NUMERICAL_EDGE_NEITHER_FAILS"


def main():
    new_keys, retained_keys, after_curv_by_key = identify_new_curvature_failures()
    print(f"NEW curvature failures: {len(new_keys)}")
    print(f"RETAINED curvature failures: {len(retained_keys)}")

    affected_maneuvers = sorted({k[0] for k in new_keys} | {k[0] for k in retained_keys})
    print(f"Affected maneuvers (targeted replay): {len(affected_maneuvers)}")

    captured, env = replay_and_capture(
        affected_maneuvers,
        checkpoint_path="outputs/checkpoints/full_seed0_step000036.pkl",
        dataset_config_path="configs/dataset.yaml",
        max_episode_steps=100,
    )

    planner_config, feasibility_limits, collision_limits = _feasibility_limits()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []
    missing_state = []

    for key in sorted(new_keys | retained_keys):
        maneuver_id, step_index = key
        state = captured.get(key)
        if state is None:
            missing_state.append(key)
            continue

        # Reference line resolution deferred to the reference_cache pass
        # below (must correspond to the SAME episode/scene the step was
        # captured from).
        state["_key"] = key
        all_results.append(state)

    # Reference lines are per-maneuver/per-active-transition -- resolve
    # them by replaying each affected maneuver once more (env reset only,
    # reference lines are geometry, not policy-dependent) and caching.
    reference_cache = {}
    train_specs = {s.maneuver_id: s for s in load_decision_dataset_maneuver_specs("train")}
    for maneuver_id in affected_maneuvers:
        env.reset(train_specs[maneuver_id])
        source_ref, target_ref = env._get_active_reference_lines()
        reference_cache[maneuver_id] = {"source": source_ref, "target": target_ref}

    counterfactual_rows = []
    for state in all_results:
        maneuver_id, step_index = state["_key"]
        reference_lane = state["reference_lane"]
        refs = reference_cache.get(maneuver_id)
        if refs is None or reference_lane not in refs:
            continue
        state["_reference"] = refs[reference_lane]
        cf = counterfactual_for_step(state, planner_config, feasibility_limits, collision_limits)
        if cf is None:
            continue
        category = classify(cf)
        row = {
            "maneuver_id": maneuver_id,
            "step_index": step_index,
            "is_new": (maneuver_id, step_index) in new_keys,
            "outcome": state["outcome"],
            "phase": state["phase"],
            "action": state["action"],
            "ego_speed_mps": state["ego_speed_mps"],
            "ego_x": state["ego_x"], "ego_y": state["ego_y"], "ego_yaw": state["ego_yaw"],
            **{f"frenet_{k}": v for k, v in cf["ego_frenet"].items()},
            "requested_reference_speed_mps": cf["requested_reference_speed_mps"],
            "feasible_terminal_speed_mps": cf["feasible_terminal_speed_mps"],
            "terminal_speed_was_limited": cf["terminal_speed_was_limited"],
            "delta_terminal_speed": cf["delta_terminal_speed"],
            "max_abs_curvature_off": cf["max_abs_curvature_off"],
            "max_abs_accel_off": cf["max_abs_accel_off"],
            "max_abs_curvature_on": cf["max_abs_curvature_on"],
            "max_abs_accel_on": cf["max_abs_accel_on"],
            "delta_curvature": cf["max_abs_curvature_on"] - cf["max_abs_curvature_off"],
            "category": category,
        }
        counterfactual_rows.append(row)

    if missing_state:
        print(f"WARNING: {len(missing_state)} (maneuver,step) keys had no captured planner "
              f"state (likely episode terminated before reaching that step in this replay -- "
              f"deterministic policy should reproduce identically, but flagging explicitly): "
              f"{missing_state[:5]}{'...' if len(missing_state) > 5 else ''}")

    # Write CSVs.
    new_curvature_path = os.path.join(OUTPUT_DIR, "new_curvature_failures.csv")
    new_rows_for_csv = [r for r in counterfactual_rows if r["is_new"]]
    if new_rows_for_csv:
        with open(new_curvature_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(new_rows_for_csv[0].keys()))
            writer.writeheader()
            for r in new_rows_for_csv:
                writer.writerow(r)

    counterfactual_path = os.path.join(OUTPUT_DIR, "counterfactual_results.csv")
    if counterfactual_rows:
        with open(counterfactual_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(counterfactual_rows[0].keys()))
            writer.writeheader()
            for r in counterfactual_rows:
                writer.writerow(r)

    # Summary stats.
    new_rows = [r for r in counterfactual_rows if r["is_new"]]
    retained_rows = [r for r in counterfactual_rows if not r["is_new"]]

    category_counts_new = Counter(r["category"] for r in new_rows)
    category_counts_retained = Counter(r["category"] for r in retained_rows)

    print()
    print("=" * 60)
    print(f"Counterfactual coverage: {len(counterfactual_rows)}/{len(new_keys) + len(retained_keys)} "
          f"target steps successfully replayed")
    print("=" * 60)

    print()
    print("=" * 60)
    print("Category breakdown -- NEW curvature failures")
    print("=" * 60)
    total_new = len(new_rows)
    for cat in ("DIRECT_REGRESSION", "EXISTING_GEOMETRY_FAILURE", "SHAPING_IMPROVED", "NUMERICAL_EDGE_NEITHER_FAILS"):
        count = category_counts_new.get(cat, 0)
        rate = (count / total_new * 100) if total_new else 0.0
        print(f"  {cat:<30}{count:>6}   {rate:.1f}%")
    print(f"  {'Total':<30}{total_new:>6}   100%")

    direct_regression_rows = [r for r in new_rows if r["category"] == "DIRECT_REGRESSION"]
    if direct_regression_rows:
        print()
        print("DIRECT_REGRESSION detail:")
        print(f"  count: {len(direct_regression_rows)}")
        print(f"  % of new curvature cases: {len(direct_regression_rows)/total_new*100:.1f}%")
        print(f"  outcome distribution: {Counter(r['outcome'] for r in direct_regression_rows)}")
        curv_increases = [r["delta_curvature"] for r in direct_regression_rows]
        print(f"  median curvature increase: {statistics.median(curv_increases):.4f}")
        print(f"  max curvature increase: {max(curv_increases):.4f}")
        speed_reductions = [r["delta_terminal_speed"] for r in direct_regression_rows]
        print(f"  median terminal-speed reduction: {statistics.median(speed_reductions):.4f}")
        print(f"  max terminal-speed reduction: {max(speed_reductions):.4f}")
        print(f"  affected maneuvers: {len(set(r['maneuver_id'] for r in direct_regression_rows))}")

    print()
    print("=" * 60)
    print("Category breakdown -- RETAINED (78) curvature failures")
    print("=" * 60)
    total_retained = len(retained_rows)
    for cat in ("DIRECT_REGRESSION", "EXISTING_GEOMETRY_FAILURE", "SHAPING_IMPROVED", "NUMERICAL_EDGE_NEITHER_FAILS"):
        count = category_counts_retained.get(cat, 0)
        rate = (count / total_retained * 100) if total_retained else 0.0
        print(f"  {cat:<30}{count:>6}   {rate:.1f}%")

    # Correlation analysis (new rows only).
    print()
    print("=" * 60)
    print("Mechanism / correlation analysis (NEW curvature failures)")
    print("=" * 60)
    if len(new_rows) > 1:
        speed_reductions = np.array([r["delta_terminal_speed"] for r in new_rows])
        curv_increases = np.array([r["delta_curvature"] for r in new_rows])
        ego_speeds = np.array([r["ego_speed_mps"] for r in new_rows])
        d_values = np.array([abs(r["frenet_d"]) for r in new_rows])
        d_d_values = np.array([abs(r["frenet_d_d"]) for r in new_rows])

        def _corr(a, b, name):
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                print(f"  {name}: undefined (constant input)")
                return
            pearson = float(np.corrcoef(a, b)[0, 1])
            try:
                from scipy.stats import spearmanr
                spearman = float(spearmanr(a, b).correlation)
            except ImportError:
                spearman = None
            print(f"  {name}: Pearson={pearson:.4f} Spearman={spearman}")

        _corr(speed_reductions, curv_increases, "terminal-speed reduction vs curvature increase")
        _corr(ego_speeds, curv_increases, "ego speed vs curvature increase")
        _corr(d_values, curv_increases, "|d| (lateral offset) vs curvature increase")
        _corr(d_d_values, curv_increases, "|d_d| (lateral rate) vs curvature increase")

    result_summary = {
        "new_curvature_count": len(new_keys),
        "retained_curvature_count": len(retained_keys),
        "counterfactual_coverage": len(counterfactual_rows),
        "category_counts_new": dict(category_counts_new),
        "category_counts_retained": dict(category_counts_retained),
        "direct_regression_count_new": len(direct_regression_rows),
        "missing_state_count": len(missing_state),
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(result_summary, f, indent=2, default=str)

    print(f"\nWrote {new_curvature_path}")
    print(f"Wrote {counterfactual_path}")
    print(f"Wrote {OUTPUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
