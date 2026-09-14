"""Phase 3 Stage 3-G: canonical 168-maneuver robustness/regression audit.

Loads every canonical maneuver (110 TRAIN + 58 VALIDATION, via
``src.environment.full_split_evaluator.load_maneuver_specs``) and, for
EACH one under ``downstream_mode="frenet_mpc"``, drives a bounded
rollout with a fixed, precedented action script (MERGE every step --
matching ``test_merge_environment.py``'s own success-test convention)
purely as a robustness stimulus. This script does NOT evaluate FSM/PPO
decision quality, does NOT require a target success rate, and does NOT
tune anything -- it only asks "does the new downstream stack survive
every real maneuver in the canonical pool without crashing, producing
NaN/Inf, or leaving EpisodeContext in an invalid state?".

Gate (see docs/phase3/OVERNIGHT_PROGRESS.md Stage 3-G entry): zero
unhandled exceptions, zero systematic NaN/Inf, across all 168
maneuvers. A maneuver ending in collision/offroad/timeout is NOT itself
a gate failure.

Output: a compact per-maneuver CSV + JSON summary under
outputs/phase3/robustness/ (168 rows, no raw trajectory dumps) plus an
aggregate stdout report.
"""

import collections
import csv
import dataclasses
import json
import math
import time
from typing import Optional

import numpy as np

from src.environment.behavior_action import BehaviorAction
from src.environment.full_split_evaluator import load_maneuver_specs
from src.environment.merge_environment import MergeEnvironment

DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"
MAX_STEPS = 120  # generous bound above MAX_EPISODE_HORIZON_FRAMES-derived per-episode horizons
OUTPUT_DIR = "outputs/phase3/robustness"
DOWNSTREAM_STATUSES = ["OK", "INVALID_REFERENCE", "PLANNER_INFEASIBLE", "COLLISION_BLOCKED", "CONTROLLER_FAILURE"]


@dataclasses.dataclass
class ManeuverAuditResult:
    maneuver_id: str
    split: str
    scene_key: Optional[str]
    source_shard: str
    record_index: int
    is_chained: bool
    lane_chain_length: int
    steps_executed: int
    termination_reason: Optional[str]
    terminated: bool
    truncated: bool
    exception: Optional[str]
    nan_or_inf_observed: bool
    nan_or_inf_detail: Optional[str]
    downstream_status_counts: dict
    chain_advanced_count: int
    final_active_transition_index: Optional[int]
    transition_index_out_of_bounds: bool
    success: bool
    collision: bool
    offroad: bool
    wall_time_s: float


def _is_finite_array(arr) -> bool:
    arr = np.asarray(arr, dtype=np.float64)
    return bool(np.all(np.isfinite(arr)))


def audit_one_maneuver(env: MergeEnvironment, spec, split: str) -> ManeuverAuditResult:
    is_chained = len(spec.lane_chain) > 2

    downstream_status_counts = {s: 0 for s in DOWNSTREAM_STATUSES}
    chain_advanced_count = 0
    nan_or_inf_observed = False
    nan_or_inf_detail = None
    exception = None
    termination_reason = None
    terminated = False
    truncated = False
    success = False
    collision = False
    offroad = False
    steps_executed = 0
    final_transition_index = None
    transition_index_out_of_bounds = False
    scene_key = None

    start = time.perf_counter()
    try:
        observation, info = env.reset(spec)
        scene_key = env._episode_context.scene_key

        if not _is_finite_array(observation):
            nan_or_inf_observed = True
            nan_or_inf_detail = "non-finite value in reset() observation"

        for step_idx in range(MAX_STEPS):
            observation, reward, terminated, truncated, info = env.step(BehaviorAction.MERGE)
            steps_executed = step_idx + 1

            if not _is_finite_array(observation):
                nan_or_inf_observed = True
                nan_or_inf_detail = (
                    nan_or_inf_detail
                    or f"non-finite value in observation at step {step_idx}"
                )

            accel = info.get("controller_acceleration_mps2")
            curvature = info.get("controller_steering_curvature")
            for name, value in (("acceleration", accel), ("curvature", curvature)):
                if value is not None and not math.isfinite(value):
                    nan_or_inf_observed = True
                    nan_or_inf_detail = (
                        nan_or_inf_detail
                        or f"non-finite controller {name} at step {step_idx}"
                    )

            ds = info.get("downstream_status")
            if ds is not None:
                downstream_status_counts[ds] = downstream_status_counts.get(ds, 0) + 1

            if info.get("chain_advanced"):
                chain_advanced_count += 1

            transition_index = info.get("active_transition_index")
            if transition_index is not None:
                if not (0 <= transition_index <= len(spec.lane_chain) - 2):
                    transition_index_out_of_bounds = True
                final_transition_index = transition_index

            if terminated or truncated:
                termination_reason = info.get("termination_reason")
                break
        else:
            termination_reason = "truncation_horizon_script_cap"
            truncated = True

        success = termination_reason == "success"
        collision = termination_reason == "failure_collision"
        offroad = termination_reason == "failure_offroad"

    except Exception as exc:  # noqa: BLE001 -- deliberately broad: one
        # maneuver's exception must not abort the full 168-maneuver
        # audit; it is recorded as its own gate-relevant field instead.
        exception = f"{type(exc).__name__}: {exc}"

    wall_time_s = time.perf_counter() - start

    return ManeuverAuditResult(
        maneuver_id=spec.maneuver_id,
        split=split,
        scene_key=scene_key,
        source_shard=spec.source_shard,
        record_index=spec.record_index,
        is_chained=is_chained,
        lane_chain_length=len(spec.lane_chain),
        steps_executed=steps_executed,
        termination_reason=termination_reason,
        terminated=bool(terminated),
        truncated=bool(truncated),
        exception=exception,
        nan_or_inf_observed=nan_or_inf_observed,
        nan_or_inf_detail=nan_or_inf_detail,
        downstream_status_counts=downstream_status_counts,
        chain_advanced_count=chain_advanced_count,
        final_active_transition_index=final_transition_index,
        transition_index_out_of_bounds=transition_index_out_of_bounds,
        success=success,
        collision=collision,
        offroad=offroad,
        wall_time_s=wall_time_s,
    )


def main():
    env = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH, downstream_mode="frenet_mpc")

    train_specs = load_maneuver_specs("train")
    val_specs = load_maneuver_specs("validation")
    print(f"Loaded {len(train_specs)} TRAIN + {len(val_specs)} VALIDATION maneuvers "
          f"= {len(train_specs) + len(val_specs)} total.")

    results = []
    for split, specs in (("train", train_specs), ("validation", val_specs)):
        for i, spec in enumerate(specs):
            result = audit_one_maneuver(env, spec, split)
            results.append(result)
            status = "OK" if result.exception is None and not result.nan_or_inf_observed else "FLAG"
            print(
                f"[{split} {i+1}/{len(specs)}] {spec.maneuver_id}: "
                f"steps={result.steps_executed} term={result.termination_reason} "
                f"chained={result.is_chained} chain_adv={result.chain_advanced_count} "
                f"exc={result.exception} nan={result.nan_or_inf_observed} [{status}]"
            )

    # ------------------------------------------------------------------
    # Save compact per-maneuver summary (CSV + JSON), no raw trajectories.
    # ------------------------------------------------------------------
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, "audit_168_maneuvers.csv")
    json_path = os.path.join(OUTPUT_DIR, "audit_168_maneuvers.json")

    fieldnames = [f.name for f in dataclasses.fields(ManeuverAuditResult)]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = dataclasses.asdict(r)
            row["downstream_status_counts"] = json.dumps(row["downstream_status_counts"])
            writer.writerow(row)

    with open(json_path, "w") as f:
        json.dump([dataclasses.asdict(r) for r in results], f, indent=2)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    n = len(results)
    n_exceptions = sum(1 for r in results if r.exception is not None)
    n_nan = sum(1 for r in results if r.nan_or_inf_observed)
    n_oob = sum(1 for r in results if r.transition_index_out_of_bounds)
    termination_counts = collections.Counter(r.termination_reason for r in results)
    n_with_planner_failure = sum(
        1 for r in results
        if any(r.downstream_status_counts.get(s, 0) > 0
               for s in ("INVALID_REFERENCE", "PLANNER_INFEASIBLE", "COLLISION_BLOCKED", "CONTROLLER_FAILURE"))
    )
    chained = [r for r in results if r.is_chained]
    n_chained = len(chained)
    n_chained_advanced = sum(1 for r in chained if r.chain_advanced_count > 0)

    all_status_totals = collections.Counter()
    for r in results:
        for s, c in r.downstream_status_counts.items():
            all_status_totals[s] += c

    print("\n=== Stage 3-G 168-maneuver audit: aggregate summary ===")
    print(f"Total maneuvers audited: {n}")
    print(f"Zero exceptions: {n - n_exceptions}/{n} (exceptions: {n_exceptions})")
    print(f"Zero NaN/Inf: {n - n_nan}/{n} (flagged: {n_nan})")
    print(f"Transition-index-out-of-bounds: {n_oob}/{n}")
    print(f"Termination reason distribution: {dict(termination_counts)}")
    print(f"Maneuvers with >=1 planner/controller non-OK status: {n_with_planner_failure}/{n}")
    print(f"Chained maneuvers (lane_chain length > 2): {n_chained}")
    print(f"Chained maneuvers where chain advanced >=1 time: {n_chained_advanced}/{n_chained}")
    print(f"downstream_status distribution across all steps of all maneuvers: {dict(all_status_totals)}")

    gate_pass = (n_exceptions == 0) and (n_nan == 0) and (n_oob == 0)
    print(f"\nGATE (zero exceptions, zero NaN/Inf, zero out-of-bounds transition index): "
          f"{'PASS' if gate_pass else 'FAIL'}")

    if n_exceptions > 0:
        print("\nManeuvers with exceptions:")
        for r in results:
            if r.exception is not None:
                print(f"  {r.maneuver_id} ({r.split}): {r.exception}")

    if n_nan > 0:
        print("\nManeuvers with NaN/Inf:")
        for r in results:
            if r.nan_or_inf_observed:
                print(f"  {r.maneuver_id} ({r.split}): {r.nan_or_inf_detail}")

    print("\nChained maneuver detail:")
    for r in chained:
        print(
            f"  {r.maneuver_id} ({r.split}): lane_chain_length={r.lane_chain_length} "
            f"chain_advanced={r.chain_advanced_count} final_idx={r.final_active_transition_index} "
            f"term={r.termination_reason} steps={r.steps_executed}"
        )

    return gate_pass


if __name__ == "__main__":
    import sys
    passed = main()
    sys.exit(0 if passed else 1)
