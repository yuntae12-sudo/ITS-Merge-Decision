"""Phase 3 Stage 3-G.5 + 3-G.6: performance snapshot (measurement only,
no optimization) and legacy-vs-frenet_mpc diagnostic comparison.

3-G.5: measures p50/p95 wall-clock timing for the main pipeline stages
across a representative sample of real maneuver rollouts, using
lightweight time.perf_counter() instrumentation in THIS script (not
modifying production code) plus the timing diagnostics
frenet_planner.plan() already returns in PlanResult.diagnostics
["timing_s"].

3-G.6: runs the SAME fixed action script (MERGE repeatedly) through
BOTH downstream_mode="legacy" and downstream_mode="frenet_mpc" on a
fixed subset of real maneuvers, and records (does not assert-equal)
termination reason / collision / offroad flags side by side. Purely
diagnostic, for Stage 3-H's later human review -- no research
conclusion is drawn here.
"""

import time

import numpy as np

from src.environment.behavior_action import BehaviorAction
from src.environment.common_downstream import CommonDownstream
from src.environment.full_split_evaluator import load_maneuver_specs
from src.environment.merge_environment import MergeEnvironment
from src.control.ltv_mpc import LtvMpcController
from src.planning.reference import ReferenceLine

DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"
N_STEPS = 25
PERF_SAMPLE_SIZE = 12
COMPARISON_SAMPLE_SIZE = 12


def percentile(values, p):
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values), p))


def run_performance_snapshot():
    print("=" * 70)
    print("Stage 3-G.5: performance snapshot (frenet_mpc mode)")
    print("=" * 70)

    env = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH, downstream_mode="frenet_mpc")
    specs = load_maneuver_specs("train")
    stride = max(len(specs) // PERF_SAMPLE_SIZE, 1)
    sample = specs[::stride][:PERF_SAMPLE_SIZE]

    reference_construction_s = []
    planner_plan_s = []
    mpc_solve_s = []
    common_downstream_step_s = []
    env_step_total_s = []

    orig_ref_build = ReferenceLine.from_lane_polyline
    def timed_ref_build(polyline):
        t0 = time.perf_counter()
        result = orig_ref_build(polyline)
        reference_construction_s.append(time.perf_counter() - t0)
        return result
    ReferenceLine.from_lane_polyline = staticmethod(timed_ref_build)

    orig_cd_step = CommonDownstream.step
    def timed_cd_step(self, request):
        t0 = time.perf_counter()
        result = orig_cd_step(self, request)
        common_downstream_step_s.append(time.perf_counter() - t0)
        planner_time = result.diagnostics.get("planner", {}).get("timing_s", {})
        gen_and_eval = planner_time.get("generation", 0.0) + planner_time.get("evaluation", 0.0) + planner_time.get("projection", 0.0)
        if gen_and_eval:
            planner_plan_s.append(gen_and_eval)
        controller_diag = result.diagnostics.get("controller")
        return result
    CommonDownstream.step = timed_cd_step

    orig_mpc_solve = LtvMpcController.solve
    def timed_mpc_solve(self, state, reference):
        t0 = time.perf_counter()
        result = orig_mpc_solve(self, state, reference)
        mpc_solve_s.append(time.perf_counter() - t0)
        return result
    LtvMpcController.solve = timed_mpc_solve

    try:
        for spec in sample:
            env.reset(spec)
            for _ in range(N_STEPS):
                t0 = time.perf_counter()
                _, _, terminated, truncated, _ = env.step(BehaviorAction.MERGE)
                env_step_total_s.append(time.perf_counter() - t0)
                if terminated or truncated:
                    break
    finally:
        ReferenceLine.from_lane_polyline = orig_ref_build
        CommonDownstream.step = orig_cd_step
        LtvMpcController.solve = orig_mpc_solve

    def report(name, values):
        if not values:
            print(f"  {name}: no samples collected")
            return
        print(
            f"  {name}: n={len(values)} p50={percentile(values, 50)*1000:.3f}ms "
            f"p95={percentile(values, 95)*1000:.3f}ms max={max(values)*1000:.3f}ms"
        )

    print(f"\nSample: {len(sample)} maneuvers, up to {N_STEPS} steps each\n")
    report("ReferenceLine construction", reference_construction_s)
    report("frenet_planner (project+generate+evaluate)", planner_plan_s)
    report("LtvMpcController.solve()", mpc_solve_s)
    report("CommonDownstream.step() total", common_downstream_step_s)
    report("MergeEnvironment.step() total (frenet_mpc)", env_step_total_s)

    if env_step_total_s:
        steps_per_second = 1.0 / np.mean(env_step_total_s)
        print(f"\nOverall steps/second (frenet_mpc mode): {steps_per_second:.2f}")

    # Identify bottleneck by mean share of MergeEnvironment.step() time.
    mean_env_step = np.mean(env_step_total_s) if env_step_total_s else float("nan")
    mean_mpc = np.mean(mpc_solve_s) if mpc_solve_s else 0.0
    mean_planner = np.mean(planner_plan_s) if planner_plan_s else 0.0
    print(f"\nBottleneck identification (measurement only, no optimization performed):")
    print(f"  mean MergeEnvironment.step() total: {mean_env_step*1000:.3f}ms")
    print(f"  mean LtvMpcController.solve() share: {mean_mpc*1000:.3f}ms "
          f"({100*mean_mpc/mean_env_step:.1f}% of step total)" if mean_env_step else "")
    print(f"  mean frenet_planner share: {mean_planner*1000:.3f}ms "
          f"({100*mean_planner/mean_env_step:.1f}% of step total)" if mean_env_step else "")


def run_legacy_vs_frenet_mpc_comparison():
    print("\n" + "=" * 70)
    print("Stage 3-G.6: legacy vs frenet_mpc diagnostic comparison (NOT a research conclusion)")
    print("=" * 70)

    specs = load_maneuver_specs("train")
    stride = max(len(specs) // COMPARISON_SAMPLE_SIZE, 1)
    sample = specs[::stride][:COMPARISON_SAMPLE_SIZE]

    env_legacy = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH, downstream_mode="legacy")
    env_frenet = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH, downstream_mode="frenet_mpc")

    print(f"\n{'maneuver_id':<12} {'legacy_term':<20} {'frenet_mpc_term':<20} {'legacy_steps':<13} {'frenet_steps':<13}")
    rows = []
    for spec in sample:
        def rollout(env):
            env.reset(spec)
            reason = None
            steps = 0
            for _ in range(100):
                _, _, terminated, truncated, info = env.step(BehaviorAction.MERGE)
                steps += 1
                if terminated or truncated:
                    reason = info["termination_reason"]
                    break
            else:
                reason = "truncation_horizon_script_cap"
            return reason, steps

        try:
            legacy_reason, legacy_steps = rollout(env_legacy)
        except Exception as exc:  # noqa: BLE001
            legacy_reason, legacy_steps = f"EXCEPTION:{type(exc).__name__}", -1
        try:
            frenet_reason, frenet_steps = rollout(env_frenet)
        except Exception as exc:  # noqa: BLE001
            frenet_reason, frenet_steps = f"EXCEPTION:{type(exc).__name__}", -1

        print(f"{spec.maneuver_id:<12} {legacy_reason:<20} {frenet_reason:<20} {legacy_steps:<13} {frenet_steps:<13}")
        rows.append((spec.maneuver_id, legacy_reason, frenet_reason, legacy_steps, frenet_steps))

    agree = sum(1 for r in rows if r[1] == r[2])
    print(f"\nTermination-reason agreement: {agree}/{len(rows)} (diagnostic only, not a pass/fail gate)")


if __name__ == "__main__":
    run_performance_snapshot()
    run_legacy_vs_frenet_mpc_comparison()
