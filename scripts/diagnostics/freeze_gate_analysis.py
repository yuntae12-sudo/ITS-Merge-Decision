#!/usr/bin/env python3
"""Downstream Freeze Gate analysis (read-only, diagnostic-only).

Reuses already-collected Fixed TRAIN64/step36 real-env diagnostic CSVs
(no new environment rollout) to compute the exact Gate A-E figures for
the PPO Full Retraining Freeze Gate:

  Gate A: full downstream_status breakdown (OK/PLANNER_INFEASIBLE/
          COLLISION_BLOCKED/CONTROLLER_FAILURE/INVALID_REFERENCE) and
          longitudinal_accel/curvature/forward_progress/jerk failure
          counts, post Planner-side feasibility fix.
  Gate B: curvature failure severity distribution + phase/action/
          outcome breakdown.
  Gate C: forward_progress failure breakdown.
  Gate D: overall PLANNER_INFEASIBLE contiguous-streak statistics,
          split from the check-specific (accel/curvature/
          forward_progress) streak statistics -- these are NOT the
          same quantity and must not be conflated.
  Gate E: intervention/fallback rate (any non-OK downstream_status).

Inputs (already generated, reused verbatim):
  outputs/diagnostics/accel_rootcause_step36_plannerfix/step_diagnostics.csv
    -- every physical step (2286 rows), downstream_status for all.
  outputs/diagnostics/planner_infeasible_step36_plannerfix/episode_summary.csv
    -- only PLANNER_INFEASIBLE rows (229), with checks_failed + severity values.
"""

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict

STEP_DIAGNOSTICS_CSV = "outputs/diagnostics/accel_rootcause_step36_plannerfix/step_diagnostics.csv"
EPISODE_SUMMARY_CSV = "outputs/diagnostics/planner_infeasible_step36_plannerfix/episode_summary.csv"
OUTPUT_DIR = "outputs/diagnostics/freeze_gate_step36"

CURVATURE_THRESHOLD = 0.3
FORWARD_PROGRESS_THRESHOLD = -0.1


def load_step_rows():
    with open(STEP_DIAGNOSTICS_CSV) as f:
        return list(csv.DictReader(f))


def load_infeasible_rows():
    with open(EPISODE_SUMMARY_CSV) as f:
        return list(csv.DictReader(f))


def gate_a_status_breakdown(step_rows):
    counts = Counter(r["downstream_status"] for r in step_rows)
    total = len(step_rows)
    return {
        "total_steps": total,
        "OK": counts.get("OK", 0),
        "PLANNER_INFEASIBLE": counts.get("PLANNER_INFEASIBLE", 0),
        "COLLISION_BLOCKED": counts.get("COLLISION_BLOCKED", 0),
        "CONTROLLER_FAILURE": counts.get("CONTROLLER_FAILURE", 0),
        "INVALID_REFERENCE": counts.get("INVALID_REFERENCE", 0),
    }


def gate_bc_check_breakdown(infeasible_rows, check_name, threshold, value_field):
    matching = [r for r in infeasible_rows if check_name in r["checks_failed"].split(";")]
    by_phase = Counter(r["phase"] for r in matching)
    by_action = Counter(r["action"] for r in matching)
    by_outcome = Counter(r["outcome"] for r in matching)
    values = [float(r[value_field]) for r in matching]

    return {
        "count": len(matching),
        "by_phase": dict(by_phase),
        "by_action": dict(by_action),
        "by_outcome": dict(by_outcome),
        "values": values,
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": (sorted(values)[int(0.9 * (len(values) - 1))] if values else None),
        "p95": (sorted(values)[int(0.95 * (len(values) - 1))] if values else None),
        "max": max(values) if values else None,
        "min": min(values) if values else None,
        "threshold": threshold,
    }


def curvature_severity_buckets(curvature_rows):
    buckets = [(0.300, 0.320), (0.320, 0.400), (0.400, 0.600), (0.600, float("inf"))]
    result = []
    for lo, hi in buckets:
        in_bucket = [r for r in curvature_rows if lo < float(r["max_abs_curvature_per_m"]) <= hi]
        result.append({
            "range": f"{lo:.3f}-{hi if hi != float('inf') else 'inf'}",
            "count": len(in_bucket),
            "phase": dict(Counter(r["phase"] for r in in_bucket)),
            "outcome": dict(Counter(r["outcome"] for r in in_bucket)),
            "action": dict(Counter(r["action"] for r in in_bucket)),
        })
    return result


def _compute_streaks(episode_steps, predicate):
    """episode_steps: list of dict sorted by step_index for ONE episode.
    predicate(row) -> bool. Returns list of streak lengths (contiguous
    runs where predicate is True)."""

    streaks = []
    current = 0
    for row in episode_steps:
        if predicate(row):
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return streaks


def _transition_probabilities(episode_steps_by_id, predicate):
    from_true = {"next_true": 0, "next_false": 0}
    from_false = {"next_true": 0, "next_false": 0}
    for steps in episode_steps_by_id.values():
        for i in range(len(steps) - 1):
            cur = predicate(steps[i])
            nxt = predicate(steps[i + 1])
            bucket = from_true if cur else from_false
            bucket["next_true" if nxt else "next_false"] += 1

    def _p(bucket):
        total = bucket["next_true"] + bucket["next_false"]
        return (bucket["next_true"] / total) if total else None

    return _p(from_true), _p(from_false)


def gate_d_streak_analysis(step_rows, predicate, label):
    by_episode = defaultdict(list)
    for r in step_rows:
        by_episode[r["maneuver_id"]].append(r)
    for mid in by_episode:
        by_episode[mid].sort(key=lambda r: int(r["step_index"]))

    all_streaks = []
    episodes_affected = 0
    outcome_streaks = defaultdict(list)
    phase_true_counts = Counter()

    for mid, steps in by_episode.items():
        streaks = _compute_streaks(steps, predicate)
        if streaks:
            episodes_affected += 1
            all_streaks.extend(streaks)
            outcome = steps[0]["outcome"]
            outcome_streaks[outcome].extend(streaks)
        for r in steps:
            if predicate(r):
                phase_true_counts[r["phase"]] += 1

    p_next_given_true, p_next_given_false = _transition_probabilities(by_episode, predicate)

    total_true_steps = sum(1 for r in step_rows if predicate(r))

    def _stats(streaks):
        if not streaks:
            return {"median": None, "mean": None, "p90": None, "max": None, "count": 0}
        s = sorted(streaks)
        return {
            "median": statistics.median(s),
            "mean": statistics.mean(s),
            "p90": s[int(0.9 * (len(s) - 1))],
            "max": max(s),
            "count": len(s),
        }

    return {
        "label": label,
        "total_true_steps": total_true_steps,
        "episodes_affected": episodes_affected,
        "total_episodes": len(by_episode),
        "num_streaks": len(all_streaks),
        "streak_stats_all": _stats(all_streaks),
        "streak_stats_by_outcome": {o: _stats(s) for o, s in outcome_streaks.items()},
        "p_next_true_given_current_true": p_next_given_true,
        "p_next_true_given_current_false": p_next_given_false,
        "phase_true_counts": dict(phase_true_counts),
    }


def gate_e_intervention(step_rows):
    by_episode = defaultdict(list)
    for r in step_rows:
        by_episode[r["maneuver_id"]].append(r)

    def is_intervention(r):
        return r["downstream_status"] != "OK"

    overall_total = len(step_rows)
    overall_intervention = sum(1 for r in step_rows if is_intervention(r))

    by_phase = defaultdict(lambda: [0, 0])  # [intervention, total]
    by_outcome = defaultdict(lambda: [0, 0])
    for r in step_rows:
        by_phase[r["phase"]][1] += 1
        by_outcome[r["outcome"]][1] += 1
        if is_intervention(r):
            by_phase[r["phase"]][0] += 1
            by_outcome[r["outcome"]][0] += 1

    return {
        "overall_intervention_rate": overall_intervention / overall_total if overall_total else None,
        "overall_intervention_steps": overall_intervention,
        "overall_total_steps": overall_total,
        "by_phase": {k: {"rate": v[0] / v[1] if v[1] else None, "count": v[0], "total": v[1]} for k, v in by_phase.items()},
        "by_outcome": {k: {"rate": v[0] / v[1] if v[1] else None, "count": v[0], "total": v[1]} for k, v in by_outcome.items()},
    }


def main():
    step_rows = load_step_rows()
    infeasible_rows = load_infeasible_rows()

    gate_a = gate_a_status_breakdown(step_rows)
    check_counts = Counter()
    for r in infeasible_rows:
        for c in r["checks_failed"].split(";"):
            if c:
                check_counts[c] += 1
    gate_a["check_counts"] = dict(check_counts)

    gate_b = gate_bc_check_breakdown(infeasible_rows, "curvature", CURVATURE_THRESHOLD, "max_abs_curvature_per_m")
    curvature_rows = [r for r in infeasible_rows if "curvature" in r["checks_failed"].split(";")]
    gate_b["severity_buckets"] = curvature_severity_buckets(curvature_rows)

    gate_c = gate_bc_check_breakdown(infeasible_rows, "forward_progress", FORWARD_PROGRESS_THRESHOLD, "min_forward_progress_s_dot_mps")

    gate_d_overall = gate_d_streak_analysis(
        step_rows, lambda r: r["downstream_status"] == "PLANNER_INFEASIBLE", "Overall PLANNER_INFEASIBLE",
    )

    infeasible_by_id = defaultdict(dict)
    for r in infeasible_rows:
        infeasible_by_id[r["maneuver_id"]][int(r["step_index"])] = r["checks_failed"].split(";")

    def _check_predicate(check_name):
        def predicate(r):
            mid = r["maneuver_id"]
            step_index = int(r["step_index"])
            checks = infeasible_by_id.get(mid, {}).get(step_index)
            return checks is not None and check_name in checks
        return predicate

    gate_d_accel = gate_d_streak_analysis(step_rows, _check_predicate("longitudinal_accel"), "Longitudinal-Accel Infeasible")
    gate_d_curvature = gate_d_streak_analysis(step_rows, _check_predicate("curvature"), "Curvature Infeasible")
    gate_d_forward_progress = gate_d_streak_analysis(step_rows, _check_predicate("forward_progress"), "Forward-Progress Infeasible")

    gate_e = gate_e_intervention(step_rows)

    result = {
        "gate_a_status_breakdown": gate_a,
        "gate_b_curvature": gate_b,
        "gate_c_forward_progress": gate_c,
        "gate_d_overall_streak": gate_d_overall,
        "gate_d_accel_streak": gate_d_accel,
        "gate_d_curvature_streak": gate_d_curvature,
        "gate_d_forward_progress_streak": gate_d_forward_progress,
        "gate_e_intervention": gate_e,
    }

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print("=" * 60)
    print("Gate A: Downstream status breakdown")
    print("=" * 60)
    for k, v in gate_a.items():
        if k != "check_counts":
            print(f"  {k}: {v}")
    print(f"  check_counts: {gate_a['check_counts']}")

    print()
    print("=" * 60)
    print("Gate B: Curvature failure")
    print("=" * 60)
    print(f"  count={gate_b['count']} mean={gate_b['mean']:.4f} median={gate_b['median']:.4f} "
          f"p90={gate_b['p90']:.4f} p95={gate_b['p95']:.4f} max={gate_b['max']:.4f}")
    print(f"  by_phase: {gate_b['by_phase']}")
    print(f"  by_action: {gate_b['by_action']}")
    print(f"  by_outcome: {gate_b['by_outcome']}")
    print("  severity buckets:")
    for b in gate_b["severity_buckets"]:
        print(f"    {b['range']}: count={b['count']} outcome={b['outcome']} phase={b['phase']}")

    print()
    print("=" * 60)
    print("Gate C: Forward-progress failure")
    print("=" * 60)
    print(f"  count={gate_c['count']} mean={gate_c['mean']} median={gate_c['median']} "
          f"max={gate_c['max']} min={gate_c['min']}")
    print(f"  by_phase: {gate_c['by_phase']}")
    print(f"  by_action: {gate_c['by_action']}")
    print(f"  by_outcome: {gate_c['by_outcome']}")

    print()
    print("=" * 60)
    print("Gate D: Streak analysis (Overall vs check-specific)")
    print("=" * 60)
    for gate_d in (gate_d_overall, gate_d_accel, gate_d_curvature, gate_d_forward_progress):
        print(f"--- {gate_d['label']} ---")
        print(f"  total_true_steps={gate_d['total_true_steps']} episodes_affected={gate_d['episodes_affected']}/{gate_d['total_episodes']}")
        print(f"  streak stats (all): {gate_d['streak_stats_all']}")
        print(f"  P(next|current=True)={gate_d['p_next_true_given_current_true']} P(next|current=False)={gate_d['p_next_true_given_current_false']}")
        print(f"  by outcome: {gate_d['streak_stats_by_outcome']}")

    print()
    print("=" * 60)
    print("Gate E: Intervention/fallback")
    print("=" * 60)
    print(f"  overall_intervention_rate={gate_e['overall_intervention_rate']:.4f} "
          f"({gate_e['overall_intervention_steps']}/{gate_e['overall_total_steps']})")
    print(f"  by_phase: {gate_e['by_phase']}")
    print(f"  by_outcome: {gate_e['by_outcome']}")

    print(f"\nWrote {OUTPUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
