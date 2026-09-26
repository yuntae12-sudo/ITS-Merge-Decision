#!/usr/bin/env python3
"""Reward V1 Spec Section 18: offline magnitude-balance dry-run.

Applies the PROPOSED (not yet implemented) Reward V1 component formulas to
TRAIN Tier A+B's logged evidence states, purely to sanity-check relative
magnitude/scale -- NOT a rollout, NOT a training run. No production reward
code is touched; the formulas below exist only in this analysis script and
in REWARD_V1_SPEC.md, both non-authoritative until a future implementation
session.
"""

import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TIER_MANIFEST = "data/manifests/v2/merge_decision_train_candidates_v2.csv"
EVIDENCE_JSONL = "data/manifests/v2/evidence_training.jsonl"
OUT_DIR = Path("outputs/reward_v1_spec")

# --- Proposed formulas (data-derived thresholds; see REWARD_V1_SPEC.md) ---
TTC_CRITICAL_S = 2.5   # near front_ttc finite p50 (2.46s)
TTC_CAUTION_S = 9.0    # near front_ttc finite p75 (8.77s)
GAP_UNSAFE_M = -2.0    # near front_gap p5-p10 band (-3.3 to -1.9m)
GAP_SUFFICIENT_M = 20.0  # near front_gap p75-p90 band (23.2-36.5m)


def r_safety(ttc):
    """Piecewise-linear penalty: 0 at/above TTC_CAUTION_S, -1 at/below
    TTC_CRITICAL_S, linear between. +Inf (no closing) and the gap<=0
    TTC=0.0 sentinel are handled by the CALLER (see main loop) -- this
    function only receives a value it should treat as a genuine finite
    closing TTC."""
    if ttc >= TTC_CAUTION_S:
        return 0.0
    if ttc <= TTC_CRITICAL_S:
        return -1.0
    frac = (TTC_CAUTION_S - ttc) / (TTC_CAUTION_S - TTC_CRITICAL_S)
    return -frac


def r_gap(gap):
    """Piecewise-linear: -1 at/below GAP_UNSAFE_M, 0 at/above
    GAP_SUFFICIENT_M (saturates -- larger gap is not "more reward"),
    linear between."""
    if gap <= GAP_UNSAFE_M:
        return -1.0
    if gap >= GAP_SUFFICIENT_M:
        return 0.0
    frac = (gap - GAP_UNSAFE_M) / (GAP_SUFFICIENT_M - GAP_UNSAFE_M)
    return -1.0 + frac


def r_progress(delta_progress_m, window_s):
    """Small positive dense signal proportional to realized longitudinal
    progress per second of decision window -- discourages infinite
    KEEP/FOLLOW waiting without dominating terminal reward."""
    if window_s <= 0:
        return 0.0
    return min(1.0, delta_progress_m / window_s / 10.0)  # /10 m/s as a soft normalizer


def r_speed(speed_mps, nominal_mps=15.0):
    """Penalizes deviation from nominal cruise speed, saturating."""
    dev = abs(speed_mps - nominal_mps) / nominal_mps
    return -min(1.0, dev)


def r_comfort(abs_accel, abs_jerk):
    accel_term = -min(1.0, abs_accel / 3.0)
    jerk_term = -min(1.0, abs_jerk / 10.0)
    return 0.5 * accel_term + 0.5 * jerk_term


def load_tier_ab():
    with open(TIER_MANIFEST, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["decision_tier"] in ("A", "B")]
    assert len(rows) == 1097
    return {r["candidate_id"] for r in rows}


def main():
    ids = load_tier_ab()
    safety_vals, gap_vals, progress_vals, speed_vals = [], [], [], []

    with open(EVIDENCE_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["candidate_id"] not in ids:
                continue
            ie = rec["interaction_evidence"]
            commit_frame = ie["commit_frame"]
            sample = next((s for s in rec["gap_timeseries"] if s["frame"] == commit_frame), None)
            if sample is None and rec["gap_timeseries"]:
                sample = rec["gap_timeseries"][-1]
            if sample:
                gap = sample["front_gap_m"]
                ttc = sample["front_ttc_s"]
                if gap is not None:
                    gap_vals.append(r_gap(gap))
                # Safety shaping only applies to a genuine finite closing
                # TTC -- +Inf (no closing) contributes 0 (no penalty,
                # matches r_safety's own saturation at TTC_CAUTION_S), and
                # the gap<=0 TTC==0.0 sentinel is EXCLUDED per this
                # session's brief Section 7 (never treat it as automatic
                # max penalty) rather than silently evaluated at ttc=0.
                if ttc is not None and math.isfinite(ttc) and not (gap is not None and gap <= 0.0 and ttc == 0.0):
                    safety_vals.append(r_safety(ttc))
                elif ttc == float("inf"):
                    safety_vals.append(0.0)

            window_s = (ie["commit_frame"] - ie["decision_start_frame"]) * 0.1
            progress_vals.append(r_progress(ie["longitudinal_progress_m"], window_s))

    with open(OUT_DIR / "05_speed_distribution.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["field"] == "ego_speed_mps_all_frames" and r["tier"] == "A+B":
                speed_mean = float(r["mean"])
                speed_p10 = float(r["p10"])
                speed_p50 = float(r["p50"])
                speed_p90 = float(r["p90"])

    for v in (speed_p10, speed_p50, speed_p90, speed_mean):
        speed_vals.append(r_speed(v))

    def summarize(name, values):
        if not values:
            return {"component": name, "n": 0}
        values_sorted = sorted(values)
        n = len(values_sorted)
        return {
            "component": name, "n": n,
            "mean": statistics.fmean(values), "median": statistics.median(values),
            "p10": values_sorted[int(0.1 * (n - 1))], "p90": values_sorted[int(0.9 * (n - 1))],
            "p99": values_sorted[int(0.99 * (n - 1))], "min": min(values), "max": max(values),
        }

    rows = [
        summarize("r_safety", safety_vals),
        summarize("r_gap", gap_vals),
        summarize("r_progress", progress_vals),
        summarize("r_speed", speed_vals),
    ]
    # comfort using distribution summary values directly (already-aggregated quantiles, not per-frame recompute)
    with open(OUT_DIR / "06_acceleration_jerk_distribution.csv", newline="") as f:
        accel_p50 = jerk_p50 = accel_p90 = jerk_p90 = None
        for r in csv.DictReader(f):
            if r["field"] == "ego_abs_acceleration_mps2":
                accel_p50, accel_p90 = float(r["p50"]), float(r["p90"])
            if r["field"] == "ego_abs_jerk_mps3":
                jerk_p50, jerk_p90 = float(r["p50"]), float(r["p90"])
    comfort_vals = [r_comfort(accel_p50, jerk_p50), r_comfort(accel_p90, jerk_p90)]
    rows.append(summarize("r_comfort", comfort_vals))

    with open(OUT_DIR / "11_reward_magnitude_dryrun.csv", "w", newline="") as f:
        fieldnames = ["component", "n", "mean", "median", "p10", "p90", "p99", "min", "max"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("magnitude dry-run (dense, per-step candidate component scale):")
    for r in rows:
        print(f"  {r['component']}: n={r['n']} mean={r.get('mean')} p10={r.get('p10')} p90={r.get('p90')}")

    # dominance check: is any one component's |mean| > 5x another's?
    means = {r["component"]: abs(r.get("mean", 0) or 0) for r in rows}
    max_c, max_v = max(means.items(), key=lambda kv: kv[1])
    min_c, min_v = min(((k, v) for k, v in means.items() if v > 0), key=lambda kv: kv[1], default=(None, 0))
    ratio = max_v / min_v if min_v else float("inf")
    print(f"\ndominance ratio (max/min |mean|): {max_c}={max_v:.4f} / {min_c}={min_v:.4f} = {ratio:.2f}x")
    dominant = ratio > 5.0
    print(f"DOMINANCE_DETECTED={dominant}")
    return rows, dominant


if __name__ == "__main__":
    main()
