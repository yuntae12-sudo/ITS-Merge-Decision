#!/usr/bin/env python3
"""Analysis-only DECISION RELEVANCE audit for TRAIN MERGE v2 evidence.

Distinguishes "topology-valid MERGE" (the existing v2 automatic_decision)
from "does this scene actually pose a PPO behavior-decision problem" --
i.e. would KEEP/FOLLOW/MERGE/STOP plausibly have different values in
different states, or is one action (usually MERGE) trivially always
correct.

Ground rule (never violated by this script): decision_relevance is NOT
a ground-truth action label. HIGH relevance means "state-dependent
action value is plausible here", not "the correct action is MERGE".
TRAIN only -- VALIDATION is never read by this script.

Uses the production PPO action semantics (src/environment/behavior_
action.py) and observation contract (configs/phase2_common_state.yaml)
to define affordance diagnostics, but the diagnostics themselves are
audit heuristics over stored evidence, not a re-implementation of
those modules and not a change to classify_v2_merge's semantics.

Key schema note (see docs/MERGE_DATASET_V2.md discussion + interaction_
evidence.py docstring): gap_timeseries's front/rear gap+TTC are
TARGET-lane evidence, matching the production observation's
target_front_*/target_rear_* fields. There is no stored source-lane
lead-vehicle evidence in v2 (interaction_evidence.py never computes
it -- v2's whole point is topology+target-lane evidence). FOLLOW
affordance is therefore approximated from decision-window timing
(how much runway exists before the merge commits), not a real
source_front_gap; this limitation is reported, not hidden.
"""

import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRAIN_EVIDENCE = "data/manifests/v2/evidence_training.jsonl"
OUT_DIR = Path("outputs/merge_v2_decision_audit")
DT_S = 0.1  # WOMD/Waymax fixed timestep (src/environment/merge_environment.py)

# Exploratory tiers derived from TRAIN's own gap/TTC quantiles
# (outputs/merge_v2_calibration/training_gap_ttc_summary.csv) -- not
# frozen thresholds, purely for this audit's tiering.
TTC_SAFETY_CRITICAL_S = 3.0   # near TRAIN front_ttc_s p50 (2.87s)
TTC_PRESSURE_S = 10.0         # near TRAIN front_ttc_s p75-ish (10.7s)
GAP_TIGHT_M = 6.0             # near TRAIN rear_gap_m p50 (6.6m)
GAP_COMFORTABLE_M = 20.0      # near TRAIN front_gap_m p90 (35m) scaled down
APPROACH_RUNWAY_S = 2.0       # >=2s of decision-window runway before commit


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def finite(v):
    return v is not None and math.isfinite(v)


def per_candidate_summary(rec):
    """Summarizes one candidate's target-lane front/rear gap+TTC over
    its decision window: value AT commit (merge-point proximity) and
    the MIN over the window (worst-case safety pressure), since either
    can matter for behavior-decision relevance."""

    ie = rec["interaction_evidence"]
    commit_frame = ie["commit_frame"]
    samples = rec["gap_timeseries"]

    def at_commit(field):
        for s in samples:
            if s["frame"] == commit_frame:
                return s[field]
        # commit frame not sampled (can happen if commit == target_end_frame
        # edge) -- fall back to the last available sample.
        return samples[-1][field] if samples else None

    def min_finite(field):
        vals = [s[field] for s in samples if finite(s[field])]
        return min(vals) if vals else None

    def min_signed(field):
        # gap can legitimately be negative (already-overlapping/just-
        # entered geometry) -- unlike TTC, do not filter by finiteness
        # alone, just take the true minimum observed.
        vals = [s[field] for s in samples if s[field] is not None]
        return min(vals) if vals else None

    return {
        "front_gap_at_commit": at_commit("front_gap_m"),
        "rear_gap_at_commit": at_commit("rear_gap_m"),
        "front_ttc_at_commit": at_commit("front_ttc_s"),
        "rear_ttc_at_commit": at_commit("rear_ttc_s"),
        "front_gap_min": min_signed("front_gap_m"),
        "rear_gap_min": min_signed("rear_gap_m"),
        "front_ttc_min_finite": min_finite("front_ttc_s"),
        "rear_ttc_min_finite": min_finite("rear_ttc_s"),
        "decision_runway_s": (commit_frame - ie["decision_start_frame"]) * DT_S,
    }


def affordance_tier(has_lead, min_gap, min_ttc):
    """Returns LOW/MEDIUM/HIGH for one side's (front or rear) affordance
    contribution -- HIGH means this vehicle plausibly forces a
    non-trivial speed/gap decision, LOW means it's near-absent or far
    enough to be a non-factor."""

    if not has_lead:
        return "LOW"
    ttc_pressure = finite(min_ttc) and min_ttc < TTC_PRESSURE_S
    gap_tight = min_gap is not None and min_gap < GAP_COMFORTABLE_M
    if (finite(min_ttc) and min_ttc < TTC_SAFETY_CRITICAL_S) or (
        min_gap is not None and min_gap < GAP_TIGHT_M
    ):
        return "HIGH"
    if ttc_pressure or gap_tight:
        return "MEDIUM"
    return "LOW"


def compute_affordances(rec, summary):
    ie = rec["interaction_evidence"]
    has_front = ie["front_vehicle_id"] is not None
    has_rear = ie["rear_vehicle_id"] is not None

    front_tier = affordance_tier(has_front, summary["front_gap_min"], summary["front_ttc_min_finite"])
    rear_tier = affordance_tier(has_rear, summary["rear_gap_min"], summary["rear_ttc_min_finite"])

    tier_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    merge_affordance = max(front_tier, rear_tier, key=lambda t: tier_rank[t]) if (has_front or has_rear) else "LOW"
    # A clear (uncontested or very roomy) merge is itself a real,
    # non-trivial affordance -- MERGE is only a non-choice if literally
    # nothing about gap/TTC matters, i.e. no lead+trail at all.
    if not has_front and not has_rear:
        merge_affordance = "LOW"
    elif front_tier == "LOW" and rear_tier == "LOW":
        merge_affordance = "MEDIUM"  # clear gap: a real, easy MERGE opportunity

    follow_affordance = front_tier  # target-lane front pressure is the only
    # stored proxy for a "regulate speed instead of committing" pressure;
    # source-lane lead evidence does not exist in this schema (see module
    # docstring) -- reported as a limitation, not fabricated.

    if summary["decision_runway_s"] >= APPROACH_RUNWAY_S and front_tier != "HIGH" and rear_tier != "HIGH":
        keep_affordance = "HIGH" if summary["decision_runway_s"] >= 2 * APPROACH_RUNWAY_S else "MEDIUM"
    else:
        keep_affordance = "LOW"

    # IMPORTANT LIMITATION (see module docstring / audit report):
    # front_gap_m/rear_gap_m <= 0.0 forces front_ttc_s/rear_ttc_s == 0.0
    # by _compute_ttc's sentinel rule (scenario_features.py) -- this is
    # NOT "0 seconds to a closing collision", it is "already <=0 m
    # bumper-to-bumper at this frame", which this audit found occurs in
    # ~30% of TRAIN candidates (often BEFORE commit_frame, i.e. while
    # still approaching -- plausibly a target-polyline projection
    # artifact during the pre-merge approach, not a real physical
    # overlap; distinguishing the two would need scene-level review,
    # not static evidence alone). A single transient frame of gap<=0 is
    # therefore NOT treated as safety-critical here -- only a genuinely
    # small (< -0.5m, ruling out near-zero rounding) SUSTAINED
    # (>= min_source_occupancy_frames-scale persistence) overlap is,
    # matching the persistence philosophy classify_v2_merge already
    # applies to gap_order_persistence_frames.
    def max_consecutive_deep_negative(field, floor_m=-0.5):
        max_run, run = 0, 0
        for s in rec["gap_timeseries"]:
            v = s[field]
            if v is not None and v < floor_m:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        return max_run

    sustained_overlap_frames = max(
        max_consecutive_deep_negative("front_gap_m"),
        max_consecutive_deep_negative("rear_gap_m"),
    )
    safety_critical = (
        sustained_overlap_frames >= 5
        or (finite(summary["front_ttc_min_finite"]) and 0 < summary["front_ttc_min_finite"] < 1.5)
        or (finite(summary["rear_ttc_min_finite"]) and 0 < summary["rear_ttc_min_finite"] < 1.5)
    )
    stop_affordance = "HIGH" if safety_critical else "LOW"

    return {
        "keep_affordance": keep_affordance,
        "follow_affordance": follow_affordance,
        "merge_affordance": merge_affordance,
        "stop_affordance": stop_affordance,
        "front_tier": front_tier,
        "rear_tier": rear_tier,
    }


def interaction_strength(has_front, has_rear, front_tier, rear_tier):
    tier_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    if not has_front and not has_rear:
        return "WEAK"
    strongest = max(front_tier, rear_tier, key=lambda t: tier_rank[t])
    if strongest == "HIGH":
        return "STRONG"
    if strongest == "MEDIUM":
        return "MODERATE"
    return "WEAK"


def num_meaningful_actions(affordances):
    tier_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    keys = ("keep_affordance", "follow_affordance", "merge_affordance", "stop_affordance")
    return sum(1 for k in keys if tier_rank[affordances[k]] >= 1)


def decision_relevance(affordances, n_meaningful, interaction_str):
    if n_meaningful >= 3 or (n_meaningful == 2 and interaction_str == "STRONG"):
        return "HIGH"
    if n_meaningful == 2 or (n_meaningful == 1 and interaction_str != "WEAK"):
        return "MEDIUM"
    return "LOW"


def archetype(rec, summary, affordances, interaction_str):
    ie = rec["interaction_evidence"]
    has_front = ie["front_vehicle_id"] is not None
    has_rear = ie["rear_vehicle_id"] is not None
    front_tier = affordances["front_tier"]
    rear_tier = affordances["rear_tier"]

    safety_critical = affordances["stop_affordance"] == "HIGH"
    if safety_critical:
        return "SAFETY_CRITICAL"

    front_relevant = front_tier in ("MEDIUM", "HIGH")
    rear_relevant = rear_tier in ("MEDIUM", "HIGH")

    if front_relevant and rear_relevant:
        return "BOTH_SIDES_CONSTRAINED"
    if rear_relevant and not front_relevant:
        return "REAR_PRESSURE"
    if front_relevant and not rear_relevant:
        return "FRONT_CONSTRAINED"
    if not has_front and not has_rear:
        return "PATH_FOLLOWING_LIKE" if summary["decision_runway_s"] < APPROACH_RUNWAY_S else "APPROACH_KEEP_CONTEXT"
    if interaction_str == "WEAK":
        return "WEAK_INTERACTION"
    if summary["decision_runway_s"] >= APPROACH_RUNWAY_S:
        return "APPROACH_KEEP_CONTEXT"
    return "CLEAR_MERGE_OPPORTUNITY"


def quantiles(values):
    if not values:
        return {}
    values = sorted(values)
    n = len(values)

    def pct(p):
        if n == 1:
            return values[0]
        idx = p / 100 * (n - 1)
        lo, hi = int(math.floor(idx)), int(math.ceil(idx))
        if lo == hi:
            return values[lo]
        frac = idx - lo
        return values[lo] + (values[hi] - values[lo]) * frac

    return {"min": values[0], "max": values[-1], "mean": statistics.fmean(values),
            "p10": pct(10), "p50": pct(50), "p90": pct(90)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records(TRAIN_EVIDENCE)
    n = len(records)
    print(f"loaded {n} TRAIN evidence records")

    ids = [r["candidate_id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate candidate_id"
    assert n == 5545, f"expected 5545, got {n}"

    rows = []
    for rec in records:
        ie = rec["interaction_evidence"]
        summary = per_candidate_summary(rec)
        affordances = compute_affordances(rec, summary)
        has_front = ie["front_vehicle_id"] is not None
        has_rear = ie["rear_vehicle_id"] is not None
        istrength = interaction_strength(has_front, has_rear, affordances["front_tier"], affordances["rear_tier"])
        n_meaningful = num_meaningful_actions(affordances)
        relevance = decision_relevance(affordances, n_meaningful, istrength)
        arch = archetype(rec, summary, affordances, istrength)

        rows.append({
            "candidate_id": rec["candidate_id"],
            "scenario_id": rec["scenario_id"],
            "current_decision": rec["automatic_decision"],
            "automatic_reason": rec["automatic_reason"],
            "decision_relevance": relevance,
            "interaction_strength": istrength,
            "num_meaningful_actions": n_meaningful,
            "archetype": arch,
            "keep_affordance": affordances["keep_affordance"],
            "follow_affordance": affordances["follow_affordance"],
            "merge_affordance": affordances["merge_affordance"],
            "stop_affordance": affordances["stop_affordance"],
            "front_gap_at_commit": summary["front_gap_at_commit"],
            "rear_gap_at_commit": summary["rear_gap_at_commit"],
            "front_ttc_min_finite": summary["front_ttc_min_finite"],
            "rear_ttc_min_finite": summary["rear_ttc_min_finite"],
            "decision_runway_s": summary["decision_runway_s"],
        })

    # sanity
    for row in rows:
        assert row["decision_relevance"] in ("HIGH", "MEDIUM", "LOW")
        assert row["interaction_strength"] in ("STRONG", "MODERATE", "WEAK")
        assert row["archetype"]

    with open(OUT_DIR / "training_decision_relevance.csv", "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote training_decision_relevance.csv ({len(rows)} rows)")

    from collections import Counter

    relevance_counts = Counter(r["decision_relevance"] for r in rows)
    with open(OUT_DIR / "training_decision_relevance_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "count", "ratio"])
        for level in ("HIGH", "MEDIUM", "LOW"):
            c = relevance_counts.get(level, 0)
            w.writerow([level, c, round(c / n, 4)])
    print("decision relevance:", dict(relevance_counts))

    interaction_counts = Counter(r["interaction_strength"] for r in rows)
    meaningful_counts = Counter(r["num_meaningful_actions"] for r in rows)
    archetype_counts = Counter(r["archetype"] for r in rows)

    with open(OUT_DIR / "training_action_affordance_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["interaction_strength", "count", "ratio"])
        for level in ("STRONG", "MODERATE", "WEAK"):
            c = interaction_counts.get(level, 0)
            w.writerow([level, c, round(c / n, 4)])
        w.writerow([])
        w.writerow(["num_meaningful_actions", "count", "ratio"])
        for k in sorted(meaningful_counts):
            c = meaningful_counts[k]
            w.writerow([k, c, round(c / n, 4)])
    print("interaction strength:", dict(interaction_counts))
    print("meaningful actions:", dict(sorted(meaningful_counts.items())))

    with open(OUT_DIR / "training_archetype_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["archetype", "count", "ratio"])
        for arch, c in sorted(archetype_counts.items(), key=lambda kv: -kv[1]):
            w.writerow([arch, c, round(c / n, 4)])
    print("archetypes:", dict(archetype_counts))

    # ACCEPT-only audit
    accept_rows = [r for r in rows if r["current_decision"] == "accept"]
    n_accept = len(accept_rows)
    accept_relevance = Counter(r["decision_relevance"] for r in accept_rows)
    accept_archetype = Counter(r["archetype"] for r in accept_rows)
    accept_interaction = Counter(r["interaction_strength"] for r in accept_rows)
    with open(OUT_DIR / "training_accept_relevance_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "level", "count", "ratio_of_accept"])
        for level in ("HIGH", "MEDIUM", "LOW"):
            c = accept_relevance.get(level, 0)
            w.writerow(["decision_relevance", level, c, round(c / n_accept, 4) if n_accept else 0])
        for level in ("STRONG", "MODERATE", "WEAK"):
            c = accept_interaction.get(level, 0)
            w.writerow(["interaction_strength", level, c, round(c / n_accept, 4) if n_accept else 0])
        for arch, c in sorted(accept_archetype.items(), key=lambda kv: -kv[1]):
            w.writerow(["archetype", arch, c, round(c / n_accept, 4) if n_accept else 0])
    print(f"ACCEPT ({n_accept}) relevance:", dict(accept_relevance))
    print(f"ACCEPT ({n_accept}) archetype:", dict(accept_archetype))

    # trivial merge risk
    trivial_pool = [r for r in rows if r["archetype"] in ("PATH_FOLLOWING_LIKE", "CLEAR_MERGE_OPPORTUNITY")
                    and r["num_meaningful_actions"] <= 1]
    weak_pool = [r for r in rows if r["interaction_strength"] == "WEAK"]
    strong_pool = [r for r in rows if r["interaction_strength"] == "STRONG"]
    print(f"trivial_merge_risk pool: {len(trivial_pool)} ({len(trivial_pool)/n:.2%})")
    print(f"weak interaction: {len(weak_pool)} ({len(weak_pool)/n:.2%})")
    print(f"strong interaction: {len(strong_pool)} ({len(strong_pool)/n:.2%})")

    # top decision-relevant candidates
    def rank_key(r):
        tier_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        relevance_rank = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
        return (
            relevance_rank[r["decision_relevance"]],
            r["num_meaningful_actions"],
            tier_rank[{"STRONG": "HIGH", "MODERATE": "MEDIUM", "WEAK": "LOW"}[r["interaction_strength"]]],
        )

    ranked = sorted(rows, key=rank_key, reverse=True)
    top20 = ranked[:20]
    with open(OUT_DIR / "top_decision_relevant_candidates.csv", "w", newline="") as f:
        fieldnames = [
            "candidate_id", "scenario_id", "current_decision", "decision_relevance",
            "interaction_strength", "archetype", "keep_affordance", "follow_affordance",
            "merge_affordance", "stop_affordance", "front_gap_at_commit", "rear_gap_at_commit",
            "front_ttc_min_finite", "rear_ttc_min_finite", "decision_runway_s", "reason_for_rank",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in top20:
            row = dict(r)
            row["reason_for_rank"] = (
                f"{r['num_meaningful_actions']} meaningful actions, "
                f"{r['interaction_strength'].lower()} interaction, "
                f"front_ttc_min={r['front_ttc_min_finite']}, rear_ttc_min={r['rear_ttc_min_finite']}, "
                f"front_gap@commit={r['front_gap_at_commit']}, rear_gap@commit={r['rear_gap_at_commit']}"
            )
            w.writerow(row)
    print(f"wrote top_decision_relevant_candidates.csv (top {len(top20)})")

    # MAN_0013 check
    man0013 = [r for r in rows if r["candidate_id"].endswith("t57__204_203")]
    if man0013:
        print("MAN_0013 row:", man0013[0]["decision_relevance"], man0013[0]["archetype"],
              "current_decision=", man0013[0]["current_decision"])

    print("done")
    return rows


if __name__ == "__main__":
    main()
