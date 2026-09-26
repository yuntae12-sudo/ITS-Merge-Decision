#!/usr/bin/env python3
"""Part C: recomputes KEEP/FOLLOW/MERGE discriminability for TRAIN MERGE v2,
using (1) the SAFETY geometry audit's finding that a sample of 20 sustained
negative-target-gap candidates showed 0/20 real physical OBB overlaps
(Waymax's own SAT check) and 17/20 AMBIGUOUS/3/20 PROJECTION_ARTIFACT_LIKELY,
and (2) the new source-front evidence augmentation
(outputs/merge_v2_decision_audit_v2/training_decision_evidence_augmented.jsonl)
giving FOLLOW a real, independent signal for the first time.

Ground rule (unchanged from outputs/merge_v2_decision_audit/audit_report.md):
this is an action-AFFORDANCE audit, not a ground-truth action label. A
candidate being in `follow_vs_merge_candidates.csv` does NOT mean "the
correct action is FOLLOW" -- it means both actions plausibly have
state-dependent value here.

Per this session's brief:
- Primary action set for `num_meaningful_kfm_actions` is KEEP/FOLLOW/MERGE
  only. STOP is demoted to a secondary diagnostic
  (`stop_affordance_raw`/`stop_affordance_geometry_checked`) because the
  underlying gap<=0 sentinel signal's physical meaning remains only
  partially resolved (0/20 confirmed physical conflict, but 17/20 stayed
  AMBIGUOUS rather than confirmed-safe) -- see safety_review/index.html.
- FOLLOW meaningful requires REAL source_front evidence (source_front_
  present OR a non-trivial source_front_presence_ratio_history), not the
  old target-front proxy used in outputs/merge_v2_decision_audit's script.
"""

import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TRAIN_EVIDENCE = "data/manifests/v2/evidence_training.jsonl"
AUGMENTED_EVIDENCE = "outputs/merge_v2_decision_audit_v2/training_decision_evidence_augmented.jsonl"
OUT_DIR = Path("outputs/merge_v2_decision_audit_v2")
DT_S = 0.1

# Same exploratory tiers as outputs/merge_v2_decision_audit's script
# (derived from TRAIN's own quantiles, not frozen thresholds) -- reused,
# not redefined, for the target-lane (MERGE-side) affordance so MERGE
# semantics do not silently drift between the two audits.
TTC_SAFETY_CRITICAL_S = 3.0
TTC_PRESSURE_S = 10.0
GAP_TIGHT_M = 6.0
GAP_COMFORTABLE_M = 20.0
APPROACH_RUNWAY_S = 2.0

# Source-front thresholds: no prior frozen value exists for this schema gap
# (audit_report.md's whole point), so this session's brief explicitly asks
# for exploratory sensitivity across a few settings (see
# heuristic_sensitivity.csv) rather than a single frozen cutoff. This is
# the "current" setting used for the primary tables.
SOURCE_FRONT_TTC_PRESSURE_S = 10.0
SOURCE_FRONT_GAP_TIGHT_M = 10.0
SOURCE_FRONT_HISTORY_PRESENCE_MIN = 0.3  # >=30% of decision-window frames


def finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def load_jsonl_by_id(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["candidate_id"]] = rec
    return out


def _decode_inf(v):
    if v == "Infinity":
        return float("inf")
    if v == "-Infinity":
        return float("-inf")
    return v


def per_candidate_target_summary(rec):
    ie = rec["interaction_evidence"]
    commit_frame = ie["commit_frame"]
    samples = rec["gap_timeseries"]

    def at_commit(field):
        for s in samples:
            if s["frame"] == commit_frame:
                return s[field]
        return samples[-1][field] if samples else None

    def min_finite(field):
        vals = [s[field] for s in samples if finite(s[field])]
        return min(vals) if vals else None

    def min_signed(field):
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


def max_consecutive_deep_negative(samples, field, floor_m=-0.5):
    max_run, run = 0, 0
    for s in samples:
        v = s[field]
        if v is not None and v < floor_m:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def source_front_tier(aug_rec, gap_thresh, ttc_thresh, history_thresh):
    """FOLLOW affordance tier from the NEW real source_front evidence.

    'has_lead' is true if either the AT-decision-frame snapshot shows a
    source-front vehicle, OR the presence_ratio_history exceeds
    history_thresh (a source-front vehicle was around for a meaningful
    fraction of the decision window even if it isn't there at the exact
    decision frame) -- both are online-safe (t <= decision_frame only,
    enforced in augment_merge_v2_source_front_evidence.py)."""

    present_now = bool(aug_rec["source_front_present"])
    history_ratio = aug_rec["source_front_presence_ratio_history"] or 0.0
    has_lead = present_now or history_ratio >= history_thresh

    if not has_lead:
        return "LOW"

    gap = aug_rec["source_front_gap_m"] if present_now else aug_rec["source_front_min_gap_history_m"]
    ttc = _decode_inf(aug_rec["source_front_ttc_s"]) if present_now else _decode_inf(aug_rec["source_front_min_ttc_history_s"])

    ttc_pressure = finite(ttc) and ttc < ttc_thresh
    gap_tight = gap is not None and gap < gap_thresh
    closing = (aug_rec["source_front_closing_ratio_history"] or 0.0) >= 0.5

    if gap_tight or (ttc_pressure and closing):
        return "HIGH"
    if ttc_pressure or gap_tight or closing:
        return "MEDIUM"
    return "MEDIUM" if has_lead else "LOW"  # a real, present lead is itself non-trivial vs no lead at all


def compute_kfm_affordances(rec, aug_rec, target_summary, gap_thresh, ttc_thresh, history_thresh):
    ie = rec["interaction_evidence"]
    has_front = ie["front_vehicle_id"] is not None
    has_rear = ie["rear_vehicle_id"] is not None

    front_tier = affordance_tier(has_front, target_summary["front_gap_min"], target_summary["front_ttc_min_finite"])
    rear_tier = affordance_tier(has_rear, target_summary["rear_gap_min"], target_summary["rear_ttc_min_finite"])

    tier_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    if not has_front and not has_rear:
        merge_affordance = "LOW"
    elif front_tier == "LOW" and rear_tier == "LOW":
        merge_affordance = "MEDIUM"
    else:
        merge_affordance = max(front_tier, rear_tier, key=lambda t: tier_rank[t])

    follow_affordance = source_front_tier(aug_rec, gap_thresh, ttc_thresh, history_thresh)

    if target_summary["decision_runway_s"] >= APPROACH_RUNWAY_S and front_tier != "HIGH" and rear_tier != "HIGH":
        keep_affordance = "HIGH" if target_summary["decision_runway_s"] >= 2 * APPROACH_RUNWAY_S else "MEDIUM"
    else:
        keep_affordance = "LOW"

    # STOP: secondary diagnostic only (brief Section 18/D). Raw = old
    # heuristic (sustained deep-negative-gap run or genuine closing TTC<1.5s
    # excluding the 0.0 sentinel). Geometry-checked = same trigger, but
    # only counted when a targeted geometry re-check exists AND confirmed
    # PHYSICAL_CONFLICT_LIKELY (none did in the 20-sample audit -- see
    # safety_review/index.html) -- so geometry-checked is conservatively
    # 0 for all TRAIN candidates outside that 20-sample audit, which is
    # itself the headline finding, not a bug.
    sustained_overlap_frames = max(
        max_consecutive_deep_negative(rec["gap_timeseries"], "front_gap_m"),
        max_consecutive_deep_negative(rec["gap_timeseries"], "rear_gap_m"),
    )
    genuine_closing_ttc = (
        (finite(target_summary["front_ttc_min_finite"]) and 0 < target_summary["front_ttc_min_finite"] < 1.5)
        or (finite(target_summary["rear_ttc_min_finite"]) and 0 < target_summary["rear_ttc_min_finite"] < 1.5)
    )
    stop_affordance_raw = "HIGH" if (sustained_overlap_frames >= 5 or genuine_closing_ttc) else "LOW"

    return {
        "keep_affordance": keep_affordance,
        "follow_affordance": follow_affordance,
        "merge_affordance": merge_affordance,
        "stop_affordance_raw": stop_affordance_raw,
        "stop_affordance_geometry_checked": "LOW",  # see docstring: none of 20 geometry-audited samples confirmed
        "front_tier": front_tier,
        "rear_tier": rear_tier,
        "sustained_overlap_frames": sustained_overlap_frames,
    }


def num_meaningful_kfm_actions(aff):
    tier_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    keys = ("keep_affordance", "follow_affordance", "merge_affordance")
    return sum(1 for k in keys if tier_rank[aff[k]] >= 1)


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


def decision_relevance_kfm(aff, n_meaningful, interaction_str):
    if n_meaningful >= 3 or (n_meaningful == 2 and interaction_str == "STRONG"):
        return "HIGH"
    if n_meaningful == 2 or (n_meaningful == 1 and interaction_str != "WEAK"):
        return "MEDIUM"
    return "LOW"


def archetype_v2(rec, target_summary, aff, interaction_str):
    ie = rec["interaction_evidence"]
    has_front = ie["front_vehicle_id"] is not None
    has_rear = ie["rear_vehicle_id"] is not None
    front_tier = aff["front_tier"]
    rear_tier = aff["rear_tier"]

    # SAFETY_CRITICAL split: UNCERTAIN unless a targeted geometry re-check
    # confirmed PHYSICAL_CONFLICT_LIKELY (none did among the 20 audited --
    # see safety_review/index.html); this keeps the label honest about
    # what was actually geometry-verified vs raw gap-sign-triggered.
    if aff["stop_affordance_raw"] == "HIGH":
        return "SAFETY_CRITICAL_UNCERTAIN" if aff["stop_affordance_geometry_checked"] != "HIGH" else "SAFETY_CRITICAL_CONFIRMED"

    front_relevant = front_tier in ("MEDIUM", "HIGH")
    rear_relevant = rear_tier in ("MEDIUM", "HIGH")
    follow_relevant = aff["follow_affordance"] in ("MEDIUM", "HIGH")

    if front_relevant and rear_relevant:
        return "BOTH_SIDES_CONSTRAINED"
    if follow_relevant and (front_relevant or rear_relevant):
        return "BOTH_SIDES_CONSTRAINED"
    if rear_relevant and not front_relevant:
        return "REAR_PRESSURE"
    if front_relevant and not rear_relevant:
        return "FRONT_CONSTRAINED"
    if not has_front and not has_rear and not follow_relevant:
        return "PATH_FOLLOWING_LIKE" if target_summary["decision_runway_s"] < APPROACH_RUNWAY_S else "APPROACH_KEEP_CONTEXT"
    if interaction_str == "WEAK" and not follow_relevant:
        return "WEAK_INTERACTION"
    if target_summary["decision_runway_s"] >= APPROACH_RUNWAY_S:
        return "APPROACH_KEEP_CONTEXT"
    return "CLEAR_MERGE_OPPORTUNITY"


def build_rows(train_records, aug_by_id, gap_thresh, ttc_thresh, history_thresh):
    rows = []
    for cid, rec in train_records.items():
        aug_rec = aug_by_id[cid]
        target_summary = per_candidate_target_summary(rec)
        aff = compute_kfm_affordances(rec, aug_rec, target_summary, gap_thresh, ttc_thresh, history_thresh)
        ie = rec["interaction_evidence"]
        has_front = ie["front_vehicle_id"] is not None
        has_rear = ie["rear_vehicle_id"] is not None
        istrength = interaction_strength(has_front, has_rear, aff["front_tier"], aff["rear_tier"])
        n_meaningful = num_meaningful_kfm_actions(aff)
        relevance = decision_relevance_kfm(aff, n_meaningful, istrength)
        arch = archetype_v2(rec, target_summary, aff, istrength)

        rows.append({
            "candidate_id": cid,
            "scenario_id": rec["scenario_id"],
            "current_decision": rec["automatic_decision"],
            "automatic_reason": rec["automatic_reason"],
            "decision_relevance": relevance,
            "interaction_strength": istrength,
            "num_meaningful_kfm_actions": n_meaningful,
            "archetype": arch,
            "keep_affordance": aff["keep_affordance"],
            "follow_affordance": aff["follow_affordance"],
            "merge_affordance": aff["merge_affordance"],
            "stop_affordance_raw": aff["stop_affordance_raw"],
            "stop_affordance_geometry_checked": aff["stop_affordance_geometry_checked"],
            "source_front_present": aug_rec["source_front_present"],
            "source_front_gap_m": aug_rec["source_front_gap_m"],
            "source_front_relative_speed_mps": aug_rec["source_front_relative_speed_mps"],
            "source_front_ttc_s": _decode_inf(aug_rec["source_front_ttc_s"]),
            "source_front_presence_ratio_history": aug_rec["source_front_presence_ratio_history"],
            "front_gap_at_commit": target_summary["front_gap_at_commit"],
            "rear_gap_at_commit": target_summary["rear_gap_at_commit"],
            "front_ttc_min_finite": target_summary["front_ttc_min_finite"],
            "rear_ttc_min_finite": target_summary["rear_ttc_min_finite"],
            "decision_runway_s": target_summary["decision_runway_s"],
        })
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_records = load_jsonl_by_id(TRAIN_EVIDENCE)
    aug_by_id = load_jsonl_by_id(AUGMENTED_EVIDENCE)
    n = len(train_records)
    assert n == 5545, f"expected 5545, got {n}"
    assert set(train_records) == set(aug_by_id), "candidate_id mismatch between TRAIN evidence and augmented evidence"
    print(f"loaded {n} TRAIN records, {len(aug_by_id)} augmented source-front records")

    rows = build_rows(
        train_records, aug_by_id,
        SOURCE_FRONT_GAP_TIGHT_M, SOURCE_FRONT_TTC_PRESSURE_S, SOURCE_FRONT_HISTORY_PRESENCE_MIN,
    )

    ids = [r["candidate_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate candidate_id"

    with open(OUT_DIR / "training_kfm_discriminability.csv", "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote training_kfm_discriminability.csv ({len(rows)} rows)")

    # A. K/F/M meaningful action count
    meaningful_counts = Counter(r["num_meaningful_kfm_actions"] for r in rows)
    with open(OUT_DIR / "training_kfm_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_meaningful_kfm_actions", "count", "ratio"])
        for k in range(4):
            c = meaningful_counts.get(k, 0)
            w.writerow([k, c, round(c / n, 4)])
    print("K/F/M meaningful action counts:", dict(sorted(meaningful_counts.items())))

    # B. new archetype distribution
    archetype_counts = Counter(r["archetype"] for r in rows)
    with open(OUT_DIR / "training_archetype_summary_v2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["archetype", "count", "ratio"])
        for arch, c in sorted(archetype_counts.items(), key=lambda kv: -kv[1]):
            w.writerow([arch, c, round(c / n, 4)])
    print("archetypes v2:", dict(archetype_counts))

    # C. source-front distribution
    present_count = sum(1 for r in rows if r["source_front_present"])
    finite_ttc_count = sum(1 for r in rows if finite(r["source_front_ttc_s"]))
    gaps = sorted(r["source_front_gap_m"] for r in rows if r["source_front_gap_m"] is not None)
    rel_speeds = sorted(r["source_front_relative_speed_mps"] for r in rows if r["source_front_relative_speed_mps"] is not None)

    def q(vals, p):
        if not vals:
            return None
        idx = int(p / 100 * (len(vals) - 1))
        return vals[idx]

    with open(OUT_DIR / "source_front_distribution.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["train_candidates", n])
        w.writerow(["source_front_present_at_decision_frame", present_count])
        w.writerow(["source_front_present_ratio", round(present_count / n, 4)])
        w.writerow(["finite_source_front_ttc", finite_ttc_count])
        w.writerow(["finite_source_front_ttc_ratio", round(finite_ttc_count / n, 4)])
        history_present = sum(1 for r in rows if (aug_by_id[r["candidate_id"]]["source_front_presence_ratio_history"] or 0) > 0)
        w.writerow(["any_source_front_in_history_window", history_present])
        w.writerow(["any_source_front_in_history_window_ratio", round(history_present / n, 4)])
        for label, vals in (("gap_m", gaps), ("relative_speed_mps", rel_speeds)):
            for p in (10, 25, 50, 75, 90):
                w.writerow([f"{label}_p{p}", q(vals, p)])
    print(f"source_front_present: {present_count}/{n} ({present_count/n:.2%}); "
          f"any-in-history: {history_present}/{n} ({history_present/n:.2%})")

    # D. decision relevance (recomputed with K/F/M-only + real source-front)
    relevance_counts = Counter(r["decision_relevance"] for r in rows)
    print("K/F/M decision relevance:", dict(relevance_counts))

    # ACCEPT-only
    accept_rows = [r for r in rows if r["current_decision"] == "accept"]
    n_accept = len(accept_rows)
    accept_kfm2plus = sum(1 for r in accept_rows if r["num_meaningful_kfm_actions"] >= 2)
    accept_relevance = Counter(r["decision_relevance"] for r in accept_rows)
    accept_archetype = Counter(r["archetype"] for r in accept_rows)
    with open(OUT_DIR / "training_accept_discriminability_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "level", "count", "ratio_of_accept"])
        w.writerow(["kfm_meaningful_actions_ge2", "", accept_kfm2plus, round(accept_kfm2plus / n_accept, 4) if n_accept else 0])
        for level in ("HIGH", "MEDIUM", "LOW"):
            c = accept_relevance.get(level, 0)
            w.writerow(["decision_relevance", level, c, round(c / n_accept, 4) if n_accept else 0])
        for arch, c in sorted(accept_archetype.items(), key=lambda kv: -kv[1]):
            w.writerow(["archetype", arch, c, round(c / n_accept, 4) if n_accept else 0])
    print(f"ACCEPT ({n_accept}): K/F/M>=2 meaningful = {accept_kfm2plus} ({accept_kfm2plus/n_accept:.2%})")
    print(f"ACCEPT relevance:", dict(accept_relevance))
    print(f"ACCEPT archetype:", dict(accept_archetype))

    # FOLLOW vs MERGE subset
    follow_vs_merge = [
        r for r in rows
        if r["follow_affordance"] in ("MEDIUM", "HIGH") and r["merge_affordance"] in ("MEDIUM", "HIGH")
    ]
    tier_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    follow_vs_merge.sort(
        key=lambda r: (tier_rank[r["follow_affordance"]] + tier_rank[r["merge_affordance"]], r["num_meaningful_kfm_actions"]),
        reverse=True,
    )
    with open(OUT_DIR / "follow_vs_merge_candidates.csv", "w", newline="") as f:
        fieldnames = [
            "candidate_id", "scenario_id", "source_front_gap_m", "source_front_relative_speed_mps",
            "source_front_ttc_s", "front_gap_at_commit", "rear_gap_at_commit", "front_ttc_min_finite",
            "rear_ttc_min_finite", "decision_relevance", "archetype", "reason_for_selection",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in follow_vs_merge:
            row = dict(r)
            row["reason_for_selection"] = (
                f"follow={r['follow_affordance']} merge={r['merge_affordance']} "
                f"(both meaningful, state-dependent tradeoff plausible, NOT a ground-truth action label)"
            )
            w.writerow(row)
    print(f"FOLLOW-vs-MERGE candidates: {len(follow_vs_merge)}")

    # KEEP vs MERGE subset
    keep_vs_merge = [
        r for r in rows
        if r["keep_affordance"] in ("MEDIUM", "HIGH") and r["merge_affordance"] in ("MEDIUM", "HIGH")
        and r["follow_affordance"] == "LOW"
    ]
    keep_vs_merge.sort(key=lambda r: r["num_meaningful_kfm_actions"], reverse=True)
    with open(OUT_DIR / "keep_vs_merge_candidates.csv", "w", newline="") as f:
        fieldnames = [
            "candidate_id", "scenario_id", "decision_runway_s", "front_gap_at_commit", "rear_gap_at_commit",
            "front_ttc_min_finite", "rear_ttc_min_finite", "decision_relevance", "archetype", "reason_for_selection",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in keep_vs_merge:
            row = dict(r)
            row["reason_for_selection"] = (
                f"keep={r['keep_affordance']} merge={r['merge_affordance']} follow=LOW "
                "(runway to wait vs a real merge opportunity, NOT a ground-truth action label)"
            )
            w.writerow(row)
    print(f"KEEP-vs-MERGE candidates: {len(keep_vs_merge)}")

    # heuristic sensitivity across source-front thresholds
    sensitivity_rows = []
    settings = [
        ("tight", 6.0, 6.0, 0.2),
        ("current", SOURCE_FRONT_GAP_TIGHT_M, SOURCE_FRONT_TTC_PRESSURE_S, SOURCE_FRONT_HISTORY_PRESENCE_MIN),
        ("loose", 15.0, 15.0, 0.5),
    ]
    for name, gap_t, ttc_t, hist_t in settings:
        alt_rows = build_rows(train_records, aug_by_id, gap_t, ttc_t, hist_t)
        alt_meaningful = Counter(r["num_meaningful_kfm_actions"] for r in alt_rows)
        alt_follow_meaningful = sum(1 for r in alt_rows if r["follow_affordance"] in ("MEDIUM", "HIGH"))
        alt_ge2 = sum(1 for r in alt_rows if r["num_meaningful_kfm_actions"] >= 2)
        alt_archetype = Counter(r["archetype"] for r in alt_rows)
        sensitivity_rows.append({
            "setting": name, "gap_tight_m": gap_t, "ttc_pressure_s": ttc_t, "history_presence_min": hist_t,
            "follow_meaningful_count": alt_follow_meaningful,
            "kfm_ge2_count": alt_ge2,
            "meaningful_0": alt_meaningful.get(0, 0), "meaningful_1": alt_meaningful.get(1, 0),
            "meaningful_2": alt_meaningful.get(2, 0), "meaningful_3": alt_meaningful.get(3, 0),
            "both_sides_constrained": alt_archetype.get("BOTH_SIDES_CONSTRAINED", 0),
            "front_constrained": alt_archetype.get("FRONT_CONSTRAINED", 0),
            "rear_pressure": alt_archetype.get("REAR_PRESSURE", 0),
        })
    with open(OUT_DIR / "heuristic_sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sensitivity_rows[0].keys()))
        w.writeheader()
        for r in sensitivity_rows:
            w.writerow(r)
    print("heuristic sensitivity:")
    for r in sensitivity_rows:
        print(f"  {r}")

    # MAN_0013 check (must remain reject, not flipped to a positive label)
    man0013 = [r for r in rows if r["candidate_id"].endswith("t57__204_203")]
    if man0013:
        print("MAN_0013:", man0013[0]["decision_relevance"], man0013[0]["archetype"],
              "current_decision=", man0013[0]["current_decision"])
        assert man0013[0]["current_decision"] == "reject"

    print("done")
    return rows


if __name__ == "__main__":
    main()
