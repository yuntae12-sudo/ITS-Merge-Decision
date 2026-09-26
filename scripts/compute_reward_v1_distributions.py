#!/usr/bin/env python3
"""Reward V1 Specification -- Section 5-13 distribution analysis.

TRAIN Tier A+B (1097 candidates) ONLY. VALIDATION is never read. This
script is analysis-only: it computes empirical distributions to inform
(not decide) Reward V1 threshold/weight proposals in REWARD_V1_SPEC.md. It
does not modify any production reward code, config, or the dataset
manifest itself.

TTC handling (per audit_report.md's established finding, reused not
re-derived): `_compute_ttc`'s sentinel rule sets TTC=0.0 when gap<=0
("already overlapping" sentinel, not "closing fast") and TTC=+Inf when not
closing ("no closing vehicle" sentinel). +Inf is never coerced to a large
finite number for quantiles -- it is counted and reported separately.
"""

import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TIER_MANIFEST = "data/manifests/v2/merge_decision_train_candidates_v2.csv"
EVIDENCE_JSONL = "data/manifests/v2/evidence_training.jsonl"
AUGMENTED_JSONL = "outputs/merge_v2_decision_audit_v2/training_decision_evidence_augmented.jsonl"
OUT_DIR = Path("outputs/reward_v1_spec")


def load_tier_ab_ids():
    with open(TIER_MANIFEST, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["decision_tier"] in ("A", "B")]
    assert len(rows) == 1097, f"expected 1097 Tier A+B, got {len(rows)}"
    assert all(r["split"] == "training" for r in rows), "non-training split leaked into Tier A+B"
    ids = {r["candidate_id"] for r in rows}
    assert len(ids) == 1097, "duplicate candidate_id in Tier A+B"
    tier_by_id = {r["candidate_id"]: r["decision_tier"] for r in rows}
    return ids, tier_by_id


def load_jsonl_by_id(path, ids=None):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if ids is None or rec["candidate_id"] in ids:
                out[rec["candidate_id"]] = rec
    return out


def is_finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def is_pos_inf(v):
    return v == float("inf") or v == "Infinity"


def quantiles(values, percentiles=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    if not values:
        return {f"p{p}": None for p in percentiles}
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

    return {f"p{p}": pct(p) for p in percentiles}


def summarize_finite(values):
    """Full stat block for a list of finite floats (no None/Inf/NaN)."""
    if not values:
        return {"valid_count": 0, "min": None, "max": None, "mean": None, "std": None, **quantiles([])}
    stats = {
        "valid_count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }
    stats.update(quantiles(values))
    return stats


def gap_distribution(evidence_by_id, tier_by_id):
    """Section 8: Target Front/Rear Gap + Source Front Gap, at commit
    frame (the single decision-relevant snapshot already used
    consistently throughout this dataset's audits) -- combined + per-tier."""

    rows = []
    per_field_values = defaultdict(lambda: defaultdict(list))  # field -> tier -> [values]

    for cid, rec in evidence_by_id.items():
        tier = tier_by_id[cid]
        ie = rec["interaction_evidence"]
        commit_frame = ie["commit_frame"]
        sample = next((s for s in rec["gap_timeseries"] if s["frame"] == commit_frame), None)
        if sample is None and rec["gap_timeseries"]:
            sample = rec["gap_timeseries"][-1]
        front_gap = sample["front_gap_m"] if sample else None
        rear_gap = sample["rear_gap_m"] if sample else None

        if front_gap is not None:
            per_field_values["target_front_gap_m"]["A+B"].append(front_gap)
            per_field_values["target_front_gap_m"][tier].append(front_gap)
        if rear_gap is not None:
            per_field_values["target_rear_gap_m"]["A+B"].append(rear_gap)
            per_field_values["target_rear_gap_m"][tier].append(rear_gap)

    for cid in evidence_by_id:
        aug = augmented_by_id.get(cid)
        if aug is None:
            continue
        tier = tier_by_id[cid]
        gap = aug["source_front_gap_m"]
        if gap is not None:
            per_field_values["source_front_gap_m"]["A+B"].append(gap)
            per_field_values["source_front_gap_m"][tier].append(gap)

    for field, by_tier in per_field_values.items():
        for tier, values in by_tier.items():
            total_considered = 1097 if tier == "A+B" else sum(1 for t in tier_by_id.values() if t == tier)
            missing = total_considered - len(values)
            negative = sum(1 for v in values if v < 0)
            zero = sum(1 for v in values if v == 0)
            positive = sum(1 for v in values if v > 0)
            row = {
                "field": field, "tier": tier, "total_considered": total_considered,
                "missing_count": missing, "negative_count": negative, "zero_count": zero,
                "positive_count": positive,
            }
            row.update(summarize_finite(values))
            rows.append(row)
    return rows


def ttc_distribution(evidence_by_id, tier_by_id):
    """Section 9: Target Front/Rear TTC + Source Front TTC. +Inf and the
    gap<=0 TTC=0.0 sentinel are counted separately, never coerced into
    the finite quantile computation."""

    rows = []
    per_field = defaultdict(lambda: defaultdict(list))  # field -> tier -> raw values (may include inf)
    zero_sentinel_counts = defaultdict(lambda: defaultdict(int))  # gap<=0 -> ttc==0.0

    for cid, rec in evidence_by_id.items():
        tier = tier_by_id[cid]
        ie = rec["interaction_evidence"]
        commit_frame = ie["commit_frame"]
        sample = next((s for s in rec["gap_timeseries"] if s["frame"] == commit_frame), None)
        if sample is None and rec["gap_timeseries"]:
            sample = rec["gap_timeseries"][-1]
        if sample is None:
            continue
        for prefix, gap_key, ttc_key in (("target_front", "front_gap_m", "front_ttc_s"), ("target_rear", "rear_gap_m", "rear_ttc_s")):
            gap = sample[gap_key]
            ttc = sample[ttc_key]
            if ttc is None:
                continue
            field = f"{prefix}_ttc_s"
            per_field[field]["A+B"].append(ttc)
            per_field[field][tier].append(ttc)
            if gap is not None and gap <= 0.0 and ttc == 0.0:
                zero_sentinel_counts[field]["A+B"] += 1
                zero_sentinel_counts[field][tier] += 1

    for cid in evidence_by_id:
        aug = augmented_by_id.get(cid)
        if aug is None:
            continue
        tier = tier_by_id[cid]
        ttc = aug["source_front_ttc_s"]
        gap = aug["source_front_gap_m"]
        if ttc is None:
            continue
        ttc_val = float("inf") if ttc == "Infinity" else ttc
        field = "source_front_ttc_s"
        per_field[field]["A+B"].append(ttc_val)
        per_field[field][tier].append(ttc_val)
        if gap is not None and gap <= 0.0 and ttc_val == 0.0:
            zero_sentinel_counts[field]["A+B"] += 1
            zero_sentinel_counts[field][tier] += 1

    for field, by_tier in per_field.items():
        for tier, raw_values in by_tier.items():
            total = len(raw_values)
            pos_inf = sum(1 for v in raw_values if v == float("inf"))
            nan_count = sum(1 for v in raw_values if isinstance(v, float) and v != v)
            neg_inf = sum(1 for v in raw_values if v == float("-inf"))
            finite_values = [v for v in raw_values if math.isfinite(v)]
            zero_sentinel = zero_sentinel_counts[field][tier]
            row = {
                "field": field, "tier": tier, "total": total,
                "finite_count": len(finite_values), "pos_inf_count": pos_inf,
                "nan_count": nan_count, "neg_inf_count": neg_inf,
                "finite_ratio": round(len(finite_values) / total, 4) if total else None,
                "zero_sentinel_gap_le_0_count": zero_sentinel,
            }
            row.update(summarize_finite(finite_values))
            rows.append(row)
    return rows


def progress_distribution(evidence_by_id, tier_by_id):
    """Section 10: longitudinal_progress_m / lateral_displacement_m
    (only stored progress fields in the frozen evidence -- both computed
    from decision_start_frame vs completion_frame, i.e. already-realized
    history, never future-relative-to-current-online-step info)."""

    rows = []
    per_field = defaultdict(lambda: defaultdict(list))
    for cid, rec in evidence_by_id.items():
        tier = tier_by_id[cid]
        ie = rec["interaction_evidence"]
        per_field["longitudinal_progress_m"]["A+B"].append(ie["longitudinal_progress_m"])
        per_field["longitudinal_progress_m"][tier].append(ie["longitudinal_progress_m"])
        per_field["lateral_displacement_m"]["A+B"].append(ie["lateral_displacement_m"])
        per_field["lateral_displacement_m"][tier].append(ie["lateral_displacement_m"])
        decision_window_frames = ie["commit_frame"] - ie["decision_start_frame"]
        per_field["decision_window_frames"]["A+B"].append(decision_window_frames)
        per_field["decision_window_frames"][tier].append(decision_window_frames)

    for field, by_tier in per_field.items():
        for tier, values in by_tier.items():
            row = {"field": field, "tier": tier}
            row.update(summarize_finite(values))
            rows.append(row)
    return rows


def main():
    global augmented_by_id
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tier_ab_ids, tier_by_id = load_tier_ab_ids()
    print(f"Tier A+B selection audit: {len(tier_ab_ids)} candidates (expected 1097)")

    evidence_by_id = load_jsonl_by_id(EVIDENCE_JSONL, tier_ab_ids)
    augmented_by_id = load_jsonl_by_id(AUGMENTED_JSONL, tier_ab_ids)
    missing_evidence = tier_ab_ids - set(evidence_by_id)
    missing_aug = tier_ab_ids - set(augmented_by_id)
    print(f"missing evidence: {len(missing_evidence)}, missing augmented: {len(missing_aug)}")
    assert not missing_evidence and not missing_aug, "STOP: evidence coverage gap for Tier A+B"

    # 01: selection audit
    audit_rows = [{
        "expected_count": 1097, "actual_count": len(tier_ab_ids),
        "duplicate_count": 0, "missing_evidence_count": len(missing_evidence),
        "missing_augmented_count": len(missing_aug),
        "validation_candidates_used": 0,
        "tier_a_count": sum(1 for t in tier_by_id.values() if t == "A"),
        "tier_b_count": sum(1 for t in tier_by_id.values() if t == "B"),
    }]
    with open(OUT_DIR / "01_frozen_train_dataset_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        w.writeheader()
        w.writerows(audit_rows)
    print("wrote 01_frozen_train_dataset_audit.csv")

    # 02: gap distribution
    gap_rows = gap_distribution(evidence_by_id, tier_by_id)
    with open(OUT_DIR / "02_gap_distribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gap_rows[0].keys()))
        w.writeheader()
        w.writerows(gap_rows)
    print(f"wrote 02_gap_distribution.csv ({len(gap_rows)} rows)")

    # 03: ttc distribution
    ttc_rows = ttc_distribution(evidence_by_id, tier_by_id)
    with open(OUT_DIR / "03_ttc_distribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ttc_rows[0].keys()))
        w.writeheader()
        w.writerows(ttc_rows)
    print(f"wrote 03_ttc_distribution.csv ({len(ttc_rows)} rows)")

    # 04: progress distribution
    progress_rows = progress_distribution(evidence_by_id, tier_by_id)
    with open(OUT_DIR / "04_progress_distribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(progress_rows[0].keys()))
        w.writeheader()
        w.writerows(progress_rows)
    print(f"wrote 04_progress_distribution.csv ({len(progress_rows)} rows)")

    return tier_ab_ids, tier_by_id, evidence_by_id, augmented_by_id


if __name__ == "__main__":
    main()
