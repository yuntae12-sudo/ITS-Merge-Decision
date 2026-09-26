#!/usr/bin/env python3
"""PHASE D+: applies the TRAIN-frozen K/F/M discriminability + merge-context
eligibility + tier criteria to VALIDATION, byte-for-byte identical to how
TRAIN was computed -- imports the exact frozen functions from the TRAIN
scripts (recompute_merge_v2_kfm_discriminability.py,
classify_merge_v2_context_eligibility.py) instead of reimplementing them, so
there is no possibility of the two splits' logic silently drifting apart.

Never re-tunes any threshold based on VALIDATION's own distribution -- the
`gap_thresh`/`ttc_thresh`/`history_thresh` passed to `build_rows` are the
exact "current" values frozen in configs/merge_v2_decision_filter.yaml
(SOURCE_FRONT_GAP_TIGHT_M / SOURCE_FRONT_TTC_PRESSURE_S /
SOURCE_FRONT_HISTORY_PRESENCE_MIN from the TRAIN script, imported directly,
not restated as separate literals here).
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.recompute_merge_v2_kfm_discriminability as kfm_mod
import scripts.classify_merge_v2_context_eligibility as elig_mod
import scripts.assign_merge_v2_decision_tiers as tier_mod

VALIDATION_EVIDENCE = "data/manifests/v2/evidence_validation.jsonl"
VALIDATION_AUGMENTED = "outputs/merge_v2_decision_audit_v2/validation_decision_evidence_augmented.jsonl"
OUT_DIR = Path("outputs/merge_v2_decision_audit_v2")
EXPECTED_COUNT = 2006


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    val_records = kfm_mod.load_jsonl_by_id(VALIDATION_EVIDENCE)
    aug_by_id = kfm_mod.load_jsonl_by_id(VALIDATION_AUGMENTED)
    n = len(val_records)
    assert n == EXPECTED_COUNT, f"expected {EXPECTED_COUNT}, got {n}"
    assert set(val_records) == set(aug_by_id), "candidate_id mismatch between VALIDATION evidence and augmented evidence"
    print(f"loaded {n} VALIDATION records, {len(aug_by_id)} augmented source-front records")

    # 1. K/F/M discriminability -- frozen thresholds imported directly from
    # the TRAIN script's module-level constants (not restated).
    kfm_rows = kfm_mod.build_rows(
        val_records, aug_by_id,
        kfm_mod.SOURCE_FRONT_GAP_TIGHT_M, kfm_mod.SOURCE_FRONT_TTC_PRESSURE_S,
        kfm_mod.SOURCE_FRONT_HISTORY_PRESENCE_MIN,
    )
    kfm_by_id = {r["candidate_id"]: r for r in kfm_rows}

    kfm_csv = OUT_DIR / "validation_kfm_discriminability.csv"
    with open(kfm_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kfm_rows[0].keys()))
        w.writeheader()
        for row in kfm_rows:
            w.writerow(row)
    print(f"wrote {kfm_csv} ({len(kfm_rows)} rows)")

    # 2. Merge-context eligibility -- frozen classify_eligibility function.
    elig_rows = []
    for rec in val_records.values():
        te = rec["topology_evidence"]
        ie = rec["interaction_evidence"]
        decision = rec["automatic_decision"]
        reason = rec["automatic_reason"]
        status, eligibility_reason = elig_mod.classify_eligibility(decision, reason)
        topology_valid = status != "MERGE_CONTEXT_INELIGIBLE"
        merge_zone_relevant = (
            ie["front_vehicle_id"] is not None or ie["rear_vehicle_id"] is not None
            or bool(ie["conflict_vehicle_ids"])
        )
        decision_window_valid = ie["commit_frame"] > ie["decision_start_frame"]
        elig_rows.append({
            "candidate_id": rec["candidate_id"],
            "scenario_id": rec["scenario_id"],
            "automatic_decision": decision,
            "automatic_reason": reason or "",
            "merge_context_status": status,
            "eligibility_reason": eligibility_reason,
            "topology_valid": topology_valid,
            "merge_zone_relevant": merge_zone_relevant,
            "decision_window_valid": decision_window_valid,
        })
    elig_by_id = {r["candidate_id"]: r for r in elig_rows}

    elig_csv = Path("data/manifests/v2/validation_merge_context_eligibility.csv")
    with open(elig_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(elig_rows[0].keys()))
        w.writeheader()
        for row in elig_rows:
            w.writerow(row)
    print(f"wrote {elig_csv} ({len(elig_rows)} rows)")

    # 3. Tier assignment -- frozen assign_tier function.
    tier_rows = []
    for cid, e in elig_by_id.items():
        k = kfm_by_id[cid]
        tier = tier_mod.assign_tier(e, k)
        tier_rows.append({
            "candidate_id": cid,
            "scenario_id": e["scenario_id"],
            "merge_context_status": e["merge_context_status"],
            "decision_tier": tier,
            "dataset_role": tier_mod.ROLE_BY_TIER[tier],
            "decision_relevance": k["decision_relevance"],
            "num_meaningful_kfm_actions": k["num_meaningful_kfm_actions"],
            "archetype": k["archetype"],
            "automatic_decision": e["automatic_decision"],
            "automatic_reason": e["automatic_reason"],
            "source_front_present": k["source_front_present"],
            "keep_affordance": k["keep_affordance"],
            "follow_affordance": k["follow_affordance"],
            "merge_affordance": k["merge_affordance"],
            "split": "validation",
        })

    ids = [r["candidate_id"] for r in tier_rows]
    assert len(ids) == len(set(ids)), "duplicate candidate_id"

    tier_csv = Path("data/manifests/v2/merge_decision_validation_candidates_v2.csv")
    with open(tier_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tier_rows[0].keys()))
        w.writeheader()
        for row in tier_rows:
            w.writerow(row)
    print(f"wrote {tier_csv} ({len(tier_rows)} rows)")

    from collections import Counter
    tier_counts = Counter(r["decision_tier"] for r in tier_rows)
    for t in ("A", "B", "C", "INELIGIBLE"):
        c = tier_counts.get(t, 0)
        print(f"  Tier {t}: {c} ({c/n:.2%})")

    elig_counts = Counter(r["merge_context_status"] for r in elig_rows)
    for s in ("MERGE_CONTEXT_ELIGIBLE", "MERGE_CONTEXT_INELIGIBLE", "MERGE_CONTEXT_UNCERTAIN"):
        c = elig_counts.get(s, 0)
        print(f"  {s}: {c} ({c/n:.2%})")

    return tier_rows, elig_rows, kfm_rows


if __name__ == "__main__":
    main()
