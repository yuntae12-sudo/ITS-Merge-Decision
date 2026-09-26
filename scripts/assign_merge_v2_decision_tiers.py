#!/usr/bin/env python3
"""PHASE C: cross merge_context_status with K/F/M decision relevance and
assign TRAIN Tier A/B/C/INELIGIBLE, per the freeze session's design:

  INELIGIBLE:  merge_context_status == MERGE_CONTEXT_INELIGIBLE
               (not a real merge context at all -- excluded regardless of
               K/F/M score, since a high K/F/M score on a lane-change or
               serial-continuation scene is not evidence of a merge
               decision problem)

  Tier A:      MERGE_CONTEXT_ELIGIBLE AND num_meaningful_kfm_actions >= 2
               AND archetype not in (PATH_FOLLOWING_LIKE,) AND
               decision_relevance != LOW
               (PPO training core -- confirmed multi-action tradeoff in a
               confirmed real merge context)

  Tier B:      MERGE_CONTEXT_ELIGIBLE AND num_meaningful_kfm_actions >= 1
               AND not already Tier A
               (single dominant but meaningful interaction, or a clear
               merge opportunity / approach-keep context with real
               coverage value)

  Tier C:      everything else that is not INELIGIBLE:
               - MERGE_CONTEXT_ELIGIBLE but num_meaningful_kfm_actions == 0
                 (trivial/path-following-like even in a real merge context)
               - MERGE_CONTEXT_UNCERTAIN (genuine ambiguity -- not
                 confidently a decision context, held out of the core
                 rather than discarded)

Tiers are NOT action ground-truth. Rare archetypes (REAR_PRESSURE,
BOTH_SIDES_CONSTRAINED) are never down-weighted by this assignment -- Tier
membership is a fixed rule per candidate, with no random subsampling here
(coverage is reported separately, not enforced by dropping rows).
"""

import csv
from pathlib import Path

ELIGIBILITY_CSV = "data/manifests/v2/training_merge_context_eligibility.csv"
KFM_CSV = "outputs/merge_v2_decision_audit_v2/training_kfm_discriminability.csv"
OUT_CSV = "data/manifests/v2/merge_decision_train_candidates_v2.csv"


def load_csv_by_id(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["candidate_id"]: r for r in csv.DictReader(f)}


def assign_tier(elig_row, kfm_row):
    status = elig_row["merge_context_status"]
    if status == "MERGE_CONTEXT_INELIGIBLE":
        return "INELIGIBLE"

    n_meaningful = int(kfm_row["num_meaningful_kfm_actions"])
    archetype = kfm_row["archetype"]
    relevance = kfm_row["decision_relevance"]

    if status == "MERGE_CONTEXT_UNCERTAIN":
        return "C"

    # status == MERGE_CONTEXT_ELIGIBLE from here on
    if n_meaningful >= 2 and archetype != "PATH_FOLLOWING_LIKE" and relevance != "LOW":
        return "A"
    if n_meaningful >= 1:
        return "B"
    return "C"


ROLE_BY_TIER = {"A": "CORE", "B": "SUPPORT", "C": "EXCLUDED", "INELIGIBLE": "EXCLUDED"}


def main():
    elig = load_csv_by_id(ELIGIBILITY_CSV)
    kfm = load_csv_by_id(KFM_CSV)
    assert set(elig) == set(kfm), "candidate_id set mismatch between eligibility and KFM tables"
    n = len(elig)
    print(f"loaded {n} TRAIN candidates")

    rows = []
    for cid, e in elig.items():
        k = kfm[cid]
        tier = assign_tier(e, k)
        rows.append({
            "candidate_id": cid,
            "scenario_id": e["scenario_id"],
            "merge_context_status": e["merge_context_status"],
            "decision_tier": tier,
            "dataset_role": ROLE_BY_TIER[tier],
            "decision_relevance": k["decision_relevance"],
            "num_meaningful_kfm_actions": k["num_meaningful_kfm_actions"],
            "archetype": k["archetype"],
            "automatic_decision": e["automatic_decision"],
            "automatic_reason": e["automatic_reason"],
            "source_front_present": k["source_front_present"],
            "keep_affordance": k["keep_affordance"],
            "follow_affordance": k["follow_affordance"],
            "merge_affordance": k["merge_affordance"],
            "split": "training",
        })

    ids = [r["candidate_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate candidate_id"

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")

    from collections import Counter
    tier_counts = Counter(r["decision_tier"] for r in rows)
    for t in ("A", "B", "C", "INELIGIBLE"):
        c = tier_counts.get(t, 0)
        print(f"  Tier {t}: {c} ({c/n:.2%})")

    # Tier A detail: K/F/M breakdown + archetype coverage
    tier_a = [r for r in rows if r["decision_tier"] == "A"]
    tier_b = [r for r in rows if r["decision_tier"] == "B"]
    print(f"\nTier A (n={len(tier_a)}):")
    keep_m = sum(1 for r in tier_a if r["keep_affordance"] in ("MEDIUM", "HIGH"))
    follow_m = sum(1 for r in tier_a if r["follow_affordance"] in ("MEDIUM", "HIGH"))
    merge_m = sum(1 for r in tier_a if r["merge_affordance"] in ("MEDIUM", "HIGH"))
    print(f"  KEEP meaningful: {keep_m} ({keep_m/len(tier_a):.2%})" if tier_a else "  (empty)")
    print(f"  FOLLOW meaningful: {follow_m} ({follow_m/len(tier_a):.2%})" if tier_a else "")
    print(f"  MERGE meaningful: {merge_m} ({merge_m/len(tier_a):.2%})" if tier_a else "")
    arch_counts = Counter(r["archetype"] for r in tier_a)
    print("  archetype coverage:", dict(arch_counts))

    print(f"\nTier B (n={len(tier_b)}):")
    arch_counts_b = Counter(r["archetype"] for r in tier_b)
    print("  archetype coverage:", dict(arch_counts_b))

    # rare archetype check across A+B
    for rare in ("REAR_PRESSURE", "BOTH_SIDES_CONSTRAINED"):
        in_ab = sum(1 for r in tier_a + tier_b if r["archetype"] == rare)
        total_rare = sum(1 for r in rows if r["archetype"] == rare)
        print(f"  rare archetype {rare}: {in_ab}/{total_rare} preserved in A+B")

    core_support = len(tier_a) + len(tier_b)
    print(f"\nCORE+SUPPORT total: {core_support} ({core_support/n:.2%})")

    return rows


if __name__ == "__main__":
    main()
