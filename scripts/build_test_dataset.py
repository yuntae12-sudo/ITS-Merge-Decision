#!/usr/bin/env python3
"""Builds the independent WOMD TEST dataset tiers/manifest/audit from TEST
evidence, reusing the frozen TRAIN/VALIDATION criteria byte-for-byte.

Imports the exact frozen functions (K/F/M discriminability, merge-context
eligibility, tier assignment) and thresholds -- nothing is re-tuned for
TEST. Writes ONLY under data/manifests/test/; never touches the
TRAIN/VALIDATION manifests.

Per-shard audit rows are emitted so the caller can apply the stop rule
(first shard where cumulative Tier A+B ELIGIBLE >= 400).
"""

import argparse
import csv
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.assign_merge_v2_decision_tiers as tier_mod
import scripts.classify_merge_v2_context_eligibility as elig_mod
import scripts.recompute_merge_v2_kfm_discriminability as kfm_mod
from scripts.build_merge_decision_canonical_manifest_v2 import CANDIDATE_FIELDS, write_csv

TEST_DIR = Path("data/manifests/test")
EVIDENCE = TEST_DIR / "evidence_test.jsonl"
AUGMENTED = TEST_DIR / "test_decision_evidence_augmented.jsonl"
TIER_CSV = TEST_DIR / "merge_decision_test_candidates.csv"
MANIFEST = TEST_DIR / "merge_decision_test_manifest.csv"
SPLIT_CSV = TEST_DIR / "merge_test_split.csv"
SHARD_AUDIT = TEST_DIR / "test_shard_audit.csv"
STOP_ELIGIBLE_AB = 400


def classify_all(records, aug_by_id):
    kfm_rows = kfm_mod.build_rows(
        records, aug_by_id,
        kfm_mod.SOURCE_FRONT_GAP_TIGHT_M, kfm_mod.SOURCE_FRONT_TTC_PRESSURE_S,
        kfm_mod.SOURCE_FRONT_HISTORY_PRESENCE_MIN,
    )
    kfm_by_id = {r["candidate_id"]: r for r in kfm_rows}
    out = []
    for rec in records.values():
        status, eligibility_reason = elig_mod.classify_eligibility(
            rec["automatic_decision"], rec["automatic_reason"]
        )
        e = {"candidate_id": rec["candidate_id"], "scenario_id": rec["scenario_id"],
             "automatic_decision": rec["automatic_decision"],
             "automatic_reason": rec["automatic_reason"] or "",
             "merge_context_status": status, "eligibility_reason": eligibility_reason}
        k = kfm_by_id[rec["candidate_id"]]
        tier = tier_mod.assign_tier(e, k)
        out.append({
            "candidate_id": rec["candidate_id"],
            "maneuver_id": rec.get("maneuver_id") or rec["candidate_id"],
            "scenario_id": rec["scenario_id"],
            "scene_key": rec["scene_key"],
            "source_dataset": rec.get("source_dataset", "WOMD"),
            "source_split": rec.get("source_split", "validation"),
            "source_shard": rec["source_shard"],
            "record_index": rec["record_index"],
            "split": "test",
            "decision_tier": tier,
            "dataset_role": tier_mod.ROLE_BY_TIER[tier],
            "merge_context_status": status,
            "decision_relevance": k["decision_relevance"],
            "num_meaningful_kfm_actions": k["num_meaningful_kfm_actions"],
            "archetype": k["archetype"],
            "keep_affordance": k["keep_affordance"],
            "follow_affordance": k["follow_affordance"],
            "merge_affordance": k["merge_affordance"],
            "automatic_decision": rec["automatic_decision"],
            "automatic_reason": rec["automatic_reason"],
            "source_front_present": k["source_front_present"],
        })
    return out


def is_ab_eligible(r):
    return (r["merge_context_status"] == "MERGE_CONTEXT_ELIGIBLE"
            and r["decision_tier"] in ("A", "B")
            and r["dataset_role"] in ("CORE", "SUPPORT"))


def shard_audit(rows):
    by_shard = OrderedDict()
    for r in sorted(rows, key=lambda r: r["source_shard"]):
        by_shard.setdefault(r["source_shard"], []).append(r)
    audit, cum = [], 0
    for shard, rs in by_shard.items():
        n_ab = sum(is_ab_eligible(r) for r in rs)
        cum += n_ab
        tiers = Counter(r["decision_tier"] for r in rs)
        audit.append({
            "source_shard": shard,
            "total_candidates": len(rs),
            "merge_context_eligible": sum(r["merge_context_status"] == "MERGE_CONTEXT_ELIGIBLE" for r in rs),
            "tier_A": tiers["A"], "tier_B": tiers["B"], "tier_C": tiers["C"],
            "ineligible": tiers["INELIGIBLE"],
            "eligible_AB": n_ab, "cumulative_AB": cum,
        })
    return audit


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", default=str(EVIDENCE))
    p.add_argument("--augmented", default=str(AUGMENTED))
    p.add_argument("--write", action="store_true", help="write tier CSV/manifest/split/audit (else audit only)")
    args = p.parse_args(argv)

    records = kfm_mod.load_jsonl_by_id(args.evidence)
    aug_by_id = kfm_mod.load_jsonl_by_id(args.augmented)
    assert set(records) == set(aug_by_id), "evidence/augmented candidate_id mismatch"
    rows = classify_all(records, aug_by_id)
    audit = shard_audit(rows)
    for a in audit:
        print(a)
    if args.write:
        write_csv(TIER_CSV, rows, CANDIDATE_FIELDS)
        manifest = [r for r in rows if is_ab_eligible(r)]
        write_csv(MANIFEST, manifest, CANDIDATE_FIELDS)
        write_csv(SPLIT_CSV, [
            {"maneuver_id": r["maneuver_id"], "scene_key": r["scene_key"],
             "scenario_id": r["scenario_id"], "split": "test", "dataset_role": r["dataset_role"]}
            for r in manifest], ["maneuver_id", "scene_key", "scenario_id", "split", "dataset_role"])
        write_csv(SHARD_AUDIT, audit, list(audit[0].keys()))
        print(f"manifest rows: {len(manifest)}")
    return rows, audit


if __name__ == "__main__":
    main()
