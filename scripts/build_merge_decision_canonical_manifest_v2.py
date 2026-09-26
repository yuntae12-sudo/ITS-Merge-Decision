#!/usr/bin/env python3
"""PHASE E: builds the canonical PPO Merge Decision Dataset v2 manifest from
the frozen Tier A/B/C assignments (data/manifests/v2/merge_decision_
{train,validation}_candidates_v2.csv), joined back to the source evidence
JSONL for scenario/shard/maneuver identity fields.

This is DELIBERATELY separate from scripts/build_merge_v2_manifest.py's
existing canonical pipeline (src/scenarios/v2_manifest.py), which requires
`calibration_status == "calibrated"` in configs/merge_v2.yaml AND per-
candidate `manual_validation == "CONFIRMED_MERGE"` -- neither holds yet
(calibration_status is "pending" and all 7551 TRAIN+VALIDATION candidates
are "UNREVIEWED"; only a 136-candidate human-review SAMPLE has been drawn
so far, not a full human confirmation pass). Forcing that pipeline to run
would either crash (calibration_status gate) or silently produce zero
canonical rows (CONFIRMED_MERGE gate) -- neither is the "PPO Merge Decision
Dataset" this freeze session defines. This script's canonical inclusion
policy is the NEW decision-relevance-based Tier A/B (CORE/SUPPORT) system
frozen in configs/merge_v2_decision_filter.yaml, an explicit, disjoint
concept from the older human-confirmed topology-only canonical manifest.

Never applies candidate-level random splitting -- TRAIN/VALIDATION split
here is the existing WOMD-shard-file-level split (training_tfexample vs
validation_tfexample), already scenario-disjoint (see leakage audit), not
re-derived by this script.
"""

import csv
import json
from pathlib import Path

TIER_FILES = {
    "training": "data/manifests/v2/merge_decision_train_candidates_v2.csv",
    "validation": "data/manifests/v2/merge_decision_validation_candidates_v2.csv",
}
EVIDENCE_FILES = {
    "training": "data/manifests/v2/evidence_training.jsonl",
    "validation": "data/manifests/v2/evidence_validation.jsonl",
}
OUT_DIR = Path("data/manifests/v2")


def load_tier_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_evidence_by_id(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["candidate_id"]] = rec
    return out


CANDIDATE_FIELDS = [
    "candidate_id", "maneuver_id", "scenario_id", "scene_key",
    "source_dataset", "source_split", "source_shard", "record_index",
    "split", "decision_tier", "dataset_role",
    "merge_context_status", "decision_relevance", "num_meaningful_kfm_actions",
    "archetype", "keep_affordance", "follow_affordance", "merge_affordance",
    "automatic_decision", "automatic_reason", "source_front_present",
]


def build_candidate_rows():
    all_rows = []
    for split, tier_path in TIER_FILES.items():
        tier_rows = load_tier_rows(tier_path)
        evidence_by_id = load_evidence_by_id(EVIDENCE_FILES[split])
        for r in tier_rows:
            ev = evidence_by_id[r["candidate_id"]]
            all_rows.append({
                "candidate_id": r["candidate_id"],
                "maneuver_id": ev.get("maneuver_id") or r["candidate_id"],
                "scenario_id": r["scenario_id"],
                "scene_key": ev["scene_key"],
                "source_dataset": ev.get("source_dataset", "WOMD"),
                "source_split": ev.get("source_split", split),
                "source_shard": ev["source_shard"],
                "record_index": ev["record_index"],
                "split": split,
                "decision_tier": r["decision_tier"],
                "dataset_role": r["dataset_role"],
                "merge_context_status": r["merge_context_status"],
                "decision_relevance": r["decision_relevance"],
                "num_meaningful_kfm_actions": r["num_meaningful_kfm_actions"],
                "archetype": r["archetype"],
                "keep_affordance": r["keep_affordance"],
                "follow_affordance": r["follow_affordance"],
                "merge_affordance": r["merge_affordance"],
                "automatic_decision": r["automatic_decision"],
                "automatic_reason": r["automatic_reason"],
                "source_front_present": r["source_front_present"],
            })
    return all_rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    all_rows = build_candidate_rows()
    ids = [r["candidate_id"] for r in all_rows]
    assert len(ids) == len(set(ids)), "duplicate candidate_id across splits"
    print(f"total candidates (both splits): {len(all_rows)}")

    # Full candidate-level table (all tiers, all roles -- superset).
    write_csv(OUT_DIR / "merge_decision_candidates_v2.csv", all_rows, CANDIDATE_FIELDS)

    # Canonical manifest: CORE (Tier A) + SUPPORT (Tier B) only.
    canonical_rows = [r for r in all_rows if r["dataset_role"] in ("CORE", "SUPPORT")]
    write_csv(OUT_DIR / "merge_decision_manifest_v2.csv", canonical_rows, CANDIDATE_FIELDS)
    print(f"canonical (CORE+SUPPORT): {len(canonical_rows)}")

    # Maneuver-level table -- one row per maneuver_id (1:1 with candidate_id
    # in this schema; kept as a separate named artifact per the brief's
    # requested naming convention, not because it differs in content here).
    write_csv(OUT_DIR / "merge_decision_maneuvers_v2.csv", canonical_rows, CANDIDATE_FIELDS)

    # Split manifest: scenario_id-level grouping key + split, for downstream
    # loaders that only need identity + split (mirrors dataset_split_v2.csv's
    # existing shape from v2_manifest.py, applied to this new tier system).
    split_rows = [
        {"maneuver_id": r["maneuver_id"], "scene_key": r["scene_key"],
         "scenario_id": r["scenario_id"], "split": r["split"],
         "dataset_role": r["dataset_role"]}
        for r in canonical_rows
    ]
    write_csv(OUT_DIR / "merge_decision_split_v2.csv", split_rows,
              ["maneuver_id", "scene_key", "scenario_id", "split", "dataset_role"])

    # Per-split, per-role counts for the freeze report.
    from collections import Counter
    role_counts = Counter((r["split"], r["dataset_role"]) for r in all_rows)
    for (split, role), c in sorted(role_counts.items()):
        print(f"  {split} / {role}: {c}")

    scenario_counts = Counter(r["split"] for r in all_rows)
    scenarios_by_split = {}
    for split in TIER_FILES:
        scenarios_by_split[split] = len({r["scenario_id"] for r in all_rows if r["split"] == split})
        print(f"  {split} distinct scenarios: {scenarios_by_split[split]}")

    return all_rows, canonical_rows


if __name__ == "__main__":
    main()
