#!/usr/bin/env python3
"""Writes data/manifests/test/test_freeze.json: provenance freeze for the
independent WOMD TEST dataset. Recomputes every audit number from the
frozen files (nothing hand-entered). Read-only w.r.t. TRAIN/VALIDATION."""

import csv
import datetime
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

TEST_DIR = Path("data/manifests/test")
FILTER_CONFIG = "configs/merge_v2_decision_filter.yaml"
CANON_MANIFEST = "data/manifests/merge_decision_manifest.csv"
STOP_THRESHOLD = 400


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    manifest_path = TEST_DIR / "merge_decision_test_manifest.csv"
    manifest = read(manifest_path)
    audit = read(TEST_DIR / "test_shard_audit.csv")
    all_cands = read(TEST_DIR / "merge_decision_test_candidates.csv")
    canon = read(CANON_MANIFEST)

    shard_idx = [int(a["source_shard"].split("tfrecord-")[1][:5]) for a in audit]
    first, last = min(shard_idx), max(shard_idx)
    assert shard_idx == list(range(first, last + 1)), "shards not contiguous"
    cum = [int(a["cumulative_AB"]) for a in audit]
    assert first == 6 and cum[-1] >= STOP_THRESHOLD and all(c < STOP_THRESHOLD for c in cum[:-1])

    ts = {r["scenario_id"] for r in manifest}
    tr = {r["scenario_id"] for r in canon if r["split"] == "training"}
    va = {r["scenario_id"] for r in canon if r["split"] == "validation"}
    all_tier = Counter(r["decision_tier"] for r in all_cands)
    freeze = {
        "version": "test-v1.0",
        "freeze_timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_split": "validation",
        "source_shards": [a["source_shard"] for a in audit],
        "first_shard": f"{first:05d}",
        "last_shard": f"{last:05d}",
        "stop_rule": f"first shard where cumulative eligible (MERGE_CONTEXT_ELIGIBLE, Tier A/B, CORE/SUPPORT) >= {STOP_THRESHOLD}",
        "per_shard_cumulative_eligible": {f"{i:05d}": c for i, c in zip(shard_idx, cum)},
        "final_eligible_maneuver_count": len(manifest),
        "unique_scenario_count": len(ts),
        "tier_A_count": sum(r["decision_tier"] == "A" for r in manifest),
        "tier_B_count": sum(r["decision_tier"] == "B" for r in manifest),
        "audit_counts_all_candidates": {
            "total": len(all_cands), "A": all_tier["A"], "B": all_tier["B"],
            "C": all_tier["C"], "INELIGIBLE": all_tier["INELIGIBLE"],
        },
        "train_scenario_overlap": len(ts & tr),
        "validation_scenario_overlap": len(ts & va),
        "maneuver_duplicate": len(manifest) - len({r["maneuver_id"] for r in manifest}),
        "candidate_duplicate": len(manifest) - len({r["candidate_id"] for r in manifest}),
        "maneuver_overlap_with_canonical": len({r["maneuver_id"] for r in manifest} & {r["maneuver_id"] for r in canon}),
        "candidate_overlap_with_canonical": len({r["candidate_id"] for r in manifest} & {r["candidate_id"] for r in canon}),
        "filter_config": FILTER_CONFIG,
        "filter_config_sha256": sha256(FILTER_CONFIG),
        "dataset_config": "configs/dataset_test.yaml",
        "test_manifest": str(manifest_path),
        "test_manifest_sha256": sha256(manifest_path),
        "test_split_manifest": str(TEST_DIR / "merge_test_split.csv"),
        "test_split_manifest_sha256": sha256(TEST_DIR / "merge_test_split.csv"),
        "test_evidence": str(TEST_DIR / "evidence_test.jsonl"),
        "test_evidence_sha256": sha256(TEST_DIR / "evidence_test.jsonl"),
        "canonical_manifest_sha256_at_freeze": sha256(CANON_MANIFEST),
        "policy_evaluation_on_test": "NOT RUN",
    }
    assert freeze["train_scenario_overlap"] == 0 and freeze["validation_scenario_overlap"] == 0
    (TEST_DIR / "test_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps({k: freeze[k] for k in ("final_eligible_maneuver_count", "unique_scenario_count", "tier_A_count", "tier_B_count", "last_shard")}))


if __name__ == "__main__":
    main()
