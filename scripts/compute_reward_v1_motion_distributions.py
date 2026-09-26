#!/usr/bin/env python3
"""Reward V1 Spec Section 11-12: Speed / Acceleration / Jerk distributions
for TRAIN Tier A+B (1097 candidates), computed via a single cheap
sequential pass per training shard (10 shards total, same set already used
throughout this dataset's audits) -- NOT a re-run of the 437GB Scenario
Proto full scan (forbidden by this session's brief).

Speed/accel/jerk are not stored in evidence_training.jsonl (confirmed by
schema inspection -- only gap/TTC/progress are). This script derives them
directly from each candidate's raw ego log_trajectory (vel_x, vel_y over
the decision window [decision_start_frame, commit_frame]), the same
per-frame arrays already read for source-front augmentation, reusing that
established access pattern.

ego speed_mps: hypot(vel_x, vel_y) -- matches observation_builder.py's
`_scalar_speed` definition exactly (v_e), not a lane-tangent projection.

acceleration_mps2: finite-difference of speed_mps over consecutive valid
frames / dt (0.1s, WOMD fixed timestep).
jerk_mps3: finite-difference of acceleration_mps2 / dt.
"""

import csv
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from src.scenarios.scenario_loader import build_waymax_config, iter_scenarios, load_dataset_config

TIER_MANIFEST = "data/manifests/v2/merge_decision_train_candidates_v2.csv"
EVIDENCE_JSONL = "data/manifests/v2/evidence_training.jsonl"
DATASET_CONFIG = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"
OUT_DIR = Path("outputs/reward_v1_spec")
DT_S = 0.1


def load_tier_ab():
    with open(TIER_MANIFEST, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["decision_tier"] in ("A", "B")]
    assert len(rows) == 1097
    return {r["candidate_id"]: r["decision_tier"] for r in rows}


def load_evidence_by_id(ids):
    out = {}
    with open(EVIDENCE_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["candidate_id"] in ids:
                out[rec["candidate_id"]] = rec
    return out


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


def summarize(values):
    if not values:
        return {"valid_count": 0, "min": None, "max": None, "mean": None, "std": None, **quantiles([])}
    stats = {
        "valid_count": len(values), "min": min(values), "max": max(values),
        "mean": statistics.fmean(values), "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }
    stats.update(quantiles(values))
    return stats


def per_candidate_motion(record, ie):
    traj = record.state.log_trajectory
    ego = record.sdc_index
    vel_x = np.asarray(traj.vel_x)[ego]
    vel_y = np.asarray(traj.vel_y)[ego]
    valid = np.asarray(traj.valid)[ego].astype(bool)

    start, end = ie["decision_start_frame"], ie["commit_frame"]
    speeds = []
    frames = []
    for frame in range(start, min(end, vel_x.shape[0] - 1) + 1):
        if not valid[frame]:
            continue
        speeds.append(float(np.hypot(vel_x[frame], vel_y[frame])))
        frames.append(frame)

    accels = []
    for i in range(1, len(speeds)):
        if frames[i] - frames[i - 1] == 1:  # only consecutive valid frames
            accels.append((speeds[i] - speeds[i - 1]) / DT_S)

    jerks = []
    for i in range(1, len(accels)):
        jerks.append((accels[i] - accels[i - 1]) / DT_S)

    decision_point_speed = speeds[0] if speeds else None
    approach_speed = speeds[-1] if speeds else None  # speed at commit frame

    return speeds, accels, jerks, decision_point_speed, approach_speed


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tier_by_id = load_tier_ab()
    evidence_by_id = load_evidence_by_id(set(tier_by_id))
    assert len(evidence_by_id) == 1097

    dataset_expansion = load_dataset_config(DATASET_CONFIG)
    by_shard = {}
    for cid, rec in evidence_by_id.items():
        by_shard.setdefault(rec["source_shard"], []).append(cid)

    print(f"Tier A+B: {len(evidence_by_id)} candidates across {len(by_shard)} shards")

    all_speeds, all_accels, all_jerks = [], [], []
    tier_speeds = {"A": [], "B": []}
    decision_speeds, approach_speeds = [], []
    source_front_present_speeds = []
    failed = []

    aug_by_id = {}
    with open("outputs/merge_v2_decision_audit_v2/training_decision_evidence_augmented.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["candidate_id"] in evidence_by_id:
                aug_by_id[r["candidate_id"]] = r

    for shard, cids in sorted(by_shard.items()):
        physical_path = f"data/womd/training/{shard}"
        dataset_config = build_waymax_config(dataset_expansion, physical_path)
        rec_by_index = {evidence_by_id[cid]["record_index"]: cid for cid in cids}
        wanted_indices = set(rec_by_index)
        max_wanted = max(wanted_indices)
        print(f"[shard {shard}] {len(cids)} candidate(s)")

        records_by_index = {}
        for loaded in iter_scenarios(dataset_config, start_index=0, limit=max_wanted + 1,
                                     source_dataset="WOMD", source_split="training"):
            if loaded.record_index in wanted_indices:
                records_by_index[loaded.record_index] = loaded
            if loaded.record_index >= max_wanted:
                break

        for record_index, cid in rec_by_index.items():
            record = records_by_index.get(record_index)
            if record is None:
                failed.append(cid)
                continue
            ie = evidence_by_id[cid]["interaction_evidence"]
            try:
                speeds, accels, jerks, dspeed, aspeed = per_candidate_motion(record, ie)
            except Exception as exc:  # noqa: BLE001
                failed.append((cid, repr(exc)))
                continue
            all_speeds.extend(speeds)
            all_accels.extend(accels)
            all_jerks.extend(jerks)
            tier_speeds[tier_by_id[cid]].extend(speeds)
            if dspeed is not None:
                decision_speeds.append(dspeed)
            if aspeed is not None:
                approach_speeds.append(aspeed)
            if aug_by_id.get(cid, {}).get("source_front_present"):
                source_front_present_speeds.extend(speeds)

    print(f"failed candidates: {len(failed)}")

    rows = []
    rows.append({"field": "ego_speed_mps_all_frames", "tier": "A+B", **summarize(all_speeds)})
    rows.append({"field": "ego_speed_mps_all_frames", "tier": "A", **summarize(tier_speeds["A"])})
    rows.append({"field": "ego_speed_mps_all_frames", "tier": "B", **summarize(tier_speeds["B"])})
    rows.append({"field": "ego_speed_mps_at_decision_start", "tier": "A+B", **summarize(decision_speeds)})
    rows.append({"field": "ego_speed_mps_at_commit_approach", "tier": "A+B", **summarize(approach_speeds)})
    rows.append({"field": "ego_speed_mps_source_front_present_subset", "tier": "A+B", **summarize(source_front_present_speeds)})
    rows.append({"field": "ego_abs_acceleration_mps2", "tier": "A+B", **summarize([abs(a) for a in all_accels])})
    rows.append({"field": "ego_signed_acceleration_mps2", "tier": "A+B", **summarize(all_accels)})
    rows.append({"field": "ego_abs_jerk_mps3", "tier": "A+B", **summarize([abs(j) for j in all_jerks])})
    rows.append({"field": "ego_signed_jerk_mps3", "tier": "A+B", **summarize(all_jerks)})

    fieldnames = list(rows[0].keys())
    with open(OUT_DIR / "05_speed_distribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows if False else [r for r in rows if r["field"].startswith("ego_speed")]:
            w.writerow(r)

    with open(OUT_DIR / "06_acceleration_jerk_distribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if "acceleration" in r["field"] or "jerk" in r["field"]:
                w.writerow(r)

    print("wrote 05_speed_distribution.csv, 06_acceleration_jerk_distribution.csv")
    for r in rows:
        print(f"  {r['field']} [{r['tier']}]: n={r['valid_count']} mean={r.get('mean')} p50={r.get('p50')}")

    return rows, failed


if __name__ == "__main__":
    main()
