#!/usr/bin/env python3
"""Augments TRAIN MERGE v2 evidence with SOURCE-lane front-vehicle evidence
(analysis-only derived artifact -- does not modify evidence_training.jsonl).

MERGE v2's stored evidence (interaction_evidence.py) only ever computes
TARGET-lane front/rear gap+TTC (see outputs/merge_v2_decision_audit/
audit_report.md's "Known Limitation": no source_front_gap equivalent
exists). But the production PPO observation (observation_builder.py:197-221)
DOES compute a source-lane front feature -- by calling the exact same
`extract_interaction_features` used for target-lane, just passing
`source_polyline` instead of `target_polyline`. This script reuses that
same function (never reimplementing selection/gap/TTC logic) against the
raw tf_example trajectory (source polylines are reconstructed the same way
`extract_merge_v2_evidence.py` does, via `reconstruct_transition`), to fill
this dataset gap.

Online-safety constraint (audit brief Section 10): the resulting
`source_front_*` fields used for KEEP/FOLLOW/MERGE affordance MUST NOT use
privileged future information. Per-frame source-front features are only
computed for frames `t <= decision_frame` (`decision_frame :=
interaction_evidence.commit_frame`, the latest frame that is not itself
future information relative to the merge decision point) and reduced to a
history summary; any post-decision-frame value is never used by this
script's own "primary" fields (it is not even computed here, to avoid the
temptation of leakage downstream).
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios.dataset_builder import reconstruct_transition
from src.scenarios.lane_assignment import load_lane_assignment_config
from src.scenarios.scenario_features import extract_interaction_features, load_agent_selection_config
from src.scenarios.scenario_loader import build_waymax_config, iter_scenarios, load_dataset_config

PHASE1_CONFIG = "configs/phase1_merge.yaml"
OUT_DIR = Path("outputs/merge_v2_decision_audit_v2")

SPLIT_DEFAULTS = {
    "training": {
        "evidence": "data/manifests/v2/evidence_training.jsonl",
        "dataset_config": "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml",
        "shard_dir": "data/womd/training",
        "out_path": OUT_DIR / "training_decision_evidence_augmented.jsonl",
        "expected_count": 5545,
    },
    "validation": {
        "evidence": "data/manifests/v2/evidence_validation.jsonl",
        "dataset_config": "outputs/phase1/validation_6shard_pilot/dataset_validation_6shard.yaml",
        "shard_dir": "data/womd/validation",
        "out_path": OUT_DIR / "validation_decision_evidence_augmented.jsonl",
        "expected_count": 2006,
    },
}


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_source_front_history(record, source_polyline, agent_config, decision_start_frame, decision_frame):
    """Computes source-lane front-vehicle features at every frame in
    [decision_start_frame, decision_frame] (inclusive, online-safe: never
    beyond decision_frame), returning both the AT-decision-frame values
    (primary affordance input) and a history summary over the window."""

    traj = record.state.log_trajectory
    ego = record.sdc_index
    object_ids = np.asarray(record.state.object_metadata.ids)
    object_types = np.asarray(record.state.object_metadata.object_types)
    x = np.asarray(traj.x)
    y = np.asarray(traj.y)
    yaw = np.asarray(traj.yaw)
    vel_x = np.asarray(traj.vel_x)
    vel_y = np.asarray(traj.vel_y)
    length = np.asarray(traj.length)
    valid = np.asarray(traj.valid).astype(bool)

    end = min(decision_frame, x.shape[1] - 1)
    samples = []
    for frame in range(decision_start_frame, end + 1):
        if not valid[ego, frame]:
            continue
        features = extract_interaction_features(
            frame_index=frame,
            target_polyline=source_polyline,  # SOURCE-lane, per observation_builder.py:201-221
            merge_distance_m=0.0,
            ego_id=record.sdc_id,
            ego_x=float(x[ego, frame]), ego_y=float(y[ego, frame]),
            ego_vel_x=float(vel_x[ego, frame]), ego_vel_y=float(vel_y[ego, frame]),
            ego_length_m=float(length[ego, frame]),
            object_ids=object_ids, object_types=object_types,
            valid=valid[:, frame], x=x[:, frame], y=y[:, frame], yaw=yaw[:, frame],
            vel_x=vel_x[:, frame], vel_y=vel_y[:, frame], length=length[:, frame],
            config=agent_config,
        )
        samples.append({
            "frame": frame,
            "front_vehicle_id": features.front_vehicle_id,
            "front_gap_m": features.front_gap_m,
            "front_relative_speed_mps": features.front_relative_speed_mps,
            "front_ttc_s": features.front_ttc_s,
        })

    if not samples:
        return None, {
            "source_front_presence_ratio_history": 0.0,
            "source_front_min_gap_history_m": None,
            "source_front_median_gap_history_m": None,
            "source_front_min_ttc_history_s": None,
            "source_front_closing_ratio_history": 0.0,
            "history_frames_checked": 0,
        }

    at_decision = samples[-1]

    present_flags = [s["front_vehicle_id"] is not None for s in samples]
    presence_ratio = sum(present_flags) / len(samples)

    gaps = [s["front_gap_m"] for s in samples if s["front_gap_m"] is not None]
    ttcs = [s["front_ttc_s"] for s in samples if s["front_ttc_s"] is not None and math.isfinite(s["front_ttc_s"])]
    closing = [
        s for s in samples
        if s["front_relative_speed_mps"] is not None and s["front_relative_speed_mps"] > 0.0
    ]

    history = {
        "source_front_presence_ratio_history": round(presence_ratio, 4),
        "source_front_min_gap_history_m": min(gaps) if gaps else None,
        "source_front_median_gap_history_m": float(np.median(gaps)) if gaps else None,
        "source_front_min_ttc_history_s": min(ttcs) if ttcs else None,
        "source_front_closing_ratio_history": round(len(closing) / len(samples), 4),
        "history_frames_checked": len(samples),
    }
    return at_decision, history


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("training", "validation"), default="training")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = SPLIT_DEFAULTS[args.split]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(cfg["evidence"])
    assert len(records) == cfg["expected_count"], (
        f"expected {cfg['expected_count']} for split={args.split!r}, got {len(records)}"
    )

    dataset_expansion = load_dataset_config(cfg["dataset_config"])
    lane_config = load_lane_assignment_config(PHASE1_CONFIG)
    agent_config = load_agent_selection_config(PHASE1_CONFIG)

    by_shard = {}
    for rec in records:
        by_shard.setdefault(rec["source_shard"], []).append(rec)

    print(f"split={args.split} input candidates: {len(records)}, distinct shards: {len(by_shard)}")

    written = 0
    failed = []
    out_path = cfg["out_path"]
    with open(out_path, "w", encoding="utf-8") as out:
        for shard, shard_recs in sorted(by_shard.items()):
            physical_path = f"{cfg['shard_dir']}/{shard}"
            dataset_config = build_waymax_config(dataset_expansion, physical_path)
            print(f"[shard {shard}] {len(shard_recs)} candidate(s)")

            # Single sequential pass over this shard's generator (matches
            # _build_scenario_id_lookup's cost model in extract_merge_v2_
            # evidence.py) -- `iter_scenarios(start_index=N, limit=1)`
            # reconstructs every record 0..N from scratch on EACH call
            # (scenario_loader.iter_scenarios docstring: "Waymax's generator
            # has no native seek"), which is O(n^2) per shard if called once
            # per candidate. Iterating the generator once and pulling out
            # every wanted record_index as it's reached is O(n) instead.
            rec_by_index = {rec["record_index"]: rec for rec in shard_recs}
            wanted_indices = set(rec_by_index)
            max_wanted = max(wanted_indices)

            records_by_index = {}
            for loaded_record in iter_scenarios(
                dataset_config, start_index=0, limit=max_wanted + 1,
                source_dataset=shard_recs[0].get("source_dataset", "WOMD"),
                source_split=shard_recs[0].get("source_split", args.split),
            ):
                if loaded_record.record_index in wanted_indices:
                    records_by_index[loaded_record.record_index] = loaded_record
                if loaded_record.record_index >= max_wanted:
                    break

            for rec in shard_recs:
                cid = rec["candidate_id"]
                try:
                    record = records_by_index.get(rec["record_index"])
                    if record is None:
                        raise RuntimeError(f"could not load record_index={rec['record_index']}")

                    te = rec["topology_evidence"]
                    ie = rec["interaction_evidence"]
                    transition, source_polyline, target_polyline, _ = reconstruct_transition(
                        record, lane_config,
                        ie["commit_frame"], te["source_lane_id"], te["target_lane_id"], cid,
                    )
                    if source_polyline is None:
                        raise ValueError("source_polyline is None after reconstruct_transition")

                    decision_start_frame = ie["decision_start_frame"]
                    decision_frame = ie["commit_frame"]  # online-safe cutoff: never beyond commit

                    at_decision, history = compute_source_front_history(
                        record, source_polyline, agent_config, decision_start_frame, decision_frame
                    )

                    out_row = {
                        "candidate_id": cid,
                        "source_front_present": bool(at_decision is not None and at_decision["front_vehicle_id"] is not None) if at_decision else False,
                        "source_front_gap_m": at_decision["front_gap_m"] if at_decision else None,
                        "source_front_relative_speed_mps": at_decision["front_relative_speed_mps"] if at_decision else None,
                        "source_front_ttc_s": at_decision["front_ttc_s"] if at_decision else None,
                        **history,
                        "extraction_failed": False,
                        "extraction_error": "",
                    }
                except Exception as exc:  # noqa: BLE001 - recorded per-candidate, never silently skipped
                    failed.append((cid, repr(exc)))
                    out_row = {
                        "candidate_id": cid,
                        "source_front_present": False,
                        "source_front_gap_m": None,
                        "source_front_relative_speed_mps": None,
                        "source_front_ttc_s": None,
                        "source_front_presence_ratio_history": None,
                        "source_front_min_gap_history_m": None,
                        "source_front_median_gap_history_m": None,
                        "source_front_min_ttc_history_s": None,
                        "source_front_closing_ratio_history": None,
                        "history_frames_checked": 0,
                        "extraction_failed": True,
                        "extraction_error": repr(exc),
                    }
                    print(f"  FAILED {cid}: {exc!r}")

                # sanitize inf/nan for strict JSON
                for k, v in list(out_row.items()):
                    if isinstance(v, float) and not math.isfinite(v):
                        out_row[k] = None if v != v else ("Infinity" if v > 0 else "-Infinity")

                out.write(json.dumps(out_row) + "\n")
                out.flush()
                written += 1

    print(f"written={written} failed={len(failed)}")
    if failed:
        print("failures:")
        for cid, err in failed[:20]:
            print(f"  {cid}: {err}")

    # sanity checks
    out_ids = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out_ids.add(r["candidate_id"])
    in_ids = {r["candidate_id"] for r in records}
    print(f"input={len(in_ids)} output={len(out_ids)} missing={len(in_ids - out_ids)} "
          f"duplicate_check_unique={len(out_ids)}")
    assert in_ids == out_ids, "candidate_id set mismatch between input and augmented output"

    return written, failed


if __name__ == "__main__":
    main()
