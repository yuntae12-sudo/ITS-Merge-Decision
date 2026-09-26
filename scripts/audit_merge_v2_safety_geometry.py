#!/usr/bin/env python3
"""Targeted SAFETY_CRITICAL geometry audit (analysis-only, TRAIN only).

Resolves the ambiguity flagged in outputs/merge_v2_decision_audit/audit_report.md:
does a sustained negative target-lane gap (front/rear_gap_m < -0.5m for >=5
consecutive frames) reflect a genuine near-physical-conflict, or a
target-polyline arc-length projection artifact during the pre-merge approach
(ego hasn't physically entered the target lane yet, so its *projected*
position can show apparent longitudinal overlap with a laterally-separated
vehicle even though the two vehicles' actual footprints are far apart)?

Method: for each candidate's frames where the stored gap is negative, re-open
the raw tf_example record (10 distinct training shards -- same set already
used for outputs/merge_v2_calibration/human_review_rollouts) and recompute a
REAL oriented-bounding-box overlap / center-distance / clearance using
Waymax's own geometry primitives (waymax.utils.geometry.has_overlap /
compute_pairwise_overlaps), which the environment's own OverlapMetric already
uses for live collision/termination signaling (src/environment/termination.py)
-- this audit does not reimplement a new SAT test.

This produces a `diagnostic_verdict` per candidate:
    PHYSICAL_CONFLICT_LIKELY   -- negative gap AND real OBB overlap/near-zero clearance
    PROJECTION_ARTIFACT_LIKELY -- negative gap AND no real overlap, comfortable clearance
    AMBIGUOUS                  -- negative gap AND borderline clearance

`diagnostic_verdict` is NOT a human_label and is NOT used to change
automatic_decision, calibration_status, or any threshold in
configs/merge_v2.yaml.
"""

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waymax.utils import geometry as waymax_geometry

from src.scenarios.scenario_loader import build_waymax_config, iter_scenarios, load_dataset_config

TRAIN_EVIDENCE = "data/manifests/v2/evidence_training.jsonl"
DECISION_AUDIT_CSV = "outputs/merge_v2_decision_audit/training_decision_relevance.csv"
DATASET_CONFIG = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"
OUT_DIR = Path("outputs/merge_v2_decision_audit_v2")
EXISTING_GIF_ROOT = Path("outputs/merge_v2_calibration/human_review_rollouts")

TARGET_SAMPLE_SIZE = 20
CLEARANCE_ARTIFACT_M = 2.0   # comfortable circumscribed-circle clearance -> projection artifact
RNG_SEED = 20260926


def load_jsonl(path):
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records[rec["candidate_id"]] = rec
    return records


def load_decision_audit_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def max_consecutive_deep_negative_with_span(samples, field, floor_m=-0.5):
    """Same persistence definition as audit_merge_v2_decision_relevance.py's
    max_consecutive_deep_negative, but also returns the (start_frame,
    end_frame, min_value) of the longest run for targeted re-inspection."""

    best = (0, None, None, None)  # (run_len, start_frame, end_frame, min_val)
    run_len, run_start, run_min = 0, None, None
    for s in samples:
        v = s[field]
        if v is not None and v < floor_m:
            if run_len == 0:
                run_start = s["frame"]
            run_len += 1
            run_min = v if run_min is None else min(run_min, v)
            if run_len > best[0]:
                best = (run_len, run_start, s["frame"], run_min)
        else:
            run_len, run_start, run_min = 0, None, None
    return best  # (persistence_frames, start_frame, end_frame, min_gap)


def pick_stratified_sample(evidence_by_id, decision_rows):
    """Stratifies the SAFETY_CRITICAL-flagged HIGH-relevance pool by
    negative-gap magnitude (mild/moderate/large), front-vs-rear-vs-both, and
    persistence length (short/long) -- not pure random -- per audit brief
    Section 4."""

    safety_ids = {
        r["candidate_id"] for r in decision_rows
        if r["archetype"] == "SAFETY_CRITICAL" and r["decision_relevance"] == "HIGH"
    }

    candidates = []
    for cid in safety_ids:
        rec = evidence_by_id.get(cid)
        if rec is None:
            continue
        samples = rec["gap_timeseries"]
        front_run = max_consecutive_deep_negative_with_span(samples, "front_gap_m")
        rear_run = max_consecutive_deep_negative_with_span(samples, "rear_gap_m")

        side = None
        chosen = None
        if front_run[0] >= rear_run[0] and front_run[0] > 0:
            side, chosen = "front", front_run
        elif rear_run[0] > 0:
            side, chosen = "rear", rear_run
        if front_run[0] > 0 and rear_run[0] > 0:
            side = "both"
            chosen = front_run if front_run[0] >= rear_run[0] else rear_run

        if chosen is None or chosen[3] is None:
            continue

        persistence, start_f, end_f, min_gap = chosen
        if min_gap >= -2.0:
            magnitude = "mild"
        elif min_gap >= -5.0:
            magnitude = "moderate"
        else:
            magnitude = "large"
        length_bucket = "short" if persistence < 15 else "long"

        candidates.append({
            "candidate_id": cid,
            "side": side,
            "magnitude": magnitude,
            "length_bucket": length_bucket,
            "persistence_frames": persistence,
            "run_start_frame": start_f,
            "run_end_frame": end_f,
            "min_gap_m": min_gap,
        })

    rng = random.Random(RNG_SEED)
    rng.shuffle(candidates)

    strata_keys = sorted({(c["magnitude"], c["side"], c["length_bucket"]) for c in candidates})
    by_stratum = {k: [] for k in strata_keys}
    for c in candidates:
        by_stratum[(c["magnitude"], c["side"], c["length_bucket"])].append(c)

    # Guarantee MAN_0013-style coverage is irrelevant here (MAN_0013 is a
    # REJECT, never SAFETY_CRITICAL/HIGH) -- round-robin across strata
    # instead so every combination present gets at least 1 pick.
    selected = []
    seen = set()
    idx_by_stratum = {k: 0 for k in strata_keys}
    while len(selected) < TARGET_SAMPLE_SIZE:
        progressed = False
        for k in strata_keys:
            pool = by_stratum[k]
            i = idx_by_stratum[k]
            if i < len(pool):
                selected.append(pool[i])
                idx_by_stratum[k] += 1
                progressed = True
                if len(selected) >= TARGET_SAMPLE_SIZE:
                    break
        if not progressed:
            break

    return selected


def find_existing_gif(candidate_id):
    safe = candidate_id.replace("/", "_").replace("#", "_")
    for batch_dir in sorted(EXISTING_GIF_ROOT.glob("batch_*")):
        for gif in batch_dir.glob("*.gif"):
            if gif.stem == safe or safe in gif.stem:
                return str(gif)
    return None


def compute_geometry_for_candidate(record, evidence_rec, run_start_frame, run_end_frame):
    """Recomputes real OBB overlap / center distance / clearance at each
    frame in [run_start_frame, run_end_frame] using the SAME raw per-frame
    (x, y, length, width, yaw) the extraction pipeline already has in scope
    (interaction_evidence.py:44-54) but never persists to evidence JSONL.

    Reuses waymax.utils.geometry.has_overlap (the primitive behind
    OverlapMetric / termination.py's live collision signal) rather than
    reimplementing OBB/SAT overlap."""

    traj = record.state.log_trajectory
    ego_idx = record.sdc_index
    object_ids = np.asarray(record.state.object_metadata.ids)
    x = np.asarray(traj.x)
    y = np.asarray(traj.y)
    yaw = np.asarray(traj.yaw)
    length = np.asarray(traj.length)
    width = np.asarray(traj.width)
    valid = np.asarray(traj.valid).astype(bool)

    front_id = evidence_rec["interaction_evidence"]["front_vehicle_id"]
    rear_id = evidence_rec["interaction_evidence"]["rear_vehicle_id"]
    relevant_ids = {v for v in (front_id, rear_id) if v is not None}

    min_center_distance = None
    min_clearance = None
    any_overlap = False
    overlap_frames = 0
    frames_checked = 0

    for frame in range(run_start_frame, run_end_frame + 1):
        if frame >= x.shape[1] or not valid[ego_idx, frame]:
            continue
        for rel_id in relevant_ids:
            matches = np.flatnonzero(object_ids == rel_id)
            if matches.size == 0:
                continue
            other_idx = int(matches[0])
            if not valid[other_idx, frame]:
                continue

            ex, ey = float(x[ego_idx, frame]), float(y[ego_idx, frame])
            ox, oy = float(x[other_idx, frame]), float(y[other_idx, frame])
            center_distance = float(np.hypot(ex - ox, ey - oy))

            traj_5dof = np.array([
                [ex, ey, float(length[ego_idx, frame]), float(width[ego_idx, frame]), float(yaw[ego_idx, frame])],
                [ox, oy, float(length[other_idx, frame]), float(width[other_idx, frame]), float(yaw[other_idx, frame])],
            ])
            overlap_mask = np.asarray(waymax_geometry.compute_pairwise_overlaps(traj_5dof))
            overlaps = bool(overlap_mask[0, 1])

            # Footprint clearance proxy: center distance minus each box's
            # half-diagonal (sqrt((length/2)^2 + (width/2)^2)), i.e. the
            # distance at which two circumscribed circles would just touch.
            # This is a conservative (i.e. clearance is UNDERESTIMATED,
            # never overestimated) proxy for true OBB clearance -- it is
            # NOT a substitute for the actual SAT overlap check above
            # (`overlaps`), which is the ground truth used for
            # `physical_overlap_any`. A negative value here means "the
            # circumscribing circles intersect", which can happen even when
            # the real OBBs (usually narrower than their circumscribed
            # circle) do not overlap -- hence AMBIGUOUS/PHYSICAL_CONFLICT
            # verdicts below key off `overlaps` first, and only fall back to
            # this proxy as a secondary continuous signal.
            half_diag_ego = 0.5 * float(np.hypot(length[ego_idx, frame], width[ego_idx, frame]))
            half_diag_other = 0.5 * float(np.hypot(length[other_idx, frame], width[other_idx, frame]))
            clearance = center_distance - (half_diag_ego + half_diag_other)

            frames_checked += 1
            if min_center_distance is None or center_distance < min_center_distance:
                min_center_distance = center_distance
            if min_clearance is None or clearance < min_clearance:
                min_clearance = clearance
            if overlaps:
                any_overlap = True
                overlap_frames += 1

    return {
        "min_center_distance_m": min_center_distance,
        "physical_overlap_any": any_overlap,
        "physical_overlap_frames": overlap_frames,
        "min_physical_clearance_m": min_clearance,
        "frames_checked": frames_checked,
    }


def diagnostic_verdict(geom):
    """`physical_overlap_any` is the ground-truth SAT OBB check (Waymax's
    own has_overlap/OverlapMetric primitive) and is trusted directly.
    `min_physical_clearance_m` is a conservative circumscribed-circle proxy
    (see compute_geometry_for_candidate) used only when the real OBBs never
    actually overlapped -- a small/negative proxy clearance without a real
    overlap means the two circles are close but says nothing definitive
    about the true (narrower) rotated boxes, so that case is AMBIGUOUS, not
    PHYSICAL_CONFLICT_LIKELY."""

    if geom["frames_checked"] == 0:
        return "AMBIGUOUS"
    if geom["physical_overlap_any"]:
        return "PHYSICAL_CONFLICT_LIKELY"
    if geom["min_physical_clearance_m"] is not None and geom["min_physical_clearance_m"] > CLEARANCE_ARTIFACT_M:
        return "PROJECTION_ARTIFACT_LIKELY"
    return "AMBIGUOUS"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    evidence_by_id = load_jsonl(TRAIN_EVIDENCE)
    decision_rows = load_decision_audit_csv(DECISION_AUDIT_CSV)
    assert len(evidence_by_id) == 5545, f"expected 5545, got {len(evidence_by_id)}"

    sample = pick_stratified_sample(evidence_by_id, decision_rows)
    print(f"stratified SAFETY_CRITICAL sample: {len(sample)} candidates")
    strata_count = Counter((c["magnitude"], c["side"], c["length_bucket"]) for c in sample)
    for k, v in strata_count.items():
        print(f"  {k}: {v}")

    dataset_expansion = load_dataset_config(DATASET_CONFIG)

    by_shard = {}
    for c in sample:
        rec = evidence_by_id[c["candidate_id"]]
        by_shard.setdefault(rec["source_shard"], []).append(c)

    rows = []
    for shard, shard_items in sorted(by_shard.items()):
        physical_path = f"data/womd/training/{shard}"
        dataset_config = build_waymax_config(dataset_expansion, physical_path)
        print(f"[shard {shard}] {len(shard_items)} candidate(s)")
        for item in shard_items:
            cid = item["candidate_id"]
            rec = evidence_by_id[cid]
            records = list(iter_scenarios(
                dataset_config, start_index=rec["record_index"], limit=1,
                source_dataset=rec.get("source_dataset", "WOMD"),
                source_split=rec.get("source_split", "training"),
            ))
            if len(records) != 1:
                raise RuntimeError(f"could not load {cid}")
            record = records[0]

            geom = compute_geometry_for_candidate(
                record, rec, item["run_start_frame"], item["run_end_frame"]
            )
            verdict = diagnostic_verdict(geom)

            ttc_zero_frames = sum(
                1 for s in rec["gap_timeseries"]
                if (s["front_ttc_s"] == 0.0 or s["rear_ttc_s"] == 0.0)
            )
            projection_negative_frames = sum(
                1 for s in rec["gap_timeseries"]
                if (s["front_gap_m"] is not None and s["front_gap_m"] < 0)
                or (s["rear_gap_m"] is not None and s["rear_gap_m"] < 0)
            )

            gif_path = find_existing_gif(cid)
            rows.append({
                "candidate_id": cid,
                "scenario_id": rec["scenario_id"],
                "front_or_rear": item["side"],
                "negative_gap_min_m": item["min_gap_m"],
                "negative_gap_persistence_frames": item["persistence_frames"],
                "min_center_distance_m": geom["min_center_distance_m"],
                "physical_overlap_any": geom["physical_overlap_any"],
                "physical_overlap_frames": geom["physical_overlap_frames"],
                "min_physical_clearance_m": geom["min_physical_clearance_m"],
                "projection_gap_negative_frames": projection_negative_frames,
                "ttc_zero_frames": ttc_zero_frames,
                "diagnostic_verdict": verdict,
                "review_gif_path": gif_path or "",
                "notes": (
                    f"run_frames=[{item['run_start_frame']},{item['run_end_frame']}] "
                    f"frames_checked={geom['frames_checked']}"
                ),
            })
            print(f"  {cid}: verdict={verdict} min_clearance={geom['min_physical_clearance_m']}")

    out_csv = OUT_DIR / "safety_critical_targeted_audit.csv"
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_csv} ({len(rows)} rows)")

    verdict_counts = Counter(r["diagnostic_verdict"] for r in rows)
    print("verdict distribution:", dict(verdict_counts))

    with open(OUT_DIR / "safety_geometry_verdict_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["verdict", "count", "ratio"])
        for v in ("PHYSICAL_CONFLICT_LIKELY", "PROJECTION_ARTIFACT_LIKELY", "AMBIGUOUS"):
            c = verdict_counts.get(v, 0)
            w.writerow([v, c, round(c / len(rows), 4) if rows else 0])

    return rows


if __name__ == "__main__":
    main()
