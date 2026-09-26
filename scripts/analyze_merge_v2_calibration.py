#!/usr/bin/env python3
"""Analysis-only calibration material generator for MERGE v2 thresholds.

Reads TRAIN evidence only (never VALIDATION -- that split must stay
unseen until final evaluation) from ``data/manifests/v2/
evidence_training.jsonl`` and produces distribution/sensitivity/manual
-review artifacts under ``outputs/merge_v2_calibration/``.

Does not change ``configs/merge_v2.yaml`` or ``calibration_status``, and
does not call any classifier with different semantics than
``src.scenarios.merge_v2.classify_v2_merge`` -- sensitivity re-runs the
real classifier against reconstructed TopologyEvidence/InteractionEvidence,
never an independent per-feature approximation.
"""

import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios.merge_v2 import (
    InteractionEvidence,
    TopologyEvidence,
    classify_v2_merge,
)

TRAIN_EVIDENCE = "data/manifests/v2/evidence_training.jsonl"
OUT_DIR = Path("outputs/merge_v2_calibration")

CLASSIFY_KWARGS_KEYS = (
    "min_source_occupancy_frames",
    "min_target_stable_frames",
    "min_longitudinal_progress_m",
    "min_cut_in_lateral_displacement_m",
)

# required_gap_order_persistence_frames is not a classify_v2_merge()
# kwarg -- it lives on InteractionEvidence itself (see to_interaction()'s
# override), so it is tracked here for sensitivity iteration but must be
# excluded from the **kwargs passed to classify_v2_merge().
CURRENT_THRESHOLDS_ALL = dict(
    min_source_occupancy_frames=5,
    min_target_stable_frames=5,
    min_longitudinal_progress_m=0.1,
    min_cut_in_lateral_displacement_m=1.0,
    required_gap_order_persistence_frames=5,
)
CURRENT_THRESHOLDS = {k: v for k, v in CURRENT_THRESHOLDS_ALL.items() if k in CLASSIFY_KWARGS_KEYS}


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def to_topology(d):
    return TopologyEvidence(
        source_lane_id=d["source_lane_id"],
        target_lane_id=d["target_lane_id"],
        source_exit_lane_ids=tuple(d["source_exit_lane_ids"]),
        target_entry_lane_ids=tuple(d["target_entry_lane_ids"]),
        source_exits_to_target=d["source_exits_to_target"],
        target_is_roundabout=d["target_is_roundabout"],
        lanes_overlap_longitudinally=d["lanes_overlap_longitudinally"],
        terminal_heading_difference_deg=d["terminal_heading_difference_deg"],
        terminal_collinear_offset_m=d["terminal_collinear_offset_m"],
        endpoint_gap_m=d["endpoint_gap_m"],
        is_lane_change_target=d["is_lane_change_target"],
        is_intersection_transition=d["is_intersection_transition"],
        source_exit_lane_count=d["source_exit_lane_count"],
        target_polyline_point_count=d["target_polyline_point_count"],
    )


def to_interaction(d, *, required_gap_order_persistence_frames=None):
    return InteractionEvidence(
        decision_start_frame=d["decision_start_frame"],
        commit_frame=d["commit_frame"],
        completion_frame=d["completion_frame"],
        source_occupancy_frames=d["source_occupancy_frames"],
        target_stable_frames=d["target_stable_frames"],
        longitudinal_progress_m=d["longitudinal_progress_m"],
        lateral_displacement_m=d["lateral_displacement_m"],
        front_vehicle_id=d["front_vehicle_id"],
        rear_vehicle_id=d["rear_vehicle_id"],
        conflict_vehicle_ids=tuple(d["conflict_vehicle_ids"]),
        gap_order_valid_at_entry=d["gap_order_valid_at_entry"],
        gap_order_persistence_frames=d["gap_order_persistence_frames"],
        required_gap_order_persistence_frames=(
            required_gap_order_persistence_frames
            if required_gap_order_persistence_frames is not None
            else d["required_gap_order_persistence_frames"]
        ),
    )


def quantiles(values):
    if not values:
        return {}
    values = sorted(values)
    n = len(values)

    def pct(p):
        if n == 1:
            return values[0]
        idx = p / 100 * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return values[lo]
        frac = idx - lo
        return values[lo] + (values[hi] - values[lo]) * frac

    return {
        "min": values[0],
        "max": values[-1],
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if n > 1 else 0.0,
        "p1": pct(1), "p5": pct(5), "p10": pct(10), "p25": pct(25),
        "p50": pct(50), "p75": pct(75), "p90": pct(90), "p95": pct(95),
        "p99": pct(99),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    records = load_records(TRAIN_EVIDENCE)
    n = len(records)
    print(f"loaded {n} TRAIN evidence records")

    # -- sanity --
    ids = [r["candidate_id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate candidate_id in TRAIN evidence"
    assert n == 5545, f"expected 5545 TRAIN evidence records, got {n}"

    # ================= B. decision distribution =================
    decision_counts = {}
    reject_reason_counts = {}
    for r in records:
        d = r["automatic_decision"]
        decision_counts[d] = decision_counts.get(d, 0) + 1
        if d == "reject":
            reason = r["automatic_reason"]
            reject_reason_counts[reason] = reject_reason_counts.get(reason, 0) + 1

    assert sum(decision_counts.values()) == n
    assert decision_counts.get("reject", 0) == sum(reject_reason_counts.values())

    with open(OUT_DIR / "training_decision_distribution.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["decision", "count", "ratio"])
        for d, c in sorted(decision_counts.items(), key=lambda kv: -kv[1]):
            w.writerow([d, c, round(c / n, 4)])

    with open(OUT_DIR / "training_reject_reason_distribution.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reason", "count", "ratio_of_reject", "ratio_of_total"])
        reject_total = decision_counts.get("reject", 0)
        for reason, c in sorted(reject_reason_counts.items(), key=lambda kv: -kv[1]):
            w.writerow([reason, c, round(c / reject_total, 4), round(c / n, 4)])

    print("decision counts:", decision_counts)
    print("reject reasons:", reject_reason_counts)

    # ================= C. threshold feature summary =================
    feature_getters = {
        "source_occupancy_frames": lambda r: r["interaction_evidence"]["source_occupancy_frames"],
        "target_stable_frames": lambda r: r["interaction_evidence"]["target_stable_frames"],
        "longitudinal_progress_m": lambda r: r["interaction_evidence"]["longitudinal_progress_m"],
        "lateral_displacement_m": lambda r: r["interaction_evidence"]["lateral_displacement_m"],
        "gap_order_persistence_frames": lambda r: r["interaction_evidence"]["gap_order_persistence_frames"],
    }

    feature_rows = []
    per_feature_values = {}
    per_feature_by_decision = {}
    for name, getter in feature_getters.items():
        all_vals = []
        by_decision = {"accept": [], "reject": [], "review": []}
        for r in records:
            try:
                v = getter(r)
            except (KeyError, TypeError):
                continue
            if v is None:
                continue
            all_vals.append(v)
            by_decision.setdefault(r["automatic_decision"], []).append(v)
        per_feature_values[name] = all_vals
        per_feature_by_decision[name] = by_decision
        q = quantiles(all_vals)
        feature_rows.append({
            "feature": name,
            "valid_count": len(all_vals),
            "missing_count": n - len(all_vals),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in q.items()},
        })

    with open(OUT_DIR / "training_threshold_feature_summary.csv", "w", newline="") as f:
        fieldnames = ["feature", "valid_count", "missing_count", "min", "max", "mean", "std",
                      "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in feature_rows:
            w.writerow(row)

    for row in feature_rows:
        print(row["feature"], "valid=", row["valid_count"], "missing=", row["missing_count"],
              "p10=", row.get("p10"), "p50=", row.get("p50"), "p90=", row.get("p90"))

    # boundary bands around each current threshold
    boundary_rows = []

    def count_band(values, lo, hi, label):
        c = sum(1 for v in values if (lo is None or v >= lo) and (hi is None or v < hi))
        boundary_rows.append({"feature": "n/a", "band": label, "count": c})

    occ = per_feature_values["source_occupancy_frames"]
    for k in (4, 5, 6):
        c = sum(1 for v in occ if v == k)
        boundary_rows.append({"feature": "source_occupancy_frames", "band": f"=={k}", "count": c})
    boundary_rows.append({"feature": "source_occupancy_frames", "band": "<4", "count": sum(1 for v in occ if v < 4)})
    boundary_rows.append({"feature": "source_occupancy_frames", "band": ">6", "count": sum(1 for v in occ if v > 6)})

    stable = per_feature_values["target_stable_frames"]
    for k in (4, 5, 6):
        c = sum(1 for v in stable if v == k)
        boundary_rows.append({"feature": "target_stable_frames", "band": f"=={k}", "count": c})
    boundary_rows.append({"feature": "target_stable_frames", "band": "<4", "count": sum(1 for v in stable if v < 4)})
    boundary_rows.append({"feature": "target_stable_frames", "band": ">6", "count": sum(1 for v in stable if v > 6)})

    gop = per_feature_values["gap_order_persistence_frames"]
    for k in (4, 5, 6):
        c = sum(1 for v in gop if v == k)
        boundary_rows.append({"feature": "gap_order_persistence_frames", "band": f"=={k}", "count": c})
    boundary_rows.append({"feature": "gap_order_persistence_frames", "band": "<4", "count": sum(1 for v in gop if v < 4)})
    boundary_rows.append({"feature": "gap_order_persistence_frames", "band": ">6", "count": sum(1 for v in gop if v > 6)})

    prog = per_feature_values["longitudinal_progress_m"]
    prog_bands = [(None, 0.0, "<0"), (0.0, 0.05, "0~0.05"), (0.05, 0.1, "0.05~0.1"),
                  (0.1, 0.2, "0.1~0.2"), (0.2, None, ">=0.2")]
    for lo, hi, label in prog_bands:
        c = sum(1 for v in prog if (lo is None or v >= lo) and (hi is None or v < hi))
        boundary_rows.append({"feature": "longitudinal_progress_m", "band": label, "count": c})

    lat = per_feature_values["lateral_displacement_m"]
    lat_bands = [(None, 0.5, "<0.5"), (0.5, 0.8, "0.5~0.8"), (0.8, 1.0, "0.8~1.0"),
                 (1.0, 1.2, "1.0~1.2"), (1.2, 1.5, "1.2~1.5"), (1.5, None, ">=1.5")]
    for lo, hi, label in lat_bands:
        c = sum(1 for v in lat if (lo is None or v >= lo) and (hi is None or v < hi))
        boundary_rows.append({"feature": "lateral_displacement_m", "band": label, "count": c})

    with open(OUT_DIR / "training_threshold_boundary_bands.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "band", "count"])
        w.writeheader()
        for row in boundary_rows:
            w.writerow(row)

    # decision-split feature summary (for section 7)
    decision_split_rows = []
    for name in feature_getters:
        for decision in ("accept", "review", "reject"):
            vals = per_feature_by_decision[name].get(decision, [])
            q = quantiles(vals)
            decision_split_rows.append({
                "feature": name, "decision": decision, "count": len(vals),
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in q.items()},
            })
    with open(OUT_DIR / "training_feature_by_decision.csv", "w", newline="") as f:
        fieldnames = ["feature", "decision", "count", "min", "max", "mean", "std",
                      "p10", "p25", "p50", "p75", "p90"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in decision_split_rows:
            w.writerow(row)

    # reject-reason-split feature summary
    reason_split_rows = []
    focus_reasons = {
        "no_relevant_vehicle_interaction", "target_not_stable",
        "gap_order_not_persistent", "lane_change", "serial_continuation",
    }
    for name in feature_getters:
        getter = feature_getters[name]
        for reason in focus_reasons:
            vals = []
            for r in records:
                if r["automatic_decision"] == "reject" and r["automatic_reason"] == reason:
                    try:
                        v = getter(r)
                    except (KeyError, TypeError):
                        continue
                    if v is not None:
                        vals.append(v)
            q = quantiles(vals)
            reason_split_rows.append({
                "feature": name, "reject_reason": reason, "count": len(vals),
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in q.items()},
            })
    with open(OUT_DIR / "training_feature_by_reject_reason.csv", "w", newline="") as f:
        fieldnames = ["feature", "reject_reason", "count", "min", "max", "mean", "std",
                      "p10", "p25", "p50", "p75", "p90"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in reason_split_rows:
            w.writerow(row)

    # ================= D. Gap / TTC summary =================
    def gap_ttc_stats(field):
        finite_vals = []
        inf_count = 0
        nan_count = 0
        neg_inf_count = 0
        for r in records:
            for sample in r["gap_timeseries"]:
                v = sample.get(field)
                if v is None:
                    continue
                if isinstance(v, float) and math.isnan(v):
                    nan_count += 1
                elif v == float("inf"):
                    inf_count += 1
                elif v == float("-inf"):
                    neg_inf_count += 1
                else:
                    finite_vals.append(v)
        q = quantiles(finite_vals)
        total = len(finite_vals) + inf_count + nan_count + neg_inf_count
        return {
            "field": field,
            "finite_count": len(finite_vals),
            "inf_count": inf_count,
            "neg_inf_count": neg_inf_count,
            "nan_count": nan_count,
            "finite_ratio": round(len(finite_vals) / total, 4) if total else None,
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in q.items()},
        }

    gap_ttc_rows = [
        gap_ttc_stats("front_gap_m"),
        gap_ttc_stats("rear_gap_m"),
        gap_ttc_stats("front_ttc_s"),
        gap_ttc_stats("rear_ttc_s"),
    ]
    with open(OUT_DIR / "training_gap_ttc_summary.csv", "w", newline="") as f:
        fieldnames = ["field", "finite_count", "inf_count", "neg_inf_count", "nan_count",
                      "finite_ratio", "min", "max", "mean", "std",
                      "p10", "p25", "p50", "p75", "p90", "p95"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in gap_ttc_rows:
            w.writerow(row)
    for row in gap_ttc_rows:
        print(row["field"], "finite=", row["finite_count"], "inf=", row["inf_count"],
              "nan=", row["nan_count"], "neg_inf=", row["neg_inf_count"])

    # ================= E. sensitivity (real classifier re-run) =================
    sensitivity_candidates = {
        "min_source_occupancy_frames": [3, 4, 5, 6, 7],
        "min_target_stable_frames": [3, 4, 5, 6, 7],
        "min_longitudinal_progress_m": [0.0, 0.05, 0.1, 0.2, 0.5],
        "min_cut_in_lateral_displacement_m": [0.5, 0.8, 1.0, 1.2, 1.5],
        "required_gap_order_persistence_frames": [3, 4, 5, 6, 7],
    }

    # Precompute current decisions for "changed vs current" comparison.
    current_decisions = {}
    parsed = []
    for r in records:
        topo = to_topology(r["topology_evidence"])
        inter_default = to_interaction(r["interaction_evidence"])
        parsed.append((r["candidate_id"], topo, r["interaction_evidence"]))
        diag = classify_v2_merge(topo, inter_default, **CURRENT_THRESHOLDS)
        current_decisions[r["candidate_id"]] = (diag.decision.value, diag.reason)

    # sanity: reproducing current thresholds must match stored automatic_decision
    mismatches = 0
    for r in records:
        stored = (r["automatic_decision"], r["automatic_reason"])
        recomputed = current_decisions[r["candidate_id"]]
        if stored != recomputed:
            mismatches += 1
    print(f"classifier reproduction mismatches (should be 0): {mismatches}")
    assert mismatches == 0, "re-run classifier does not match stored decisions at current thresholds"

    sensitivity_rows = []
    for param, values in sensitivity_candidates.items():
        for val in values:
            kwargs = dict(CURRENT_THRESHOLDS)
            gap_persist_override = None
            if param == "required_gap_order_persistence_frames":
                gap_persist_override = val
            else:
                kwargs[param] = val

            counts = {"accept": 0, "reject": 0, "review": 0}
            changed = 0
            for candidate_id, topo, inter_dict in parsed:
                inter = to_interaction(inter_dict, required_gap_order_persistence_frames=gap_persist_override)
                diag = classify_v2_merge(topo, inter, **kwargs)
                counts[diag.decision.value] += 1
                if diag.decision.value != current_decisions[candidate_id][0]:
                    changed += 1
            is_current = val == CURRENT_THRESHOLDS_ALL[param]
            sensitivity_rows.append({
                "parameter": param, "candidate_value": val,
                "accept": counts["accept"], "reject": counts["reject"], "review": counts["review"],
                "changed_vs_current": changed, "is_current": is_current,
            })

    with open(OUT_DIR / "training_threshold_sensitivity.csv", "w", newline="") as f:
        fieldnames = ["parameter", "candidate_value", "accept", "reject", "review",
                      "changed_vs_current", "is_current"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sensitivity_rows:
            w.writerow(row)
    for row in sensitivity_rows:
        print(row)

    # ================= F. manual review sample =================
    build_manual_review_sample(records, per_feature_values)

    print("done")


def build_manual_review_sample(records, per_feature_values):
    import random
    random.seed(1234)

    by_id = {r["candidate_id"]: r for r in records}
    strata_tags = {}

    def tag(cid, stratum):
        strata_tags.setdefault(cid, set()).add(stratum)

    accepts = [r for r in records if r["automatic_decision"] == "accept"]
    reviews = [r for r in records if r["automatic_decision"] == "review"]
    rejects_by_reason = {}
    for r in records:
        if r["automatic_decision"] == "reject":
            rejects_by_reason.setdefault(r["automatic_reason"], []).append(r)

    # A. ACCEPT strata
    random.shuffle(accepts)
    for r in accepts[:15]:
        tag(r["candidate_id"], "accept_random")

    def interaction_strength(r):
        ie = r["interaction_evidence"]
        return len(ie.get("conflict_vehicle_ids") or []) + (1 if ie.get("front_vehicle_id") is not None else 0) + (1 if ie.get("rear_vehicle_id") is not None else 0)

    accepts_sorted = sorted(accepts, key=interaction_strength)
    for r in accepts_sorted[:8]:
        tag(r["candidate_id"], "accept_weak_interaction")
    for r in accepts_sorted[-8:]:
        tag(r["candidate_id"], "accept_strong_interaction")

    # boundary ACCEPT: near thresholds -- capped per band so one dense
    # band (e.g. gap_order_persistence_frames==5/6, a common value, not
    # a rare edge) doesn't blow the whole sample budget by itself.
    def tag_capped(pool, stratum, cap=6):
        subset = list(pool)
        random.shuffle(subset)
        for r in subset[:cap]:
            tag(r["candidate_id"], stratum)

    tag_capped([r for r in accepts if r["interaction_evidence"]["source_occupancy_frames"] in (5, 6)],
               "accept_boundary_source_occupancy")
    tag_capped([r for r in accepts if r["interaction_evidence"]["target_stable_frames"] in (5, 6)],
               "accept_boundary_target_stable")
    tag_capped([r for r in accepts if 0.08 <= r["interaction_evidence"]["longitudinal_progress_m"] <= 0.15],
               "accept_boundary_longitudinal_progress")
    tag_capped([r for r in accepts if r["interaction_evidence"]["gap_order_persistence_frames"] in (5, 6)],
               "accept_boundary_gap_persistence")

    # B. REVIEW -- representative sample (154 total exceeds the human
    # review budget if combined with every other stratum; cap and note
    # the true population size in the report instead).
    random.shuffle(reviews)
    for r in reviews[:40]:
        tag(r["candidate_id"], "review_sample")

    # C. reject reason representative samples
    focus_reasons = [
        "serial_continuation", "no_relevant_vehicle_interaction", "lane_change",
        "no_authoritative_merge_type", "diverge_split", "target_not_stable",
        "gap_order_not_persistent", "cut_in_not_topology_merge",
    ]
    for reason in focus_reasons:
        pool = rejects_by_reason.get(reason, [])
        random.shuffle(pool)
        take = min(6, len(pool))
        for r in pool[:take]:
            tag(r["candidate_id"], f"reject_sample_{reason}")

    # D. boundary rejects near thresholds (capped per band, same reason
    # as the ACCEPT boundary bands above).
    tag_capped(
        [r for r in rejects_by_reason.get("insufficient_source_history", [])
         if r["interaction_evidence"]["source_occupancy_frames"] in (3, 4)],
        "reject_boundary_source_occupancy",
    )
    tag_capped(
        [r for r in rejects_by_reason.get("target_not_stable", [])
         if r["interaction_evidence"]["target_stable_frames"] in (3, 4)],
        "reject_boundary_target_stable",
    )
    tag_capped(
        [r for r in rejects_by_reason.get("no_physical_transition", [])
         if r["interaction_evidence"]["longitudinal_progress_m"] <= 0.12],
        "reject_boundary_longitudinal_progress",
    )
    tag_capped(
        [r for r in rejects_by_reason.get("gap_order_not_persistent", [])
         if r["interaction_evidence"]["gap_order_persistence_frames"] in (3, 4)],
        "reject_boundary_gap_persistence",
    )
    tag_capped(
        [r for r in rejects_by_reason.get("cut_in_not_topology_merge", [])
         if 0.8 <= r["interaction_evidence"]["lateral_displacement_m"] <= 1.2],
        "reject_boundary_lateral_displacement",
    )

    # E. regression
    for r in records:
        if r["candidate_id"].endswith("t57__204_203"):
            tag(r["candidate_id"], "regression_MAN_0013")

    # cap total sample size to a human-reviewable band (~80-150)
    all_ids = list(strata_tags.keys())
    if len(all_ids) > 150:
        # keep review_all + regression + boundary tags fully; trim random/representative fillers
        must_keep = {
            cid for cid, tags in strata_tags.items()
            if any(t.startswith("review_sample") or t.startswith("regression") or "boundary" in t for t in tags)
        }
        fillers = [cid for cid in all_ids if cid not in must_keep]
        random.shuffle(fillers)
        keep_fillers = fillers[: max(0, 150 - len(must_keep))]
        keep = must_keep | set(keep_fillers)
        strata_tags = {cid: tags for cid, tags in strata_tags.items() if cid in keep}

    fieldnames = [
        "candidate_id", "scenario_id", "automatic_decision", "automatic_reason",
        "sampling_stratum", "source_occupancy_frames", "target_stable_frames",
        "longitudinal_progress_m", "lateral_displacement_m",
        "gap_order_persistence_frames", "required_gap_order_persistence_frames",
        "front_vehicle_id", "rear_vehicle_id", "review_png_path",
    ]
    with open(OUT_DIR / "manual_review_sample_training.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for cid, tags in sorted(strata_tags.items()):
            r = by_id[cid]
            ie = r["interaction_evidence"]
            safe_id = cid.replace("/", "_").replace("#", "_")
            png_path = f"outputs/merge_v2_review/training/{r['automatic_decision']}/{safe_id}.png"
            w.writerow({
                "candidate_id": cid,
                "scenario_id": r["scenario_id"],
                "automatic_decision": r["automatic_decision"],
                "automatic_reason": r["automatic_reason"],
                "sampling_stratum": ";".join(sorted(tags)),
                "source_occupancy_frames": ie["source_occupancy_frames"],
                "target_stable_frames": ie["target_stable_frames"],
                "longitudinal_progress_m": ie["longitudinal_progress_m"],
                "lateral_displacement_m": ie["lateral_displacement_m"],
                "gap_order_persistence_frames": ie["gap_order_persistence_frames"],
                "required_gap_order_persistence_frames": ie["required_gap_order_persistence_frames"],
                "front_vehicle_id": ie["front_vehicle_id"],
                "rear_vehicle_id": ie["rear_vehicle_id"],
                "review_png_path": png_path,
            })

    print(f"manual review sample: {len(strata_tags)} candidates")
    decision_breakdown = {}
    for cid in strata_tags:
        d = by_id[cid]["automatic_decision"]
        decision_breakdown[d] = decision_breakdown.get(d, 0) + 1
    print("breakdown:", decision_breakdown)
    man0013_included = any(cid.endswith("t57__204_203") for cid in strata_tags)
    print("MAN_0013 included:", man0013_included)


if __name__ == "__main__":
    main()
