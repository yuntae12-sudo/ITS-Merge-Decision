#!/usr/bin/env python3
"""Builds the human-review package for the TRAIN calibration sample.

Reads the existing 136-row ``manual_review_sample_training.csv`` (never
modifies it) plus ``evidence_training.jsonl``, and writes a new
``human_review_training.csv`` with review-priority/batch ordering and
blank human-input columns. No human field is ever auto-filled --
independent verification is the entire point of this step."""

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_CSV = "outputs/merge_v2_calibration/manual_review_sample_training.csv"
EVIDENCE_JSONL = "data/manifests/v2/evidence_training.jsonl"
OUTPUT_CSV = "outputs/merge_v2_calibration/human_review_training.csv"

HUMAN_FIELDS = [
    "human_label", "human_note",
    "source_occupancy_ok", "target_stability_ok", "gap_persistence_ok",
    "topology_merge_ok", "interaction_ok", "reviewer_confidence",
]

FIELDNAMES = [
    "review_batch", "review_order", "review_priority",
    "candidate_id", "scenario_id", "sampling_stratum",
    "automatic_decision", "automatic_reason",
    "source_occupancy_frames", "target_stable_frames",
    "longitudinal_progress_m", "lateral_displacement_m",
    "gap_order_persistence_frames",
    "review_png_path",
    *HUMAN_FIELDS,
]

# Priority is derived from sampling_stratum; a candidate tagged with
# multiple strata (';'-joined) uses only its single highest priority.
STRATUM_TO_PRIORITY = {
    "accept_boundary_source_occupancy": "P1",
    "reject_boundary_source_occupancy": "P1",
    "accept_boundary_target_stable": "P1",
    "reject_boundary_target_stable": "P1",
    "accept_boundary_gap_persistence": "P1",
    "reject_boundary_gap_persistence": "P1",
    "accept_boundary_longitudinal_progress": "P1",
    "reject_boundary_longitudinal_progress": "P1",
    "reject_boundary_lateral_displacement": "P1",
    "review_sample": "P2",
    "reject_sample_target_not_stable": "P3",
    "reject_sample_gap_order_not_persistent": "P3",
    "reject_sample_cut_in_not_topology_merge": "P4",
    "reject_sample_lane_change": "P5",
    "reject_sample_serial_continuation": "P5",
    "reject_sample_no_authoritative_merge_type": "P5",
    "reject_sample_no_relevant_vehicle_interaction": "P5",
    "reject_sample_diverge_split": "P5",
    "accept_random": "P5",
    "accept_weak_interaction": "P5",
    "accept_strong_interaction": "P5",
    "regression_MAN_0013": "P6",
}
PRIORITY_ORDER = ["P1", "P2", "P3", "P4", "P5", "P6"]


def highest_priority(stratum_field):
    strata = stratum_field.split(";")
    priorities = [STRATUM_TO_PRIORITY[s] for s in strata if s in STRATUM_TO_PRIORITY]
    if not priorities:
        raise ValueError(f"no known priority for strata: {strata}")
    # MAN_0013 must always land in its dedicated P6 slot regardless of
    # which other strata it also happens to satisfy.
    if "regression_MAN_0013" in strata:
        return "P6"
    return min(priorities, key=PRIORITY_ORDER.index)


def load_evidence_lookup(path):
    lookup = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lookup[rec["candidate_id"]] = rec
    return lookup


def assign_batches(rows, batch_size=25):
    """Priority-ordered (P1..P6, MAN_0013 forced into an early batch),
    then chunked into fixed-size batches -- gives P1/P2 the earliest
    batches while keeping decision categories mixed within each batch
    rather than segregated purely by priority."""

    def sort_key(row):
        # MAN_0013 (P6 by definition) is pulled into batch 1 explicitly
        # below, so its priority-based position here doesn't matter.
        return (PRIORITY_ORDER.index(row["review_priority"]), row["candidate_id"])

    ordered = sorted(rows, key=sort_key)

    man0013 = [r for r in ordered if r["sampling_stratum"] == "regression_MAN_0013"]
    rest = [r for r in ordered if r["sampling_stratum"] != "regression_MAN_0013"]

    # Interleave decisions within priority bands isn't necessary given
    # small batch counts; simple chunking after priority sort already
    # keeps P1/P2 up front, and reject/accept/review remain naturally
    # mixed since strata are grouped by *purpose* not by decision.
    final_order = man0013 + rest

    for idx, row in enumerate(final_order):
        row["review_order"] = idx + 1
        row["review_batch"] = f"batch_{idx // batch_size + 1:02d}"

    return final_order


def main():
    sample_rows = list(csv.DictReader(open(SAMPLE_CSV, newline="", encoding="utf-8")))
    if len(sample_rows) != 136:
        raise SystemExit(f"expected 136 sample rows, got {len(sample_rows)}")

    candidate_ids = [r["candidate_id"] for r in sample_rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise SystemExit("duplicate candidate_id in manual_review_sample_training.csv")

    evidence_lookup = load_evidence_lookup(EVIDENCE_JSONL)

    missing_evidence = [cid for cid in candidate_ids if cid not in evidence_lookup]
    if missing_evidence:
        raise SystemExit(f"missing evidence for candidates: {missing_evidence}")

    missing_png = [r["candidate_id"] for r in sample_rows if not os.path.exists(r["review_png_path"])]
    if missing_png:
        raise SystemExit(f"missing PNG for candidates: {missing_png}")

    out_rows = []
    for r in sample_rows:
        priority = highest_priority(r["sampling_stratum"])
        out_row = {
            "review_batch": None,  # filled by assign_batches
            "review_order": None,
            "review_priority": priority,
            "candidate_id": r["candidate_id"],
            "scenario_id": r["scenario_id"],
            "sampling_stratum": r["sampling_stratum"],
            "automatic_decision": r["automatic_decision"],
            "automatic_reason": r["automatic_reason"],
            "source_occupancy_frames": r["source_occupancy_frames"],
            "target_stable_frames": r["target_stable_frames"],
            "longitudinal_progress_m": r["longitudinal_progress_m"],
            "lateral_displacement_m": r["lateral_displacement_m"],
            "gap_order_persistence_frames": r["gap_order_persistence_frames"],
            "review_png_path": r["review_png_path"],
        }
        for field in HUMAN_FIELDS:
            out_row[field] = ""
        out_rows.append(out_row)

    out_rows = assign_batches(out_rows)

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print(f"wrote {len(out_rows)} rows to {OUTPUT_CSV}")

    from collections import Counter
    priority_counts = Counter(r["review_priority"] for r in out_rows)
    batch_counts = Counter(r["review_batch"] for r in out_rows)
    decision_counts = Counter(r["automatic_decision"] for r in out_rows)
    print("priority counts:", dict(sorted(priority_counts.items())))
    print("batch counts:", dict(sorted(batch_counts.items())))
    print("decision counts:", dict(decision_counts))


if __name__ == "__main__":
    main()
