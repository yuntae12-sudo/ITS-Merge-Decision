#!/usr/bin/env python3
"""PHASE B: Merge-Context Eligibility classification for TRAIN MERGE v2.

Separates two independent questions that classify_v2_merge's automatic_
decision conflates into one ACCEPT/REJECT/REVIEW label:

  (1) "Is this candidate's source/target lane pair a real merge-context
      topology?" (map-level question, decided BEFORE any interaction
      evidence is read in classify_v2_merge's fixed gate order)
  (2) "Did the logged ego trajectory stably complete the merge?" (an
      interaction/completion-level question -- irrelevant to whether a
      KEEP/FOLLOW/MERGE *decision* was posed to the vehicle)

For PPO research, (2) failing does NOT mean (1) is false: a scene where
topology confirms a real converging merge lane pair, but the logged ego
took too long to stabilize in the target lane, is still exactly the kind
of scene where KEEP/FOLLOW/MERGE have state-dependent value -- the ego
just didn't (or hadn't yet) committed by the time the log ends.

Rule (grounded directly in classify_v2_merge's actual gate order,
src/scenarios/merge_v2.py:175-247 -- not a name-based guess):

  INELIGIBLE (topology-level: fires before ANY interaction evidence is
  read -- these gates reject the map-level source/target relationship
  itself, not the completion of a specific trajectory):
    - serial_continuation      (single-path map-segment continuation, not
                                 a converging merge at all)
    - map_artifact              (degenerate/undriveable target polyline)
    - diverge_split             (a fork, not a merge)
    - lane_change                (topology never shows a converging merge;
                                 ego just changed lanes)
    - intersection_transition   (an intersection maneuver, not a lane merge)
    - cut_in_not_topology_merge  (lateral cut-in with no map-topology merge)
    - no_authoritative_merge_type (map never connects source->target at all)

  ELIGIBLE (topology-level maneuver_type WAS assigned -- meaning
  source_exits_to_target and (roundabout OR multiple target entries) --
  before these interaction-only gates fired; the failure is about this
  one logged trajectory's completion/interaction, not the lane pair):
    - accept (None reason)       -- passed every gate
    - insufficient_source_history -- real merge topology, but ego's source-
                                     lane history in this log is too short
    - no_physical_transition     -- real merge topology, ego barely moved
                                     -- decision context exists (ego RIGHT
                                     BEFORE a real merge point), completion
                                     didn't happen in-window
    - target_not_stable          -- real merge topology, target-lane
                                     dwell was too short/unstable in-log
    - gap_order_not_persistent   -- real merge topology AND a real
                                     front/rear interaction existed; only
                                     the persistence gate failed

  UNCERTAIN (genuine ambiguity this audit cannot resolve without human
  review or further topology work):
    - ambiguous_topology_requires_manual_review (REVIEW decision -- the
      classifier itself could not confidently assign a maneuver_type)
    - no_relevant_vehicle_interaction -- real merge topology, but zero
      front/rear/conflict vehicle -- a genuine merge OPPORTUNITY exists
      (KEEP vs MERGE could still differ), but there is no interaction
      pressure driving the decision, so whether this is a "decision
      context" is a judgment call, not settled by evidence alone.

Diagnostic-only classification -- this script does NOT change
automatic_decision, calibration_status, or any threshold in
configs/merge_v2.yaml.
"""

import csv
import json
from pathlib import Path

TRAIN_EVIDENCE = "data/manifests/v2/evidence_training.jsonl"
OUT_DIR = Path("outputs/merge_v2_decision_audit_v2")

INELIGIBLE_REASONS = {
    "serial_continuation",
    "map_artifact",
    "diverge_split",
    "lane_change",
    "intersection_transition",
    "cut_in_not_topology_merge",
    "no_authoritative_merge_type",
}

ELIGIBLE_REASONS = {
    None,  # accept
    "insufficient_source_history",
    "no_physical_transition",
    "target_not_stable",
    "gap_order_not_persistent",
}

UNCERTAIN_REASONS = {
    "ambiguous_topology_requires_manual_review",
    "no_relevant_vehicle_interaction",
}


def classify_eligibility(decision, reason):
    if reason in INELIGIBLE_REASONS:
        return "MERGE_CONTEXT_INELIGIBLE", f"topology-level reject: {reason}"
    if reason in ELIGIBLE_REASONS:
        if decision == "accept":
            return "MERGE_CONTEXT_ELIGIBLE", "accept: passed all gates"
        return "MERGE_CONTEXT_ELIGIBLE", f"topology confirmed merge maneuver; interaction/completion-level reject: {reason}"
    if reason in UNCERTAIN_REASONS:
        return "MERGE_CONTEXT_UNCERTAIN", f"genuine ambiguity: {reason}"
    raise ValueError(f"unrecognized (decision={decision!r}, reason={reason!r}) -- update the classification rule")


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(TRAIN_EVIDENCE)
    n = len(records)
    assert n == 5545, f"expected 5545, got {n}"

    rows = []
    for rec in records:
        te = rec["topology_evidence"]
        ie = rec["interaction_evidence"]
        decision = rec["automatic_decision"]
        reason = rec["automatic_reason"]
        status, eligibility_reason = classify_eligibility(decision, reason)

        # topology_valid: this candidate's source/target lane pair was
        # confirmed (or not yet ruled out) as a real converging-merge
        # topology by classify_v2_merge's OWN gates -- i.e. it did NOT hit
        # any of the topology-level INELIGIBLE gates.
        topology_valid = status != "MERGE_CONTEXT_INELIGIBLE"

        # merge_zone_relevant: a front and/or rear vehicle was found in the
        # target lane at some point (the only stored evidence of "a merge
        # zone with traffic context" -- lack of it is exactly the
        # no_relevant_vehicle_interaction UNCERTAIN case above).
        merge_zone_relevant = (
            ie["front_vehicle_id"] is not None or ie["rear_vehicle_id"] is not None
            or bool(ie["conflict_vehicle_ids"])
        )

        # decision_window_valid: a non-degenerate decision window existed
        # in the log (commit_frame strictly after decision_start_frame) --
        # i.e. there was actually time in which a KEEP/FOLLOW/MERGE choice
        # could have been made, regardless of whether ego ultimately
        # completed the merge.
        decision_window_valid = ie["commit_frame"] > ie["decision_start_frame"]

        rows.append({
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

    ids = [r["candidate_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate candidate_id"

    out_csv = OUT_DIR.parent.parent / "data/manifests/v2/training_merge_context_eligibility.csv"
    out_csv = Path("data/manifests/v2/training_merge_context_eligibility.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {out_csv} ({len(rows)} rows)")

    from collections import Counter
    status_counts = Counter(r["merge_context_status"] for r in rows)
    for k in ("MERGE_CONTEXT_ELIGIBLE", "MERGE_CONTEXT_INELIGIBLE", "MERGE_CONTEXT_UNCERTAIN"):
        c = status_counts.get(k, 0)
        print(f"  {k}: {c} ({c/n:.2%})")

    # breakdown by reason within each status, for the report's cross-table
    reason_status = Counter((r["automatic_decision"], r["automatic_reason"], r["merge_context_status"]) for r in rows)
    print("\nby (decision, reason) -> status:")
    for k, v in sorted(reason_status.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")

    return rows


if __name__ == "__main__":
    main()
