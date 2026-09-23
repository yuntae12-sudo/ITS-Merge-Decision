#!/usr/bin/env python3
"""Build the MERGE v2 target scenario_id pool: a HIGH-RECALL pre-filter
over v1's broad transition candidates, computed without any WOMD
Scenario-protobuf topology (tf_example geometry/interaction only).

Why not just use v1's 168+18 confirmed maneuvers: v1's ACCEPT set is
already filtered by v1's own geometric heuristics (see
``src/scenarios/merge_detector.py``), which this whole v2 effort exists
to replace -- e.g. it can both false-positive (MAN_0013-style serial
continuation) and false-negative (an interactive cut-in or a genuine
multi-entry merge v1's convergence heuristic scored too weakly). Using
only the 168+18 as the v2 topology-lookup target would silently bake in
every v1 false negative.

This script instead keeps every v1 REVIEW and ACCEPT candidate (v1's own
"not confidently rejectable" and "confident merge" buckets), plus every
v1 REJECT whose reason is a genuinely judgement-based geometric call
(parallel_lane_change, insufficient_convergence,
insufficient_target_persistence) -- because those are exactly the calls
authoritative topology could overturn (a "parallel lane change" can be a
real topological merge or interactive cut-in; "insufficient convergence"
or "insufficient target persistence" are threshold judgement calls, not
physical impossibilities).

Only v1 REJECT reasons that are physically unambiguous WITHOUT topology
are dropped:

- source_not_near_end: the source lane has not reached its own end at
  this transition point -- the ego cannot physically leave the source
  lane here regardless of what the target lane's topology turns out to
  be.
- target_too_far: the source endpoint's lateral distance to the target
  lane exceeds the max-adjacency threshold -- the lanes are not
  spatially adjacent at the candidate point, so no topology could make
  this a merge.
- heading_mismatch: the source/target headings diverge beyond the
  max-heading threshold at the convergence point -- lanes going in
  substantially different directions cannot be a merge convergence.

These three reasons are pure tf_example-trajectory geometry (no
protobuf topology involved in computing them), so dropping them can
only remove candidates that are impossible independent of topology --
this can never introduce a v2 false negative that authoritative
topology could have rescued.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios.scenario_proto_loader import iter_tfexample_scenario_ids

# Reject reasons that are unambiguous WITHOUT authoritative topology --
# see module docstring for the physical justification for each.
CONFIDENT_GEOMETRIC_REJECT_REASONS = {
    "source_not_near_end",
    "target_too_far",
    "heading_mismatch",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--candidates-csv",
        action="append",
        required=True,
        help="One or more broad-transition candidate CSVs "
        "(e.g. merge_candidates_training_10shard.csv, merge_candidates_6shard.csv). "
        "Repeat for multiple files.",
    )
    p.add_argument(
        "--v1-confirmed-maneuvers-csv",
        action="append",
        default=[],
        help="v1 maneuver-level ACCEPT files (e.g. training_visual_merge_maneuvers.csv) "
        "kept as a separate regression subset -- never used to narrow the v2 "
        "target pool, only to report how much of it the pool covers.",
    )
    p.add_argument("--output", default="data/manifests/v2/target_scenario_ids.csv")
    p.add_argument(
        "--v1-confirmed-output",
        default="data/manifests/v2/v1_confirmed_scenario_ids.csv",
    )
    return p.parse_args(argv)


def _resolve_tfexample_path(source_shard):
    if "training" in source_shard:
        return f"data/womd/training/{source_shard}"
    return f"data/womd/validation/{source_shard}"


def load_candidate_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def high_recall_prefilter(rows):
    """Returns (kept_rows, excluded_rows) -- see module docstring for
    the exact policy. Never drops a REVIEW or ACCEPT row."""

    kept, excluded = [], []
    for row in rows:
        if row["decision"] == "reject" and row["reason"] in CONFIDENT_GEOMETRIC_REJECT_REASONS:
            excluded.append(row)
        else:
            kept.append(row)
    return kept, excluded


def resolve_scenario_ids(shard_records):
    """``shard_records``: {source_shard: {record_index, ...}}. Returns
    {(source_shard, record_index): scenario_id}, one sequential pass per
    shard (never rescans a shard per candidate row)."""

    resolved = {}
    for source_shard, wanted_indices in shard_records.items():
        path = _resolve_tfexample_path(source_shard)
        max_wanted = max(wanted_indices)
        for idx, sid in iter_tfexample_scenario_ids(path):
            if idx in wanted_indices:
                resolved[(source_shard, idx)] = sid
            if idx >= max_wanted:
                break
    return resolved


def build_target_pool(candidate_rows):
    """Aggregates high-recall candidate rows to unique scenario_id,
    preserving every candidate_id that maps to each scenario so the
    pool stays auditable back to v1's own per-transition decisions."""

    by_shard_record = defaultdict(set)
    for row in candidate_rows:
        by_shard_record[row["source_shard"]].add(int(row["record_index"]))

    scenario_id_by_key = resolve_scenario_ids(by_shard_record)

    by_scenario = defaultdict(lambda: {
        "source_split": None, "source_shard": None, "candidate_ids": [], "decisions": set(),
    })
    unresolved = 0
    for row in candidate_rows:
        key = (row["source_shard"], int(row["record_index"]))
        sid = scenario_id_by_key.get(key)
        if sid is None:
            unresolved += 1
            continue
        entry = by_scenario[sid]
        if entry["source_split"] is None:
            entry["source_split"] = row["source_split"]
            entry["source_shard"] = row["source_shard"]
        elif entry["source_split"] != row["source_split"]:
            raise ValueError(
                f"scenario_id {sid} appears under both "
                f"{entry['source_split']} and {row['source_split']}"
            )
        entry["candidate_ids"].append(row["candidate_id"])
        entry["decisions"].add(row["decision"])
    if unresolved:
        print(f"WARNING: {unresolved} candidate row(s) had no resolvable scenario_id")
    return by_scenario


def write_target_pool(path, by_scenario):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario_id", "source_split", "source_shard",
            "candidate_count", "candidate_ids", "v1_decisions_present",
        ])
        for sid, entry in sorted(by_scenario.items()):
            writer.writerow([
                sid, entry["source_split"], entry["source_shard"],
                len(entry["candidate_ids"]), ";".join(entry["candidate_ids"]),
                ";".join(sorted(entry["decisions"])),
            ])


def load_v1_confirmed(paths):
    """v1 maneuver-level ACCEPT rows -- a fixed regression subset, kept
    separate from (never narrowing) the v2 high-recall target pool."""

    maneuvers = []
    for path in paths:
        split = "training" if "training" in Path(path).name or "training" in path else "validation"
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                maneuvers.append({
                    "maneuver_id": row["maneuver_id"],
                    "source_split": split,
                    "source_shard": row["source_shard"],
                    "record_index": int(row["record_index"]),
                })
    return maneuvers


def build_v1_confirmed_pool(maneuvers):
    by_shard_record = defaultdict(set)
    for m in maneuvers:
        by_shard_record[m["source_shard"]].add(m["record_index"])
    scenario_id_by_key = resolve_scenario_ids(by_shard_record)

    by_scenario = defaultdict(lambda: {"source_split": None, "source_shard": None, "maneuver_ids": []})
    for m in maneuvers:
        key = (m["source_shard"], m["record_index"])
        sid = scenario_id_by_key.get(key)
        if sid is None:
            continue
        entry = by_scenario[sid]
        entry["source_split"] = m["source_split"]
        entry["source_shard"] = m["source_shard"]
        entry["maneuver_ids"].append(m["maneuver_id"])
    return by_scenario


def write_v1_confirmed(path, by_scenario):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_id", "source_split", "source_shard", "maneuver_count", "maneuver_ids"])
        for sid, entry in sorted(by_scenario.items()):
            writer.writerow([sid, entry["source_split"], entry["source_shard"], len(entry["maneuver_ids"]), ";".join(entry["maneuver_ids"])])


def main(argv=None):
    args = parse_args(argv)

    candidate_rows = load_candidate_rows(args.candidates_csv)
    print(f"total broad-transition candidates: {len(candidate_rows)}")

    kept, excluded = high_recall_prefilter(candidate_rows)
    print(f"excluded (confident geometric reject, topology-independent): {len(excluded)}")
    from collections import Counter
    print(f"  by reason: {dict(Counter(r['reason'] for r in excluded))}")
    print(f"kept (high-recall pool): {len(kept)}")
    print(f"  by v1 decision: {dict(Counter(r['decision'] for r in kept))}")

    by_scenario = build_target_pool(kept)
    print(f"unique scenario_ids in high-recall pool: {len(by_scenario)}")
    train_n = sum(1 for v in by_scenario.values() if v["source_split"] == "training")
    val_n = sum(1 for v in by_scenario.values() if v["source_split"] == "validation")
    print(f"  training: {train_n}  validation: {val_n}")

    write_target_pool(args.output, by_scenario)
    print(f"wrote {args.output}")

    if args.v1_confirmed_maneuvers_csv:
        v1_maneuvers = load_v1_confirmed(args.v1_confirmed_maneuvers_csv)
        v1_by_scenario = build_v1_confirmed_pool(v1_maneuvers)
        write_v1_confirmed(args.v1_confirmed_output, v1_by_scenario)
        print(f"wrote {args.v1_confirmed_output} ({len(v1_by_scenario)} v1-confirmed scenario_ids)")

        covered = set(v1_by_scenario) & set(by_scenario)
        print(
            f"v1-confirmed coverage inside v2 high-recall pool: "
            f"{len(covered)}/{len(v1_by_scenario)}"
        )
        missing = set(v1_by_scenario) - set(by_scenario)
        if missing:
            print(f"  WARNING: {len(missing)} v1-confirmed scenario_id(s) NOT in the v2 pool: {sorted(missing)[:10]}")


if __name__ == "__main__":
    main()
