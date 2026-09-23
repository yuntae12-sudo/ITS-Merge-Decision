#!/usr/bin/env python3
"""Locate which WOMD Scenario-protobuf shard holds each target scenario_id.

Reads ``data/manifests/v2/target_scenario_ids.csv`` (the MERGE v2 candidate
scenario_ids -- see ``docs/MERGE_DATASET_V2.md``) and streams remote
Scenario-protobuf shards (never downloading them to local disk) looking
ONLY for those ids. It never stores full scenario/topology data -- only
the (scenario_id, shard location) pointer needed to later re-fetch that
one shard for real evidence extraction.

Resume: each shard's outcome (DONE/FAILED) is appended to a progress
JSONL file the moment that shard finishes. Re-running this script skips
every shard already marked DONE for the same (split, target-set) run --
safe across VSCode restarts, SSH drops, or a killed process. A partially
scanned shard (interrupted mid-shard) is simply not in the progress file
yet and is rescanned from record 0 next run.

Early stop: once every target scenario_id for a split has been found,
remaining shards for that split are not scanned.
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios.scenario_proto_loader import iter_scenario_protobufs

BUCKET_ROOT = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario"
SPLIT_TOTAL_SHARDS = {"training": 1000, "validation": 150}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-scenario-ids", default="data/manifests/v2/target_scenario_ids.csv")
    p.add_argument("--split", choices=("training", "validation"), required=True)
    p.add_argument("--start-shard", type=int, default=0)
    p.add_argument("--end-shard", type=int, default=None, help="Exclusive upper bound.")
    p.add_argument(
        "--locator-output",
        default="data/manifests/v2/womd_scenario_proto_locator.csv",
    )
    p.add_argument(
        "--progress-file",
        default="data/manifests/v2/womd_proto_scan_progress.jsonl",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of shards scanned concurrently (each worker owns whole "
        "shards; result merge happens serially in the parent process).",
    )
    p.add_argument(
        "--local-shard-paths",
        nargs="*",
        default=None,
        help="Debug/test mode: scan these LOCAL shard file paths (in shard-"
        "index order starting at --start-shard) instead of streaming from "
        "GCS. Used for local pilot verification against shard 0.",
    )
    return p.parse_args(argv)


def load_targets(path, split):
    """Returns {scenario_id: {"source_split", "source_shard", "label"}}
    restricted to ``split`` -- training/validation targets are tracked and
    scanned completely independently, never mixed into one remaining-set.
    ``label`` is whichever identifying column the target CSV provides
    (``candidate_ids`` for the high-recall v2 pool, ``maneuver_ids`` for
    the legacy v1-confirmed-only pool) so the locator works against
    either schema without change."""

    targets = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        label_field = "candidate_ids" if "candidate_ids" in (reader.fieldnames or []) else "maneuver_ids"
        for row in reader:
            if row["source_split"] != split:
                continue
            targets[row["scenario_id"]] = {
                "source_split": row["source_split"],
                "source_shard": row["source_shard"],
                "label": row[label_field],
            }
    return targets


def load_done_shards(progress_file, split):
    """Shards already marked DONE for this split in a prior run -- these
    are never rescanned. A FAILED or absent entry means "scan it"."""

    done = set()
    path = Path(progress_file)
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["split"] == split and record["status"] == "DONE":
                done.add(record["shard_index"])
    return done


def load_existing_matches(locator_output, split):
    """Matches already recorded for this split in a prior run -- restores
    both the found-scenario_id set (for early stop) and the rows to keep
    when the locator CSV is rewritten this run."""

    matches = {}
    path = Path(locator_output)
    if not path.exists():
        return matches
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["proto_split"] == split:
                matches[row["scenario_id"]] = row
    return matches


def append_progress(progress_file, record):
    """Appends one shard's outcome. Append-only + one JSON object per
    line: a process killed mid-write leaves at most one truncated final
    line, which ``load_done_shards`` skips via ``json.loads`` raising on
    the next read (never corrupts earlier, already-flushed lines)."""

    path = Path(progress_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def write_locator_csv(path, matches):
    fieldnames = [
        "scenario_id", "proto_split", "proto_shard_index", "proto_total_shards",
        "record_index", "source_object", "matched_candidate_label",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(matches.values(), key=lambda r: (r["proto_split"], r["scenario_id"])):
            writer.writerow({k: row[k] for k in fieldnames})


def _remote_path(split, shard_index, total_shards):
    prefix = "training" if split == "training" else "validation"
    return f"{BUCKET_ROOT}/{split}/{prefix}.tfrecord-{shard_index:05d}-of-{total_shards:05d}"


def scan_one_shard(shard_path, shard_index, total_shards, split, remaining_ids, targets):
    """Streams one shard record-by-record: parse -> check scenario_id ->
    record if wanted -> release (scenario object is never accumulated;
    only ``(scenario_id, record_index)`` pairs for matches are kept)."""

    record_count = 0
    shard_matches = {}
    for record_index, scenario in enumerate(iter_scenario_protobufs([shard_path])):
        record_count += 1
        scenario_id = str(scenario.scenario_id)
        if scenario_id in remaining_ids:
            shard_matches[scenario_id] = {
                "scenario_id": scenario_id,
                "proto_split": split,
                "proto_shard_index": shard_index,
                "proto_total_shards": total_shards,
                "record_index": record_index,
                "source_object": shard_path,
                "matched_candidate_label": targets[scenario_id]["label"],
            }
    return record_count, shard_matches


def main(argv=None):
    args = parse_args(argv)
    split = args.split
    total_shards = SPLIT_TOTAL_SHARDS[split]

    targets = load_targets(args.target_scenario_ids, split)
    targets_total = len(targets)
    print(f"[{split}] targets loaded: {targets_total}")

    existing_matches = load_existing_matches(args.locator_output, split)
    done_shards = load_done_shards(args.progress_file, split)
    remaining_ids = set(targets) - set(existing_matches)

    print(
        f"[{split}] resume state: {len(existing_matches)} already matched, "
        f"{len(done_shards)} shard(s) already DONE, "
        f"{len(remaining_ids)} target(s) remaining"
    )

    if not remaining_ids:
        print(f"[{split}] TARGETS_TOTAL={targets_total} TARGETS_FOUND={len(existing_matches)} "
              f"TARGETS_REMAINING=0 -- nothing to scan, all targets already located.")
        return 0

    all_matches = dict(existing_matches)

    if args.local_shard_paths:
        shard_iter = list(enumerate(args.local_shard_paths, start=args.start_shard))
    else:
        end_shard = args.end_shard if args.end_shard is not None else total_shards
        shard_iter = [
            (i, _remote_path(split, i, total_shards))
            for i in range(args.start_shard, end_shard)
            if i not in done_shards
        ]

    def _scan_and_time(shard_index, shard_path, remaining_snapshot):
        """Runs in a worker thread. Reads a snapshot of ``remaining_ids``
        taken when the shard was dispatched -- never the live set -- so
        concurrent workers never race on it; matching/bookkeeping
        against the live set happens serially in the parent below."""

        shard_t0 = time.time()
        try:
            record_count, shard_matches = scan_one_shard(
                shard_path, shard_index, total_shards, split, remaining_snapshot, targets
            )
            status, error = "DONE", None
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            record_count, shard_matches, status, error = 0, {}, "FAILED", repr(exc)
        return shard_index, shard_path, record_count, shard_matches, status, error, time.time() - shard_t0

    t0 = time.time()
    shard_queue = [(i, p) for i, p in shard_iter if i not in done_shards]

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        pending = {}
        queue_iter = iter(shard_queue)

        def _dispatch_next():
            try:
                shard_index, shard_path = next(queue_iter)
            except StopIteration:
                return False
            future = pool.submit(_scan_and_time, shard_index, shard_path, frozenset(remaining_ids))
            pending[future] = shard_index
            return True

        # Keep at most `workers` shards in flight at once.
        for _ in range(max(1, args.workers)):
            if not _dispatch_next():
                break

        while pending:
            if not remaining_ids:
                print(f"[{split}] EARLY STOP: all {targets_total} targets found; "
                      f"{len(pending)} in-flight shard(s) will still finish, no new shards dispatched.")
                for future in pending:
                    future.result()  # let in-flight scans finish; results still recorded below.
                # Fall through to process whatever completed.

            done_future = next(as_completed(list(pending)))
            shard_index = pending.pop(done_future)
            _, shard_path, record_count, shard_matches, status, error, elapsed = done_future.result()

            # Matches for scenario ids some other in-flight/earlier shard
            # already resolved are dropped here (parent is the single
            # writer) -- this is what keeps a duplicate scenario_id from
            # ever being recorded twice even with overlapping workers.
            shard_matches = {sid: v for sid, v in shard_matches.items() if sid in remaining_ids}
            all_matches.update(shard_matches)
            remaining_ids -= set(shard_matches)

            append_progress(args.progress_file, {
                "split": split,
                "shard_index": shard_index,
                "status": status,
                "record_count": record_count,
                "matched_count": len(shard_matches),
                "elapsed_sec": round(elapsed, 2),
                "timestamp": time.time(),
                "error": error,
            })
            if shard_matches:
                write_locator_csv(args.locator_output, all_matches)

            wall = time.time() - t0
            print(
                f"[{split}] shard {shard_index}: status={status} "
                f"records={record_count} +matched={len(shard_matches)} "
                f"elapsed={elapsed:.1f}s | "
                f"targets found: {targets_total - len(remaining_ids)}/{targets_total} "
                f"remaining: {len(remaining_ids)} | wall={wall:.1f}s"
            )

            if remaining_ids:
                _dispatch_next()

    write_locator_csv(args.locator_output, all_matches)
    print(
        f"[{split}] DONE: TARGETS_TOTAL={targets_total} "
        f"TARGETS_FOUND={targets_total - len(remaining_ids)} "
        f"TARGETS_REMAINING={len(remaining_ids)}"
    )
    if remaining_ids:
        print(f"[{split}] still missing (first 10): {sorted(remaining_ids)[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
