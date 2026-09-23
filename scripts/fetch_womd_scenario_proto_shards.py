#!/usr/bin/env python3
"""Safely fetch specific WOMD Scenario-protobuf shards from GCS.

Guards against the failure mode that previously exhausted local disk: an
open-ended ``gcloud storage cp -r .../training/*`` (or an unbounded ``-I``
shard list) that downloads the entire ~1000-shard training split (~455 GB)
or ~150-shard validation split (~41 GB) instead of the handful of shards a
pilot or extraction run actually needs.

This script NEVER constructs a wildcard/recursive remote path and NEVER
downloads a shard that is not explicitly named. There is no ``--all``
flag and none will be added.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

BUCKET_ROOT = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario"
MAX_SHARDS_PER_RUN = 20  # Raise deliberately, never silently.
APPROX_BYTES_PER_TRAINING_SHARD = 455_000_000_000 // 1000  # ~455 MB
APPROX_BYTES_PER_VALIDATION_SHARD = 41_000_000_000 // 150  # ~273 MB
SAFETY_MARGIN_BYTES = 20_000_000_000  # 20 GB headroom left unused, always.


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("training", "validation"), required=True)
    p.add_argument(
        "--shards",
        required=True,
        help="Comma-separated explicit shard indices, e.g. '0,46,49'. "
        "No wildcard/range/--all form is accepted.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to data/womd/scenario_proto/<split>/ to match this "
        "repo's existing pilot-shard layout.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (shard list, expected bytes, free space) and exit "
        "without downloading anything.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt (still requires "
        "--shards to be explicit; never implies --all).",
    )
    return p.parse_args(argv)


def _shard_total(split):
    return 1000 if split == "training" else 150


def _bytes_per_shard(split):
    return (
        APPROX_BYTES_PER_TRAINING_SHARD
        if split == "training"
        else APPROX_BYTES_PER_VALIDATION_SHARD
    )


def _parse_shard_list(raw, total_shards):
    if raw.strip() in ("*", "all", "ALL"):
        raise ValueError(
            "Wildcard/--all shard selection is permanently disabled by this "
            "script. Name every shard index explicitly, e.g. --shards 0,46,49."
        )
    try:
        indices = sorted({int(x) for x in raw.split(",") if x.strip() != ""})
    except ValueError as exc:
        raise ValueError(f"--shards must be a comma-separated integer list: {exc}") from exc
    if not indices:
        raise ValueError("--shards resolved to an empty list")
    out_of_range = [i for i in indices if i < 0 or i >= total_shards]
    if out_of_range:
        raise ValueError(
            f"shard indices out of range for this split (0..{total_shards - 1}): "
            f"{out_of_range}"
        )
    return indices


def _remote_path(split, shard_index, total_shards):
    prefix = "training" if split == "training" else "validation"
    return (
        f"{BUCKET_ROOT}/{split}/{prefix}.tfrecord-{shard_index:05d}-of-{total_shards:05d}"
    )


def main(argv=None):
    args = parse_args(argv)
    total_shards = _shard_total(args.split)
    indices = _parse_shard_list(args.shards, total_shards)

    if len(indices) > MAX_SHARDS_PER_RUN:
        raise ValueError(
            f"Refusing to fetch {len(indices)} shards in one run "
            f"(MAX_SHARDS_PER_RUN={MAX_SHARDS_PER_RUN}). Split into smaller "
            "batches -- this guard exists specifically to prevent an "
            "accidental full-split download."
        )

    output_dir = Path(args.output_dir or f"data/womd/scenario_proto/{args.split}")
    remote_paths = [_remote_path(args.split, i, total_shards) for i in indices]
    expected_bytes = len(indices) * _bytes_per_shard(args.split)

    free_bytes = shutil.disk_usage(output_dir.parent if output_dir.exists() else Path(".")).free

    print("=== WOMD Scenario protobuf fetch plan ===")
    print(f"split:              {args.split}")
    print(f"shard indices:      {indices}")
    print(f"shard count:        {len(indices)} (max allowed per run: {MAX_SHARDS_PER_RUN})")
    print(f"expected size:      ~{expected_bytes / 1e9:.2f} GB")
    print(f"output dir:         {output_dir}")
    print(f"current free space: {free_bytes / 1e9:.2f} GB")
    print(f"safety margin:      {SAFETY_MARGIN_BYTES / 1e9:.2f} GB (always reserved)")
    print(f"projected free:     {(free_bytes - expected_bytes - SAFETY_MARGIN_BYTES) / 1e9:.2f} GB")
    for p in remote_paths:
        print(f"  remote: {p}")

    if free_bytes - expected_bytes < SAFETY_MARGIN_BYTES:
        print(
            "\nABORT: projected free space after download would fall below "
            "the safety margin. Not downloading anything.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print("\n--dry-run: no files downloaded.")
        return 0

    if not args.yes:
        reply = input(f"\nProceed with downloading {len(indices)} shard(s)? [y/N] ")
        if reply.strip().lower() != "y":
            print("Aborted by user.")
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    for remote_path, shard_index in zip(remote_paths, indices):
        local_path = output_dir / Path(remote_path).name
        print(f"Fetching shard {shard_index} -> {local_path}")
        subprocess.run(
            ["gcloud", "storage", "cp", remote_path, str(local_path)], check=True
        )

    print(f"\nDone: {len(indices)} shard(s) fetched to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
