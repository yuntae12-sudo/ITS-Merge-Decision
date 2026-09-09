"""Phase 1 CLI: sync manual validation labels and build the final
CONFIRMED_MERGE manifest (Compressed Commit E).

Two responsibilities, run every time this script executes:

1. Label-template sync (safe to rerun): loads
   ``--labels`` (data/manifests/merge_manual_labels.csv) if present,
   adds UNREVIEWED rows for any candidate_id in ``--candidates`` not
   already labeled, keeps (never deletes) labels for candidate_ids no
   longer in the current scan (warns about them), and never overwrites
   an existing label's ``manual_validation``/``manual_note``. Optionally
   applies ``--set-label CANDIDATE_ID LABEL`` to the merged set.

2. Final manifest: inner-joins candidates with labels WHERE
   manual_validation == CONFIRMED_MERGE, writing
   ``--manifest-output`` (data/manifests/merge_manifest.csv). Zero rows
   is a VALID, expected result before manual review has happened.

Example:
    python scripts/build_merge_manifest.py
    python scripts/build_merge_manifest.py --set-label <candidate_id> CONFIRMED_MERGE
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenarios.dataset_builder import read_candidates_csv

DEFAULT_CANDIDATES = "data/manifests/merge_candidates.csv"
DEFAULT_LABELS = "data/manifests/merge_manual_labels.csv"
DEFAULT_MANIFEST_OUTPUT = "data/manifests/merge_manifest.csv"

ALLOWED_LABELS = frozenset(
    {"UNREVIEWED", "CONFIRMED_MERGE", "CONFIRMED_NON_MERGE", "UNCERTAIN"}
)

LABEL_FIELDS = ["candidate_id", "manual_validation", "manual_note"]

MANIFEST_FIELDS = [
    "candidate_id",
    "scene_key",
    "source_dataset",
    "source_split",
    "source_shard",
    "record_index",
    "source_lane_id",
    "target_lane_id",
    "transition_frame",
    "merge_start_frame",
    "merge_complete_frame",
    "merge_start_s",
    "merge_end_s",
    "ego_longitudinal_speed_mps",
    "front_vehicle_id",
    "front_gap_m",
    "front_relative_speed_mps",
    "front_ttc_s",
    "rear_vehicle_id",
    "rear_gap_m",
    "rear_relative_speed_mps",
    "rear_ttc_s",
    "traffic_density",
    "detector_decision",
    "detector_reason",
    "manual_validation",
    "manual_note",
]


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Sync manual merge-candidate validation labels and build "
            "the final CONFIRMED_MERGE manifest."
        )
    )
    parser.add_argument("--candidates", type=str, default=DEFAULT_CANDIDATES)
    parser.add_argument("--labels", type=str, default=DEFAULT_LABELS)
    parser.add_argument(
        "--manifest-output", type=str, default=DEFAULT_MANIFEST_OUTPUT
    )
    parser.add_argument(
        "--set-label", type=str, nargs=2, default=None,
        metavar=("CANDIDATE_ID", "LABEL"),
    )

    return parser.parse_args()


def load_labels(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return {row["candidate_id"]: dict(row) for row in reader}


def write_labels(path: Path, labels: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for candidate_id in sorted(labels.keys()):
            row = labels[candidate_id]
            writer.writerow(
                {
                    "candidate_id": candidate_id,
                    "manual_validation": row["manual_validation"],
                    "manual_note": row.get("manual_note", ""),
                }
            )


def sync_labels(candidate_ids, existing_labels: dict, set_label=None):
    """Applies the label-template sync rules. Returns the merged dict.

    Also returns the list of stale candidate_ids (labeled but no longer
    in the current candidate set) for the caller to warn about.
    """

    merged = dict(existing_labels)
    candidate_id_set = set(candidate_ids)

    for candidate_id in candidate_ids:
        if candidate_id not in merged:
            merged[candidate_id] = {
                "candidate_id": candidate_id,
                "manual_validation": "UNREVIEWED",
                "manual_note": "",
            }

    stale = sorted(set(merged.keys()) - candidate_id_set)

    if set_label is not None:
        set_candidate_id, set_value = set_label
        if set_value not in ALLOWED_LABELS:
            raise ValueError(
                f"--set-label value must be one of {sorted(ALLOWED_LABELS)}, "
                f"got {set_value!r}"
            )
        if set_candidate_id not in merged:
            merged[set_candidate_id] = {
                "candidate_id": set_candidate_id,
                "manual_validation": set_value,
                "manual_note": "",
            }
        else:
            merged[set_candidate_id]["manual_validation"] = set_value

    return merged, stale


def build_manifest(candidate_rows, labels: dict):
    manifest_rows = []
    for row in candidate_rows:
        label = labels.get(row["candidate_id"])
        if label is None or label["manual_validation"] != "CONFIRMED_MERGE":
            continue
        manifest_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "scene_key": row["scene_key"],
                "source_dataset": row["source_dataset"],
                "source_split": row["source_split"],
                "source_shard": row["source_shard"],
                "record_index": row["record_index"],
                "source_lane_id": row["source_lane_id"],
                "target_lane_id": row["target_lane_id"],
                "transition_frame": row["transition_frame"],
                "merge_start_frame": row["merge_start_frame"],
                "merge_complete_frame": row["merge_complete_frame"],
                "merge_start_s": row["merge_start_s"],
                "merge_end_s": row["merge_end_s"],
                "ego_longitudinal_speed_mps": row["ego_longitudinal_speed_mps"],
                "front_vehicle_id": row["front_vehicle_id"],
                "front_gap_m": row["front_gap_m"],
                "front_relative_speed_mps": row["front_relative_speed_mps"],
                "front_ttc_s": row["front_ttc_s"],
                "rear_vehicle_id": row["rear_vehicle_id"],
                "rear_gap_m": row["rear_gap_m"],
                "rear_relative_speed_mps": row["rear_relative_speed_mps"],
                "rear_ttc_s": row["rear_ttc_s"],
                "traffic_density": row["traffic_density"],
                "detector_decision": row["decision"],
                "detector_reason": row["reason"],
                "manual_validation": label["manual_validation"],
                "manual_note": label.get("manual_note", ""),
            }
        )
    return sorted(manifest_rows, key=lambda r: r["candidate_id"])


def write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():

    args = parse_args()

    print("=" * 70)
    print("ITS Merge Decision - Phase 1")
    print("Build Merge Manifest (Compressed Commit E)")
    print("=" * 70)

    candidate_rows = read_candidates_csv(args.candidates)
    candidate_ids = [row["candidate_id"] for row in candidate_rows]

    labels_path = Path(args.labels)
    existing_labels = load_labels(labels_path)

    merged_labels, stale = sync_labels(
        candidate_ids, existing_labels, set_label=args.set_label
    )

    if stale:
        print(
            "\nWARN: previously labeled candidates no longer in current scan:"
        )
        for candidate_id in stale:
            print(f"  {candidate_id}")

    write_labels(labels_path, merged_labels)
    print(f"\nWrote labels CSV     : {labels_path} ({len(merged_labels)} rows)")

    manifest_rows = build_manifest(candidate_rows, merged_labels)
    write_manifest(Path(args.manifest_output), manifest_rows)
    print(
        f"Wrote manifest CSV   : {args.manifest_output} "
        f"({len(manifest_rows)} rows)"
    )

    counts = {}
    for label in merged_labels.values():
        counts[label["manual_validation"]] = counts.get(label["manual_validation"], 0) + 1

    reviewed = sum(v for k, v in counts.items() if k != "UNREVIEWED")

    print("\n" + "=" * 70)
    print("Manual Review Stats")
    print("=" * 70)
    print(f"  total labeled candidates   : {len(merged_labels)}")
    print(f"  reviewed (non-UNREVIEWED)  : {reviewed}")
    print(f"  CONFIRMED_MERGE            : {counts.get('CONFIRMED_MERGE', 0)}")
    print(f"  CONFIRMED_NON_MERGE        : {counts.get('CONFIRMED_NON_MERGE', 0)}")
    print(f"  UNCERTAIN                  : {counts.get('UNCERTAIN', 0)}")
    print(f"  UNREVIEWED                 : {counts.get('UNREVIEWED', 0)}")

    decision_by_id = {row["candidate_id"]: row["decision"] for row in candidate_rows}
    breakdown = {}
    for candidate_id, label in merged_labels.items():
        detector_decision = decision_by_id.get(candidate_id)
        if detector_decision is None:
            continue
        key = f"{detector_decision.upper()} & {label['manual_validation']}"
        breakdown[key] = breakdown.get(key, 0) + 1

    print("\n  Detector-decision x manual-validation breakdown:")
    for key, count in sorted(breakdown.items()):
        print(f"    {key}: {count}")

    if len(manifest_rows) == 0:
        print(
            "\n  0 confirmed manifest rows (expected until manual review occurs)."
        )

    print("\n" + "=" * 70)
    print("BUILD MERGE MANIFEST: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
