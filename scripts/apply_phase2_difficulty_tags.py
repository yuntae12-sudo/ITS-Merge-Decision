"""Phase 2 Stage C-1: applies the frozen TRAIN-derived interaction-tag
thresholds (configs/phase2_difficulty.yaml) to every maneuver in
data/manifests/phase2_dataset_difficulty.csv, in place.

The thresholds themselves are NEVER refit here -- this script only
reads configs/phase2_difficulty.yaml (already frozen from a one-time
TRAIN-only analysis) and applies the same deterministic rule to both
TRAIN and VALIDATION rows identically. Running this script twice on
the same descriptor CSV produces byte-identical tag columns
(determinism check -- see tests/environment/test_dataset_difficulty.py).
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.environment.observation_builder import TTC_CAP_S

DESCRIPTOR_PATH = "data/manifests/phase2_dataset_difficulty.csv"
CONFIG_PATH = "configs/phase2_difficulty.yaml"

TAG_NAMES = (
    "OPEN_GAP",
    "FRONT_CONSTRAINED",
    "REAR_CONSTRAINED",
    "SOURCE_FOLLOW_RELEVANT",
    "LATE_MERGE",
    "DENSE_INTERACTION",
    "MULTI_CONSTRAINT",
)


def _present(row, slot):
    return row[f"{slot}_present"] == "1"


def compute_tags(row, config):
    thresholds = {
        name: config["interaction_tags"][name].get("thresholds", {})
        for name in TAG_NAMES
    }

    open_gap = (
        not _present(row, "target_front")
        and not _present(row, "target_rear")
        and not _present(row, "source_front")
    )

    front_constrained = False
    if _present(row, "target_front"):
        gap = float(row["target_front_gap"])
        ttc = float(row["target_front_ttc"])
        gap_thresh = thresholds["FRONT_CONSTRAINED"]["target_front_gap_p25_present"]
        ttc_thresh = thresholds["FRONT_CONSTRAINED"]["target_front_ttc_p25_noncapped"]
        front_constrained = gap <= gap_thresh or (ttc < TTC_CAP_S and ttc <= ttc_thresh)

    rear_constrained = _present(row, "target_rear")

    source_follow_relevant = False
    if _present(row, "source_front"):
        gap = float(row["source_front_gap"])
        gap_thresh = thresholds["SOURCE_FOLLOW_RELEVANT"]["source_front_gap_median_present"]
        source_follow_relevant = gap <= gap_thresh

    d_m_thresh = thresholds["LATE_MERGE"]["d_m_p25_all_train"]
    late_merge = float(row["d_m"]) <= d_m_thresh

    density_thresh = thresholds["DENSE_INTERACTION"]["traffic_density_p75_all_train"]
    dense_interaction = float(row["traffic_density"]) >= density_thresh

    component_count = sum(
        [
            front_constrained,
            rear_constrained,
            source_follow_relevant,
            late_merge,
            dense_interaction,
        ]
    )
    multi_constraint = component_count >= 2

    return {
        "OPEN_GAP": int(open_gap),
        "FRONT_CONSTRAINED": int(front_constrained),
        "REAR_CONSTRAINED": int(rear_constrained),
        "SOURCE_FOLLOW_RELEVANT": int(source_follow_relevant),
        "LATE_MERGE": int(late_merge),
        "DENSE_INTERACTION": int(dense_interaction),
        "MULTI_CONSTRAINT": int(multi_constraint),
    }


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(DESCRIPTOR_PATH, newline="") as f:
        reader = csv.DictReader(f)
        base_fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # Strip any previously-applied tag columns before recomputing, so
    # rerunning this script is idempotent rather than accumulating
    # stale duplicate columns.
    base_fieldnames = [name for name in base_fieldnames if name not in TAG_NAMES]

    for row in rows:
        tags = compute_tags(row, config)
        row.update(tags)

    fieldnames = base_fieldnames + list(TAG_NAMES)

    with open(DESCRIPTOR_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})

    print(f"Tagged {len(rows)} rows in {DESCRIPTOR_PATH}")

    for split_name in ("train", "validation"):
        split_rows = [r for r in rows if r["split"] == split_name]
        n = len(split_rows)
        print(f"\n=== {split_name.upper()} (n={n}) ===")
        for tag in TAG_NAMES:
            count = sum(int(r[tag]) for r in split_rows)
            print(f"  {tag}: {count} ({100*count/n:.0f}%)")


if __name__ == "__main__":
    main()
