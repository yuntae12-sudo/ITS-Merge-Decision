"""Split-manifest loading + split-name normalization shared by the
final MERGE Dataset loader (``src.environment.full_split_evaluator``)
and legacy split-manifest readers.
"""

import csv
import dataclasses
from typing import List

SPLIT_MANIFEST_PATH = "data/manifests/phase2_dataset_split.csv"


@dataclasses.dataclass(frozen=True)
class ManeuverSplitRow:
    maneuver_id: str
    scene_key: str
    split: str


def load_split_manifest(path: str = SPLIT_MANIFEST_PATH) -> List[ManeuverSplitRow]:
    with open(path, newline="") as handle:
        return [
            ManeuverSplitRow(
                maneuver_id=row["maneuver_id"],
                scene_key=row["scene_key"],
                split=row["split"],
            )
            for row in csv.DictReader(handle)
        ]


# Canonical semantic split names produced by this module's own logic
# (legacy Phase 2 manifest) and consumed throughout PPO/evaluator code.
SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"

# Raw manifest label -> canonical semantic split name. The final MERGE
# Decision Dataset v2 canonical split manifest
# (data/manifests/merge_split.csv) uses "training" where
# the legacy Phase 2 manifest (data/manifests/phase2_dataset_split.csv)
# uses "train" -- both mean the same TRAIN split at the PPO/evaluator
# level. Unknown labels are intentionally left untouched (never
# silently coerced into "train").
_SPLIT_NAME_ALIASES = {
    "train": SPLIT_TRAIN,
    "training": SPLIT_TRAIN,
    "validation": SPLIT_VALIDATION,
}


def normalize_split_name(split: str) -> str:
    """Maps a raw split label from any canonical manifest (legacy
    Phase 2 "train"/"validation" or final v2 "training"/"validation")
    to the semantic split name PPO/evaluator code expects. Labels with
    no known mapping are returned unchanged, never coerced."""

    return _SPLIT_NAME_ALIASES.get(split, split)
