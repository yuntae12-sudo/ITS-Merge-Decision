"""Phase 2 Stage B-1.5 deterministic train/validation split (Section A1).

Splits the 168-maneuver / 157-scene pool by ``scene_key`` (never by
``maneuver_id``) so no scene straddles both splits -- required so that
controller/FSM threshold tuning on TRAIN cannot leak information about
a scene whose OTHER maneuver ends up in VALIDATION.

Deterministic and reproducible: a fixed integer seed drives a stable
hash-based assignment over the sorted list of unique scene_keys, so
re-running this on the same maneuver table always yields the same
split membership.
"""

import csv
import dataclasses
import hashlib
from pathlib import Path
from typing import List

SPLIT_SEED = 20260911
TRAIN_FRACTION = 0.7

MANEUVER_TABLE_PATH = (
    "outputs/phase1/training_10shard_pilot/training_visual_merge_maneuvers.csv"
)
SPLIT_MANIFEST_PATH = "data/manifests/phase2_dataset_split.csv"


@dataclasses.dataclass(frozen=True)
class ManeuverSplitRow:
    maneuver_id: str
    scene_key: str
    split: str


def _scene_split_fraction(scene_key: str, seed: int) -> float:
    """Deterministic pseudo-random fraction in [0, 1) for one scene_key,
    stable across runs/machines (stdlib hashlib, not Python's salted
    built-in hash())."""

    digest = hashlib.sha256(f"{seed}:{scene_key}".encode("utf-8")).digest()
    as_int = int.from_bytes(digest[:8], byteorder="big")
    return as_int / 2 ** 64


def build_split(
    maneuver_table_path: str = MANEUVER_TABLE_PATH,
    seed: int = SPLIT_SEED,
    train_fraction: float = TRAIN_FRACTION,
) -> List[ManeuverSplitRow]:
    with open(maneuver_table_path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    scene_keys = sorted({row["scene_key"] for row in rows})
    scene_split = {
        scene_key: (
            "train"
            if _scene_split_fraction(scene_key, seed) < train_fraction
            else "validation"
        )
        for scene_key in scene_keys
    }

    return [
        ManeuverSplitRow(
            maneuver_id=row["maneuver_id"],
            scene_key=row["scene_key"],
            split=scene_split[row["scene_key"]],
        )
        for row in rows
    ]


def write_split_manifest(
    split_rows: List[ManeuverSplitRow], output_path: str = SPLIT_MANIFEST_PATH
) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["maneuver_id", "scene_key", "split"])
        for row in split_rows:
            writer.writerow([row.maneuver_id, row.scene_key, row.split])


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
