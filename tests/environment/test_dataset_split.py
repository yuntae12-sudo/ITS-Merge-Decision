"""Stage B-1.5 Section A1: deterministic scene-grouped train/validation
split. Verifies zero scene leakage and reproducibility against the
real 168-maneuver / 157-scene pool."""

from src.environment.dataset_split import (
    MANEUVER_TABLE_PATH,
    SPLIT_MANIFEST_PATH,
    build_split,
    load_split_manifest,
)


def test_split_covers_all_maneuvers():
    rows = build_split()
    assert len(rows) == 168


def test_split_has_zero_scene_leakage():
    rows = build_split()
    scenes_by_split = {"train": set(), "validation": set()}
    for row in rows:
        scenes_by_split[row.split].add(row.scene_key)

    assert scenes_by_split["train"].isdisjoint(scenes_by_split["validation"])


def test_split_is_deterministic_across_runs():
    first = [(row.maneuver_id, row.split) for row in build_split()]
    second = [(row.maneuver_id, row.split) for row in build_split()]
    assert first == second


def test_split_ratio_is_reasonably_close_to_target():
    rows = build_split()
    train_count = sum(1 for row in rows if row.split == "train")
    train_fraction = train_count / len(rows)
    assert 0.55 <= train_fraction <= 0.85


def test_tracked_manifest_matches_regenerated_split():
    tracked_rows = load_split_manifest(SPLIT_MANIFEST_PATH)
    regenerated_rows = build_split(MANEUVER_TABLE_PATH)
    tracked = [(row.maneuver_id, row.scene_key, row.split) for row in tracked_rows]
    regenerated = [
        (row.maneuver_id, row.scene_key, row.split) for row in regenerated_rows
    ]
    assert tracked == regenerated
