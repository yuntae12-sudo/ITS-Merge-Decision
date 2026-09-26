"""Stage B-1.5 Section A1: deterministic scene-grouped train/validation
split. Verifies zero scene leakage and reproducibility against the
real 168-maneuver / 157-scene pool."""

from src.environment.dataset_split import (
    MANEUVER_TABLE_PATH,
    SPLIT_MANIFEST_PATH,
    build_split,
    load_split_manifest,
    normalize_split_name,
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


# --- split-name normalization (legacy Phase 2 "train" vs final v2
# "training" PPO runtime compatibility fix) ---


def test_normalize_split_name_legacy_train():
    assert normalize_split_name("train") == "train"


def test_normalize_split_name_final_v2_training():
    assert normalize_split_name("training") == "train"


def test_normalize_split_name_validation():
    assert normalize_split_name("validation") == "validation"


def test_normalize_split_name_unknown_label_is_left_unchanged():
    # Unknown split labels must never be silently coerced into "train".
    assert normalize_split_name("tune") == "tune"
    assert normalize_split_name("bogus_split") == "bogus_split"
