"""Tests for src/scenarios/scenario_loader.py's multi-shard dataset
expansion support (Commit: "support multi-shard merge dataset
expansion").

No real WOMD data required for the loader-level tests (E/F/G/H/N):
``build_waymax_config`` is tested directly against synthetic
expansion configs, and shard_glob/shards resolution is tested against
temp files. Scene/candidate identity tests (A/B) use lightweight
synthetic ScenarioRecord-like objects, matching the existing style in
test_dataset_builder.py/test_build_merge_manifest.py.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from waymax import config as waymax_config

from src.scenarios.dataset_builder import make_candidate_id
from src.scenarios.scenario_loader import (
    DatasetExpansionConfig,
    build_waymax_config,
    load_dataset_config,
    resolve_physical_shard,
    select_single_shard_for_inspection,
)


# ---------------------------------------------------------------------
# Case E/F: split selects the correct Waymax base config.
# ---------------------------------------------------------------------


def test_case_e_training_split_selects_training_base(tmp_path):
    fake_shard = tmp_path / "training_tfexample.tfrecord-00000-of-01000"
    fake_shard.write_text("fake")

    expansion_config = DatasetExpansionConfig(
        dataset_name="WOMD",
        split="training",
        shard_paths=[str(fake_shard)],
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )

    built = build_waymax_config(expansion_config, str(fake_shard))

    # Every non-overridden field must match WOD_1_3_1_TRAINING exactly
    # (identity check against the training base, not validation).
    assert built.num_paths == waymax_config.WOD_1_3_1_TRAINING.num_paths
    assert (
        built.num_points_per_path
        == waymax_config.WOD_1_3_1_TRAINING.num_points_per_path
    )
    assert built.data_format == waymax_config.WOD_1_3_1_TRAINING.data_format
    assert built.path == str(fake_shard)
    assert built.max_num_objects == 64
    assert built.repeat == 1

    # Must NOT match the validation base's overridden fields where they
    # differ (sanity that we didn't accidentally pick validation).
    assert built.path != waymax_config.WOD_1_3_1_VALIDATION.path


def test_case_f_validation_split_selects_validation_base(tmp_path):
    fake_shard = tmp_path / "validation_tfexample.tfrecord-00000-of-00150"
    fake_shard.write_text("fake")

    expansion_config = DatasetExpansionConfig(
        dataset_name="WOMD",
        split="validation",
        shard_paths=[str(fake_shard)],
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )

    built = build_waymax_config(expansion_config, str(fake_shard))

    assert built.num_paths == waymax_config.WOD_1_3_1_VALIDATION.num_paths
    assert (
        built.num_points_per_path
        == waymax_config.WOD_1_3_1_VALIDATION.num_points_per_path
    )
    assert built.data_format == waymax_config.WOD_1_3_1_VALIDATION.data_format
    assert built.path == str(fake_shard)


def test_unsupported_split_rejected_loudly(tmp_path):
    fake_shard = tmp_path / "shard.tfrecord-00000"
    fake_shard.write_text("fake")

    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "dataset_name": "WOMD",
                "split": "testing",  # not supported by this loader
                "shards": [str(fake_shard)],
                "max_num_objects": 64,
            }
        )
    )

    with pytest.raises(ValueError, match="unsupported split"):
        load_dataset_config(str(config_path))


# ---------------------------------------------------------------------
# Case G: zero-match shard glob fails loudly.
# ---------------------------------------------------------------------


def test_case_g_zero_match_shard_glob_fails_loudly(tmp_path):
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "dataset_name": "WOMD",
                "split": "validation",
                "shard_glob": str(tmp_path / "nonexistent-*-shard.tfrecord"),
                "max_num_objects": 64,
            }
        )
    )

    with pytest.raises(ValueError, match="matched zero files"):
        load_dataset_config(str(config_path))


# ---------------------------------------------------------------------
# Case H: shard_glob resolves to exact sorted filenames.
# ---------------------------------------------------------------------


def test_case_h_shard_glob_resolves_sorted(tmp_path):
    # Create out-of-order shard files.
    names = [
        "validation_tfexample.tfrecord-00002-of-00003",
        "validation_tfexample.tfrecord-00000-of-00003",
        "validation_tfexample.tfrecord-00001-of-00003",
    ]
    for name in names:
        (tmp_path / name).write_text("fake")

    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "dataset_name": "WOMD",
                "split": "validation",
                "shard_glob": str(tmp_path / "validation_tfexample.tfrecord-*-of-00003"),
                "max_num_objects": 64,
            }
        )
    )

    expansion_config = load_dataset_config(str(config_path))

    resolved_names = [Path(p).name for p in expansion_config.shard_paths]
    assert resolved_names == sorted(names)


def test_shards_list_resolves_sorted_and_rejects_duplicates(tmp_path):
    names = ["shard_b.tfrecord", "shard_a.tfrecord"]
    for name in names:
        (tmp_path / name).write_text("fake")

    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "dataset_name": "WOMD",
                "split": "validation",
                "shards": [str(tmp_path / n) for n in names],
                "max_num_objects": 64,
            }
        )
    )

    expansion_config = load_dataset_config(str(config_path))
    resolved_names = [Path(p).name for p in expansion_config.shard_paths]
    assert resolved_names == sorted(names)

    # Duplicate paths in an explicit shard list must be rejected.
    dup_config_path = tmp_path / "dataset_dup.yaml"
    dup_path = str(tmp_path / names[0])
    dup_config_path.write_text(
        yaml.dump(
            {
                "dataset_name": "WOMD",
                "split": "validation",
                "shards": [dup_path, dup_path],
                "max_num_objects": 64,
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset_config(str(dup_config_path))


def test_missing_shard_file_fails_loudly(tmp_path):
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "dataset_name": "WOMD",
                "split": "validation",
                "shards": [str(tmp_path / "does_not_exist.tfrecord")],
                "max_num_objects": 64,
            }
        )
    )
    with pytest.raises(ValueError, match="do not exist on disk"):
        load_dataset_config(str(config_path))


def test_repeat_other_than_one_rejected(tmp_path):
    fake_shard = tmp_path / "shard.tfrecord"
    fake_shard.write_text("fake")

    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "dataset_name": "WOMD",
                "split": "validation",
                "shards": [str(fake_shard)],
                "max_num_objects": 64,
                "repeat": 3,
            }
        )
    )
    with pytest.raises(ValueError, match="repeat"):
        load_dataset_config(str(config_path))


# ---------------------------------------------------------------------
# Case N: legacy single-path config still loads, defaults to validation.
# ---------------------------------------------------------------------


def test_case_n_legacy_single_path_config_defaults_to_validation(tmp_path):
    fake_shard = tmp_path / "validation_tfexample.tfrecord-00000-of-00150"
    fake_shard.write_text("fake")

    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "path": str(fake_shard),
                "max_num_objects": 64,
                "repeat": 1,
                "shuffle_seed": None,
            }
        )
    )

    expansion_config = load_dataset_config(str(config_path))

    assert expansion_config.split == "validation"
    assert expansion_config.shard_paths == [str(fake_shard)]
    assert expansion_config.max_num_objects == 64
    assert expansion_config.repeat == 1


def test_real_configs_dataset_yaml_still_loads_as_legacy_validation():
    """Confirms configs/dataset.yaml (untouched by this commit) still
    parses under the legacy branch and defaults to split=validation.
    """

    expansion_config = load_dataset_config("configs/dataset.yaml")
    assert expansion_config.split == "validation"
    assert len(expansion_config.shard_paths) == 1
    assert expansion_config.shard_paths[0].endswith(
        "validation_tfexample.tfrecord-00000-of-00150"
    )


# ---------------------------------------------------------------------
# Case A: two shards with same record_index produce distinct scene_key.
# ---------------------------------------------------------------------


def test_case_a_same_record_index_different_shards_distinct_scene_key():
    scene_key_a = f"validation_tfexample.tfrecord-00000-of-00150#28"
    scene_key_b = f"validation_tfexample.tfrecord-00005-of-00150#28"

    assert scene_key_a != scene_key_b
    # Both share the same numeric record_index by construction.
    assert scene_key_a.rsplit("#", 1)[1] == scene_key_b.rsplit("#", 1)[1] == "28"
    # But the shard portion differs.
    assert scene_key_a.split("#")[0] != scene_key_b.split("#")[0]


# ---------------------------------------------------------------------
# Case B: two shards with identical transition metadata (same
# record_index/frame/lane ids) produce distinct candidate_id.
# ---------------------------------------------------------------------


def test_case_b_identical_transition_metadata_different_shards_distinct_candidate_id():
    scene_key_a = "validation_tfexample.tfrecord-00000-of-00150#28"
    scene_key_b = "validation_tfexample.tfrecord-00005-of-00150#28"

    candidate_id_a = make_candidate_id(scene_key_a, 50, 485, 344)
    candidate_id_b = make_candidate_id(scene_key_b, 50, 485, 344)

    assert candidate_id_a != candidate_id_b
    assert candidate_id_a == "validation_tfexample.tfrecord-00000-of-00150#28__t50__485_344"
    assert candidate_id_b == "validation_tfexample.tfrecord-00005-of-00150#28__t50__485_344"



# ---------------------------------------------------------------------
# resolve_physical_shard (fix commit: harden multi-shard inspection and
# shard reload safety). All multi-shard cases here are SYNTHETIC --
# only one real physical WOMD shard exists locally
# (data/womd/validation/validation_tfexample.tfrecord-00000-of-00150),
# so multi-shard resolution is exercised via in-memory
# DatasetExpansionConfig construction, never by copying the real 770MB
# file.
# ---------------------------------------------------------------------


def test_resolve_one_shard_config_no_split_filter(tmp_path):
    fake_shard = tmp_path / "validation_tfexample.tfrecord-00000-of-00150"
    fake_shard.write_text("fake")

    expansion_config = DatasetExpansionConfig(
        dataset_name="WOMD",
        split="validation",
        shard_paths=[str(fake_shard)],
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )

    resolved = resolve_physical_shard(
        expansion_config,
        source_shard="validation_tfexample.tfrecord-00000-of-00150",
    )
    assert resolved == str(fake_shard)


def test_resolve_multi_shard_exact_match(tmp_path):
    shard_a = tmp_path / "validation_tfexample.tfrecord-00000-of-00150"
    shard_b = tmp_path / "validation_tfexample.tfrecord-00005-of-00150"
    shard_a.write_text("fake")
    shard_b.write_text("fake")

    expansion_config = DatasetExpansionConfig(
        dataset_name="WOMD",
        split="validation",
        shard_paths=[str(shard_a), str(shard_b)],
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )

    resolved = resolve_physical_shard(
        expansion_config,
        source_shard="validation_tfexample.tfrecord-00005-of-00150",
    )
    assert resolved == str(shard_b)


def test_resolve_multi_shard_no_match_raises_no_fallback(tmp_path):
    shard_a = tmp_path / "validation_tfexample.tfrecord-00000-of-00150"
    shard_a.write_text("fake")

    expansion_config = DatasetExpansionConfig(
        dataset_name="WOMD",
        split="validation",
        shard_paths=[str(shard_a)],
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )

    with pytest.raises(ValueError, match="Unknown shard"):
        resolve_physical_shard(
            expansion_config,
            source_shard="validation_tfexample.tfrecord-00099-of-00150",
        )


def test_resolve_duplicate_basename_raises_ambiguous(tmp_path):
    dir_a = tmp_path / "A"
    dir_b = tmp_path / "B"
    dir_a.mkdir()
    dir_b.mkdir()
    path_a = dir_a / "foo.tfrecord"
    path_b = dir_b / "foo.tfrecord"
    path_a.write_text("fake")
    path_b.write_text("fake")

    expansion_config = DatasetExpansionConfig(
        dataset_name="WOMD",
        split="validation",
        shard_paths=[str(path_a), str(path_b)],
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )

    with pytest.raises(ValueError, match="Ambiguous shard"):
        resolve_physical_shard(expansion_config, source_shard="foo.tfrecord")


def test_resolve_split_mismatch_raises_before_path_work(tmp_path):
    # Deliberately reference a shard basename that does not exist
    # anywhere in shard_paths -- if the split check did NOT fire first,
    # this would instead raise the "Unknown shard" error. Asserting the
    # split-mismatch message proves the split check runs before any
    # path matching is attempted.
    fake_shard = tmp_path / "training_tfexample.tfrecord-00000-of-01000"
    fake_shard.write_text("fake")

    expansion_config = DatasetExpansionConfig(
        dataset_name="WOMD",
        split="validation",
        shard_paths=[str(fake_shard)],
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )

    with pytest.raises(ValueError, match="source_split mismatch"):
        resolve_physical_shard(
            expansion_config,
            source_shard="this_basename_does_not_exist_anywhere.tfrecord",
            source_split="training",
        )


def test_resolve_custom_arbitrary_path_no_hardcoded_convention():
    """Proves resolve_physical_shard uses ONLY expansion_config.shard_paths
    -- never a hardcoded data/womd/<split>/<file> convention. Uses a
    fabricated path that does not exist on disk (pure path-matching
    logic; resolve_physical_shard never touches the filesystem itself,
    only DatasetExpansionConfig.shard_paths as already-resolved
    in-memory strings)."""

    custom_path = "/mnt/custom/location/training_tfexample.tfrecord-00003-of-01000"

    expansion_config = DatasetExpansionConfig(
        dataset_name="WOMD",
        split="training",
        shard_paths=[custom_path],
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )

    resolved = resolve_physical_shard(
        expansion_config,
        source_shard="training_tfexample.tfrecord-00003-of-01000",
        source_split="training",
    )
    assert resolved == custom_path


# ---------------------------------------------------------------------
# select_single_shard_for_inspection (shared CLI shard-selection
# helper used by inspect_lane_geometry.py / inspect_ego_lane_sequence.py
# / inspect_merge_candidate.py).
# ---------------------------------------------------------------------


def test_select_single_shard_auto_selects_when_only_one_configured(tmp_path):
    fake_shard = tmp_path / "validation_tfexample.tfrecord-00000-of-00150"
    fake_shard.write_text("fake")

    expansion_config = DatasetExpansionConfig(
        dataset_name="WOMD",
        split="validation",
        shard_paths=[str(fake_shard)],
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )

    resolved = select_single_shard_for_inspection(expansion_config)
    assert resolved == str(fake_shard)


def test_select_single_shard_explicit_source_shard_multi_config(tmp_path):
    shard_a = tmp_path / "validation_tfexample.tfrecord-00000-of-00150"
    shard_b = tmp_path / "validation_tfexample.tfrecord-00005-of-00150"
    shard_a.write_text("fake")
    shard_b.write_text("fake")

    expansion_config = DatasetExpansionConfig(
        dataset_name="WOMD",
        split="validation",
        shard_paths=[str(shard_a), str(shard_b)],
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )

    resolved = select_single_shard_for_inspection(
        expansion_config,
        source_shard="validation_tfexample.tfrecord-00005-of-00150",
    )
    assert resolved == str(shard_b)


def test_select_single_shard_multi_config_no_source_shard_raises_ambiguous(tmp_path):
    shard_a = tmp_path / "validation_tfexample.tfrecord-00000-of-00150"
    shard_b = tmp_path / "validation_tfexample.tfrecord-00005-of-00150"
    shard_a.write_text("fake")
    shard_b.write_text("fake")

    expansion_config = DatasetExpansionConfig(
        dataset_name="WOMD",
        split="validation",
        shard_paths=[str(shard_a), str(shard_b)],
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )

    with pytest.raises(ValueError, match="ambiguous") as exc_info:
        select_single_shard_for_inspection(expansion_config, record_index=28)

    message = str(exc_info.value)
    assert "validation_tfexample.tfrecord-00000-of-00150" in message
    assert "validation_tfexample.tfrecord-00005-of-00150" in message


def test_select_single_shard_unknown_source_shard_raises(tmp_path):
    fake_shard = tmp_path / "validation_tfexample.tfrecord-00000-of-00150"
    fake_shard.write_text("fake")

    expansion_config = DatasetExpansionConfig(
        dataset_name="WOMD",
        split="validation",
        shard_paths=[str(fake_shard)],
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )

    with pytest.raises(ValueError, match="Unknown shard"):
        select_single_shard_for_inspection(
            expansion_config, source_shard="does_not_exist.tfrecord"
        )


def test_select_single_shard_duplicate_basename_raises(tmp_path):
    dir_a = tmp_path / "A"
    dir_b = tmp_path / "B"
    dir_a.mkdir()
    dir_b.mkdir()
    path_a = dir_a / "foo.tfrecord"
    path_b = dir_b / "foo.tfrecord"
    path_a.write_text("fake")
    path_b.write_text("fake")

    expansion_config = DatasetExpansionConfig(
        dataset_name="WOMD",
        split="validation",
        shard_paths=[str(path_a), str(path_b)],
        max_num_objects=64,
        repeat=1,
        shuffle_seed=None,
    )

    with pytest.raises(ValueError, match="Ambiguous shard"):
        select_single_shard_for_inspection(expansion_config, source_shard="foo.tfrecord")
