"""Safety-guard regression tests for
``scripts/fetch_womd_scenario_proto_shards.py``.

The whole point of this script is to prevent a repeat of the earlier
incident where an unbounded bulk download exhausted local disk -- these
tests exercise exactly the guards meant to make that impossible."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from scripts.fetch_womd_scenario_proto_shards import (
    MAX_SHARDS_PER_RUN,
    _parse_shard_list,
    _remote_path,
    main,
)


def test_wildcard_shard_selection_rejected():
    with pytest.raises(ValueError, match="Wildcard/--all"):
        _parse_shard_list("*", total_shards=1000)
    with pytest.raises(ValueError, match="Wildcard/--all"):
        _parse_shard_list("all", total_shards=1000)


def test_explicit_shard_list_parsed_and_sorted_deduped():
    assert _parse_shard_list("49,0,46,0", total_shards=1000) == [0, 46, 49]


def test_out_of_range_shard_index_rejected():
    with pytest.raises(ValueError, match="out of range"):
        _parse_shard_list("1000", total_shards=1000)
    with pytest.raises(ValueError, match="out of range"):
        _parse_shard_list("-1", total_shards=1000)


def test_empty_shard_list_rejected():
    with pytest.raises(ValueError, match="empty list"):
        _parse_shard_list("", total_shards=1000)


def test_remote_path_uses_correct_split_prefix_and_padding():
    assert _remote_path("training", 46, 1000) == (
        "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/"
        "training/training.tfrecord-00046-of-01000"
    )
    assert _remote_path("validation", 5, 150) == (
        "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/"
        "validation/validation.tfrecord-00005-of-00150"
    )


def test_max_shards_per_run_guard_blocks_large_batches():
    too_many = ",".join(str(i) for i in range(MAX_SHARDS_PER_RUN + 1))
    with pytest.raises(ValueError, match="Refusing to fetch"):
        main(["--split", "training", "--shards", too_many, "--dry-run"])


def test_dry_run_never_downloads(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(
        "scripts.fetch_womd_scenario_proto_shards.subprocess.run",
        lambda *a, **k: called.append(a),
    )
    exit_code = main(
        [
            "--split",
            "training",
            "--shards",
            "0,1",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert called == []
    assert list(tmp_path.iterdir()) == []


def test_low_free_space_aborts_without_downloading(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(
        "scripts.fetch_womd_scenario_proto_shards.subprocess.run",
        lambda *a, **k: called.append(a),
    )

    class _FakeUsage:
        free = 1_000  # far below the safety margin

    monkeypatch.setattr(
        "scripts.fetch_womd_scenario_proto_shards.shutil.disk_usage",
        lambda *_: _FakeUsage(),
    )
    exit_code = main(
        [
            "--split",
            "training",
            "--shards",
            "0",
            "--output-dir",
            str(tmp_path),
            "--yes",
        ]
    )
    assert exit_code == 1
    assert called == []
