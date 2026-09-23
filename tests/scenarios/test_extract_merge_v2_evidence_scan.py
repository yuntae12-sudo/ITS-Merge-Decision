"""Regression coverage for the single-pass, memory-bounded scenario-id
join used by ``scripts/extract_merge_v2_evidence.py``.

Was previously O(n^2): one full rescan of the tf_example shard from
record 0 for every candidate row. Also previously loaded every parsed
topology from ``--scenario-proto`` into memory at once, which does not
fit for the full ~1000-shard WOMD training scenario split -- fixed by
filtering to only the wanted scenario ids as they stream past."""

import sys
import types
from pathlib import Path

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.extract_merge_v2_evidence import (
    _build_scenario_id_lookup,
    _scenario_ids_for_shard,
)
from src.scenarios.scenario_proto_loader import index_scenario_topologies_for_ids


class _FakeExpansion:
    dataset_name = "WOMD"
    split = "training"


def _write_fake_tfexample_shard(path, scenario_ids):
    with tf.io.TFRecordWriter(str(path)) as writer:
        for scenario_id in scenario_ids:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "scenario/id": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[scenario_id.encode("utf-8")])
                        )
                    }
                )
            )
            writer.write(example.SerializeToString())


def test_scenario_ids_for_shard_single_pass_stops_at_highest_wanted_index(tmp_path, monkeypatch):
    shard_path = tmp_path / "fake_tfexample.tfrecord-00000-of-00001"
    ids = [f"scenario-{i}" for i in range(20)]
    _write_fake_tfexample_shard(shard_path, ids)

    visited = []
    from src.scenarios import scenario_proto_loader

    original = scenario_proto_loader.iter_tfexample_scenario_ids

    def spy(path):
        for record_index, scenario_id in original(path):
            visited.append(record_index)
            yield record_index, scenario_id

    import scripts.extract_merge_v2_evidence as mod

    monkeypatch.setattr(mod, "iter_tfexample_scenario_ids", spy)

    wanted = {3, 7}
    resolved = _scenario_ids_for_shard(str(shard_path), wanted)

    assert resolved == {3: "scenario-3", 7: "scenario-7"}
    # Single sequential pass must stop at the highest wanted index (7),
    # never scanning the remaining 12 records in the shard.
    assert visited == list(range(8))


def test_build_scenario_id_lookup_groups_by_resolved_shard_path(monkeypatch, tmp_path):
    shard0 = tmp_path / "shard0.tfrecord"
    shard1 = tmp_path / "shard1.tfrecord"
    _write_fake_tfexample_shard(shard0, [f"s0-{i}" for i in range(6)])
    _write_fake_tfexample_shard(shard1, [f"s1-{i}" for i in range(3)])

    rows = [
        {"source_shard": "shard0", "record_index": "1", "source_split": "training"},
        {"source_shard": "shard0", "record_index": "5", "source_split": "training"},
        {"source_shard": "shard1", "record_index": "2", "source_split": "training"},
    ]

    import scripts.extract_merge_v2_evidence as mod

    def fake_resolve(expansion, source_shard, source_split=None):
        return str({"shard0": shard0, "shard1": shard1}[source_shard])

    monkeypatch.setattr(mod, "resolve_physical_shard", fake_resolve)

    grouped = _build_scenario_id_lookup(rows, _FakeExpansion())

    assert grouped == {
        str(shard0): {1: "s0-1", 5: "s0-5"},
        str(shard1): {2: "s1-2"},
    }


def test_index_scenario_topologies_for_ids_ignores_unwanted_and_keeps_wanted():
    def fake_scenario(scenario_id):
        return types.SimpleNamespace(scenario_id=scenario_id, map_features=[])

    import src.scenarios.scenario_proto_loader as loader

    def fake_iter_scenario_protobufs(paths):
        for sid in ["keep-me", "skip-me", "also-keep"]:
            yield fake_scenario(sid)

    original = loader.iter_scenario_protobufs
    loader.iter_scenario_protobufs = fake_iter_scenario_protobufs
    try:
        index = index_scenario_topologies_for_ids(["unused-path"], {"keep-me", "also-keep"})
    finally:
        loader.iter_scenario_protobufs = original

    assert set(index.keys()) == {"keep-me", "also-keep"}


def test_index_scenario_topologies_for_ids_raises_on_duplicate_wanted_id():
    def fake_scenario(scenario_id):
        return types.SimpleNamespace(scenario_id=scenario_id, map_features=[])

    import src.scenarios.scenario_proto_loader as loader

    def fake_iter_scenario_protobufs(paths):
        yield fake_scenario("dup")
        yield fake_scenario("dup")

    original = loader.iter_scenario_protobufs
    loader.iter_scenario_protobufs = fake_iter_scenario_protobufs
    try:
        try:
            index_scenario_topologies_for_ids(["unused-path"], {"dup"})
            assert False, "expected ValueError for duplicate scenario id"
        except ValueError as exc:
            assert "dup" in str(exc)
    finally:
        loader.iter_scenario_protobufs = original


def test_numpy_json_encoder_serializes_numpy_bool_and_scalars():
    """Regression: geometry helpers like ``project_point_to_polyline``
    return numpy-backed values (e.g. ``np.float64`` comparisons produce
    ``np.bool_``), which the stdlib ``json`` module rejects with
    ``TypeError: Object of type bool_ is not JSON serializable`` --
    caught running the pilot extraction end-to-end against real WOMD
    data. ``_NumpyJSONEncoder`` must convert these without changing the
    represented value."""

    import json

    import numpy as np

    from scripts.extract_merge_v2_evidence import _NumpyJSONEncoder

    payload = {
        "flag": np.bool_(True),
        "count": np.int64(7),
        "distance": np.float64(3.5),
    }
    encoded = json.dumps(payload, cls=_NumpyJSONEncoder)
    decoded = json.loads(encoded)
    assert decoded == {"flag": True, "count": 7, "distance": 3.5}
