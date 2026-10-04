"""Frozen independent TEST dataset: structure, leakage and loader checks.

Dataset-structure tests only -- no policy is run on TEST.
"""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.environment.full_split_evaluator import load_decision_dataset_maneuver_specs

TEST_DIR = Path("data/manifests/test")
CANON = "data/manifests/merge_decision_manifest.csv"

pytestmark = pytest.mark.skipif(
    not (TEST_DIR / "test_freeze.json").exists(), reason="TEST dataset not built"
)


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def freeze():
    return json.loads((TEST_DIR / "test_freeze.json").read_text())


@pytest.fixture(scope="module")
def manifest():
    return _rows(TEST_DIR / "merge_decision_test_manifest.csv")


@pytest.fixture(scope="module")
def canon():
    return _rows(CANON)


@pytest.fixture(scope="module")
def test_specs():
    return load_decision_dataset_maneuver_specs("test")


def test_eligible_count_at_least_400(manifest):
    assert len(manifest) >= 400


def test_shards_start_at_00006_and_contiguous(freeze):
    idx = [int(s.split("tfrecord-")[1][:5]) for s in freeze["source_shards"]]
    assert idx[0] == 6 and freeze["first_shard"] == "00006"
    assert idx == list(range(idx[0], idx[-1] + 1))
    assert not any(i <= 5 for i in idx)


def test_final_shard_is_first_to_reach_400(freeze):
    cum = list(freeze["per_shard_cumulative_eligible"].values())
    assert cum[-1] >= 400 and all(c < 400 for c in cum[:-1])
    assert cum[-1] == freeze["final_eligible_maneuver_count"]


def test_no_scenario_overlap_with_train_or_validation(manifest, canon):
    ts = {r["scenario_id"] for r in manifest}
    for split in ("training", "validation"):
        assert not ts & {r["scenario_id"] for r in canon if r["split"] == split}


def test_no_duplicates_or_canonical_id_overlap(manifest, canon):
    for key in ("maneuver_id", "candidate_id"):
        vals = [r[key] for r in manifest]
        assert len(vals) == len(set(vals))
        assert not set(vals) & {r[key] for r in canon}


def test_loader_count_matches_frozen_manifest(test_specs, freeze, manifest):
    assert len(test_specs) == freeze["final_eligible_maneuver_count"] == len(manifest)


def test_loader_only_eligible_ab_core_support(manifest):
    assert {r["decision_tier"] for r in manifest} <= {"A", "B"}
    assert {r["dataset_role"] for r in manifest} <= {"CORE", "SUPPORT"}
    assert {r["merge_context_status"] for r in manifest} == {"MERGE_CONTEXT_ELIGIBLE"}


def test_loader_ids_exactly_manifest_ids(test_specs, manifest):
    assert {s.maneuver_id for s in test_specs} == {r["maneuver_id"] for r in manifest}


def test_test_loader_reproducible():
    a = [s.maneuver_id for s in load_decision_dataset_maneuver_specs("test")]
    b = [s.maneuver_id for s in load_decision_dataset_maneuver_specs("test")]
    assert a == b


def test_train_validation_loaders_unchanged(freeze):
    assert len(load_decision_dataset_maneuver_specs("train")) == 1097
    assert len(load_decision_dataset_maneuver_specs("validation")) == 388
    h = hashlib.sha256(Path(CANON).read_bytes()).hexdigest()
    assert h == freeze["canonical_manifest_sha256_at_freeze"]


def test_frozen_artifact_hashes_match(freeze):
    for path_key, sha_key in (("test_manifest", "test_manifest_sha256"),
                              ("test_evidence", "test_evidence_sha256"),
                              ("filter_config", "filter_config_sha256")):
        assert hashlib.sha256(Path(freeze[path_key]).read_bytes()).hexdigest() == freeze[sha_key]
    assert freeze["policy_evaluation_on_test"] == "NOT RUN"
