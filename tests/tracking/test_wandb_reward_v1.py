"""Reward V1 W&B contract tests (outputs/reward_v1_spec/
REWARD_V1_SPEC_FINAL.md Section 13/22, this session's brief Section 31).

Runs entirely in ``WANDB_MODE=offline`` (no network/auth required) --
mirrors tests/tracking/test_wandb_logger.py's existing pattern exactly.
"""

import glob
import json
import os

os.environ["WANDB_MODE"] = "offline"

from src.tracking.wandb_logger import MINIMUM_METRICS, OPTIONAL_PROVENANCE_KEYS, WandbLogger  # noqa: E402
from src.training.provenance import build_wandb_provenance_config, load_dataset_provenance  # noqa: E402


def _minimal_v1_config():
    return {
        "git_sha": "deadbeef",
        "reward_version": "v1",
        "seed": 0,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "batch_size": 64,
        "ppo_epochs": 4,
        "network_layers": [256, 64, 32],
    }


# ------------------------------------------------------------------
# Reward key presence (Section 31, tests 45-51)
# ------------------------------------------------------------------


def test_reward_total_key_declared():
    assert "reward/total" in MINIMUM_METRICS


def test_reward_terminal_key_declared():
    assert "reward/terminal" in MINIMUM_METRICS


def test_reward_safety_key_declared():
    assert "reward/safety" in MINIMUM_METRICS


def test_reward_progress_key_declared():
    assert "reward/progress" in MINIMUM_METRICS


def test_reward_decision_key_declared():
    assert "reward/decision" in MINIMUM_METRICS


def test_diagnostic_keys_not_in_minimum_metrics():
    """reward_diag/ttc_penalty and reward_diag/gap_penalty are NOT part
    of the canonical reward sum -- they must not appear in
    MINIMUM_METRICS (which documents the canonical reward/eval
    contract); log_metrics still accepts them as arbitrary names (see
    test_log_metrics_can_log_diagnostic_keys below), just not as a
    required minimum."""

    assert "reward_diag/ttc_penalty" not in MINIMUM_METRICS
    assert "reward_diag/gap_penalty" not in MINIMUM_METRICS


def test_log_metrics_can_log_diagnostic_and_reward_v1_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    logger = WandbLogger(project="its-merge-ppo-test", mode="offline", dir=str(tmp_path))
    try:
        logger.log_config(_minimal_v1_config())
        logger.log_metrics(
            {
                "reward/terminal": 1.0,
                "reward/safety": -0.02,
                "reward/progress": 0.1,
                "reward/decision": 0.0,
                "reward/decision_cost": -0.01,
                "reward/total": 1.07,
                "reward_diag/ttc_penalty": -1.0,
                "reward_diag/gap_penalty": 0.0,
            },
            step=0,
        )
        run_dir = logger.run_dir
    finally:
        logger.finish()

    assert os.path.isdir(run_dir)


# ------------------------------------------------------------------
# Run provenance (Section 31, tests 52-54 + filter/manifest hash)
# ------------------------------------------------------------------


def test_optional_provenance_keys_declared():
    assert "dataset/version" in OPTIONAL_PROVENANCE_KEYS
    assert "dataset/freeze_sha" in OPTIONAL_PROVENANCE_KEYS
    assert "reward/version" in OPTIONAL_PROVENANCE_KEYS
    assert "filter_config_hash" in OPTIONAL_PROVENANCE_KEYS
    assert "canonical_manifest_hash" in OPTIONAL_PROVENANCE_KEYS


def test_load_dataset_provenance_matches_freeze_json():
    provenance = load_dataset_provenance()
    with open("data/manifests/v2/MERGE_DECISION_DATASET_V2_FREEZE.json", encoding="utf-8") as f:
        raw = json.load(f)

    assert provenance.version == raw["version"]
    assert provenance.version == "v2.0"
    assert provenance.git_sha == raw["git_sha"]
    assert provenance.filter_config_hash == raw["filter_config_sha256"]
    assert provenance.canonical_manifest_hash == raw["canonical_manifest_sha256"]


def test_provenance_not_hardcoded_a_second_time():
    """The freeze JSON is the sole source of truth -- this module must
    read it, not embed a duplicate literal hash anywhere in its own
    source."""

    import inspect

    from src.training import provenance as provenance_module

    source = inspect.getsource(provenance_module)
    with open("data/manifests/v2/MERGE_DECISION_DATASET_V2_FREEZE.json", encoding="utf-8") as f:
        raw = json.load(f)

    assert raw["filter_config_sha256"] not in source
    assert raw["canonical_manifest_sha256"] not in source


def test_build_wandb_provenance_config_shape():
    config = build_wandb_provenance_config(reward_version="v1")
    assert set(config) == set(OPTIONAL_PROVENANCE_KEYS)
    assert config["dataset/version"] == "v2.0"
    assert config["reward/version"] == "v1"


def test_provenance_config_can_be_logged_to_wandb(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    logger = WandbLogger(project="its-merge-ppo-test", mode="offline", dir=str(tmp_path))
    try:
        config = {**_minimal_v1_config(), **build_wandb_provenance_config(reward_version="v1")}
        logger.log_config(config)
        run_dir = logger.run_dir
    finally:
        logger.finish()

    assert os.path.isdir(run_dir)
    written_files = glob.glob(os.path.join(str(tmp_path), "**", "*"), recursive=True)
    written_files = [f for f in written_files if os.path.isfile(f)]
    assert len(written_files) > 0
