"""P2 behavioral tests for the W&B logger (docs/ppo/PPO_PLAN.md
SS0.1/P2, SS8).

Runs entirely in ``WANDB_MODE=offline`` (set at module import time,
before any ``wandb.init`` call in this process) so no network/auth is
required. Verifies:
  - a config dict + at least one metric point can be logged without
    exception
  - files are actually written to the local wandb offline run
    directory
"""

import glob
import os

import pytest

os.environ["WANDB_MODE"] = "offline"

from src.tracking.wandb_logger import REQUIRED_CONFIG_KEYS, WandbLogger  # noqa: E402


def _minimal_config():
    return {
        "git_sha": "deadbeef",
        "reward_version": "v0",
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


def test_wandb_logger_module_imports():
    assert "git_sha" in REQUIRED_CONFIG_KEYS


def test_offline_smoke_run_logs_config_and_metric(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    logger = WandbLogger(project="its-merge-ppo-test", mode="offline", dir=str(tmp_path))
    try:
        logger.log_config(_minimal_config())
        logger.log_metrics(
            {
                "reward/terminal": 1.0,
                "reward/decision_cost": -0.01,
                "reward/total": 0.99,
            },
            step=0,
        )
        run_dir = logger.run_dir
        assert run_dir is not None
    finally:
        logger.finish()

    # The run directory (and its parent wandb offline-run tree) must
    # actually contain written files -- not just an in-memory no-op.
    assert os.path.isdir(run_dir)
    wandb_root = str(tmp_path)
    written_files = glob.glob(os.path.join(wandb_root, "**", "*"), recursive=True)
    written_files = [f for f in written_files if os.path.isfile(f)]
    assert len(written_files) > 0, (
        f"Expected offline wandb run files under {wandb_root}, found none. "
        f"run_dir={run_dir}"
    )


def test_log_config_rejects_missing_required_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    logger = WandbLogger(project="its-merge-ppo-test", mode="offline", dir=str(tmp_path))
    try:
        incomplete_config = dict(_minimal_config())
        del incomplete_config["git_sha"]
        with pytest.raises(ValueError):
            logger.log_config(incomplete_config)
    finally:
        logger.finish()


def test_log_metrics_accepts_arbitrary_metric_names(monkeypatch, tmp_path):
    """The logger must support logging arbitrary scalar metrics by
    name (not just MINIMUM_METRICS) so later phases can add
    train/*, ppo/*, action/*, downstream/*, safety/* without changing
    this module."""

    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    logger = WandbLogger(project="its-merge-ppo-test", mode="offline", dir=str(tmp_path))
    try:
        logger.log_metrics({"some/future_metric_not_in_minimum_list": 3.14}, step=0)
    finally:
        logger.finish()
