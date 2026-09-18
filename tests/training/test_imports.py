"""P1 import checks for the training/tracking packages (docs/ppo/
PPO_PLAN.md SS0.1/P1). Real rollout/GAE/trainer/W&B logic lands in
P2/P4/P5 -- these tests only confirm module structure/imports and that
P1 stubs raise ``NotImplementedError`` rather than silently no-op'ing.
"""

import pytest

from src.tracking import wandb_logger
from src.training import gae, rollout, trainer


def test_rollout_module_imports():
    assert hasattr(rollout, "collect_rollout")
    assert hasattr(rollout, "Transition")
    assert rollout.REQUIRED_TRANSITION_FIELDS == (
        "observation",
        "action",
        "reward",
        "next_observation",
        "terminated",
        "truncated",
        "value",
        "next_value",
        "log_prob",
        "policy_mask",
    )


def test_transition_dataclass_has_required_fields():
    field_names = {f.name for f in rollout.Transition.__dataclass_fields__.values()}
    for required in rollout.REQUIRED_TRANSITION_FIELDS:
        assert required in field_names


def test_gae_module_imports():
    assert hasattr(gae, "compute_gae")
    assert hasattr(gae, "normalize_advantages_masked")


def test_trainer_module_imports():
    assert hasattr(trainer, "run_training")


def test_wandb_logger_module_imports():
    assert hasattr(wandb_logger, "WandbLogger")
    assert "git_sha" in wandb_logger.REQUIRED_CONFIG_KEYS
    assert "reward_version" in wandb_logger.REQUIRED_CONFIG_KEYS
    assert "seed" in wandb_logger.REQUIRED_CONFIG_KEYS
    assert "train/episode_return" in wandb_logger.MINIMUM_METRICS
    assert "ppo/policy_loss" in wandb_logger.MINIMUM_METRICS


def test_training_stubs_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        rollout.collect_rollout()
    with pytest.raises(NotImplementedError):
        gae.compute_gae()
    with pytest.raises(NotImplementedError):
        gae.normalize_advantages_masked([], [])


def test_wandb_logger_init_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        wandb_logger.WandbLogger(project="its-merge-ppo")
