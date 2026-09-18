"""W&B experiment-tracking logger (docs/ppo/PPO_PLAN.md SS8, SS0.1 P2).

P2 implementation. Thin wrapper around the ``wandb`` SDK supporting
both online and offline (``WANDB_MODE=offline``) modes. Logs the
per-run config (git_sha, reward_version, seed, hyperparameters,
network_layers -- SS8) and arbitrary named scalar metrics, so later
phases (train/*, ppo/*, action/*, downstream/*, safety/*) can log
through the same interface without changing this module.

This module intentionally does not enforce that every
``REQUIRED_CONFIG_KEYS``/``MINIMUM_METRICS`` entry be present on any
single call -- most of the P0-P5 metric list only becomes available in
P3/P4/P5 (e.g. ``ppo/policy_loss`` cannot exist before the PPO loss is
implemented). ``log_config`` instead validates that all
``REQUIRED_CONFIG_KEYS`` are present (those must always be knowable up
front, from config + git, by P2), while ``log_metrics`` accepts any
metric name so callers can log a growing subset over time.
"""

from typing import Any, Dict, Optional

import wandb

# Minimum per-run config keys (PPO_PLAN.md SS8).
REQUIRED_CONFIG_KEYS = (
    "git_sha",
    "reward_version",
    "seed",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip_epsilon",
    "entropy_coef",
    "value_coef",
    "batch_size",
    "ppo_epochs",
    "network_layers",
)

# Minimum metrics (PPO_PLAN.md SS8). Not all of these are available in
# every phase -- see module docstring. log_metrics accepts any name,
# not just these, so later phases can add more without touching this
# module.
MINIMUM_METRICS = (
    "train/episode_return",
    "train/success_rate",
    "train/collision_rate",
    "train/offroad_rate",
    "train/timeout_rate",
    "train/episode_length",
    "reward/terminal",
    "reward/decision_cost",
    "reward/total",
    "ppo/policy_loss",
    "ppo/value_loss",
    "ppo/entropy",
    "ppo/approx_kl",
    "ppo/clip_fraction",
    "ppo/explained_variance",
    "ppo/grad_norm",
    "action/keep_ratio",
    "action/follow_ratio",
    "action/merge_ratio",
    "action/stop_ratio",
    "downstream/intervention_rate",
    "downstream/planner_infeasible_rate",
    "downstream/collision_blocked_rate",
    "downstream/controller_failure_rate",
    "runtime/env_steps_per_sec",
)


class WandbLogger:
    """Thin wrapper around the ``wandb`` SDK supporting both online and
    offline (``WANDB_MODE=offline``) modes.

    Online vs. offline is controlled the standard ``wandb`` way -- via
    the ``WANDB_MODE`` environment variable, or explicitly via the
    ``mode`` constructor argument (passed through to ``wandb.init``).
    If W&B auth is unavailable, set ``WANDB_MODE=offline`` (or pass
    ``mode="offline"``) rather than skipping logging entirely (SS8).
    """

    def __init__(
        self,
        project: str,
        mode: str = "offline",
        run_name: Optional[str] = None,
        **init_kwargs: Any,
    ):
        self.project = project
        self.mode = mode
        self._run = wandb.init(
            project=project,
            mode=mode,
            name=run_name,
            **init_kwargs,
        )

    def log_config(self, config: Dict[str, Any]) -> None:
        """Logs the per-run config to W&B (``wandb.config.update``).

        Validates that every key PPO_PLAN.md SS8 requires
        (``REQUIRED_CONFIG_KEYS``) is present in ``config`` -- these
        are all knowable up front (git SHA, reward version, seed,
        fixed hyperparameters, network layer sizes), so a missing one
        indicates a caller bug rather than a not-yet-available metric.
        """

        missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
        if missing:
            raise ValueError(
                f"WandbLogger.log_config: missing required config keys "
                f"{missing} (PPO_PLAN.md SS8 requires all of "
                f"{REQUIRED_CONFIG_KEYS})."
            )

        self._run.config.update(config, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Logs a dict of named scalar metrics at the given step.

        Accepts ANY metric name (not restricted to
        ``MINIMUM_METRICS``) so later phases can add train/*, ppo/*,
        action/*, downstream/*, safety/* metrics without changing this
        logger.
        """

        self._run.log(metrics, step=step)

    def finish(self) -> None:
        """Finishes the W&B run (flushes offline files to disk)."""

        if self._run is not None:
            self._run.finish()
            self._run = None

    @property
    def run_dir(self) -> Optional[str]:
        """Local directory this run's files (offline included) are
        written under, or ``None`` if the run has already finished."""

        return self._run.dir if self._run is not None else None

    @property
    def run_id(self) -> Optional[str]:
        return self._run.id if self._run is not None else None

    def __enter__(self) -> "WandbLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()
