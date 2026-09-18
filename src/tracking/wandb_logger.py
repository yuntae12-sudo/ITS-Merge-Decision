"""W&B experiment-tracking logger (docs/ppo/PPO_PLAN.md SS8, SS0.1 P2).

P1 scope: structural skeleton only. Real online/offline W&B
initialization, config logging, and metric logging land in P2. The
minimum metric names and per-run config keys are fixed here as
constants so P2's implementation and any earlier test scaffolding
agree on naming from the start (see PPO_PLAN.md SS8 for the full list
and rationale).
"""

from typing import Any, Dict

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

# Minimum metrics (PPO_PLAN.md SS8).
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

    P1 skeleton: constructor and ``log_config``/``log_metrics`` are not
    yet implemented. Real ``wandb.init``/``wandb.log`` wiring lands in
    P2 per PPO_PLAN.md SS0.1/P2.
    """

    def __init__(self, project: str, mode: str = "offline", *args, **kwargs):
        self.project = project
        self.mode = mode
        raise NotImplementedError(
            "WandbLogger initialization lands in P2 (docs/ppo/PPO_PLAN.md "
            "SS0.1/P2). P1 only fixes the required config keys/metric names."
        )

    def log_config(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError(
            "WandbLogger.log_config lands in P2 (docs/ppo/PPO_PLAN.md "
            "SS0.1/P2)."
        )

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        raise NotImplementedError(
            "WandbLogger.log_metrics lands in P2 (docs/ppo/PPO_PLAN.md "
            "SS0.1/P2)."
        )
