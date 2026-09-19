"""PPO P1: config loading for PPO run configs and reward configs.

Mirrors this repo's existing YAML-loading convention (see
``src.control.ltv_mpc.load_mpc_config``,
``src.planning.frenet_planner.load_planner_config``,
``src.scenarios.scenario_loader.load_dataset_config``): plain
``yaml.safe_load`` into a frozen dataclass with explicitly typed
fields, a loader function taking a default path. No new config
framework/dependency is introduced.

This module only loads/validates structure -- it does not interpret
what the config MEANS for reward computation (that is
``src.rewards.merge_reward``) or for the PPO network/loss (that is
``src.policies.ppo.*``). Kept deliberately dependency-light so P1 can
land config loading + seed handling + a checkpoint skeleton before any
PPO algorithm code exists.
"""

import dataclasses
from typing import List, Optional

import yaml


@dataclasses.dataclass(frozen=True)
class NetworkConfig:
    observation_dim: int
    num_actions: int
    hidden_sizes: List[int]
    activation: str


@dataclasses.dataclass(frozen=True)
class PPOHyperparameters:
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    ppo_epochs: int
    num_minibatches: int


@dataclasses.dataclass(frozen=True)
class RolloutConfig:
    downstream_mode: str


@dataclasses.dataclass(frozen=True)
class TrackingConfig:
    wandb_project: str
    wandb_mode: str


@dataclasses.dataclass(frozen=True)
class SmokeConfig:
    max_maneuvers: int
    num_updates: int
    max_episode_steps: int


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    """One fully-parsed PPO run config (``configs/ppo/*.yaml``)."""

    seed: int
    network: NetworkConfig
    hyperparameters: PPOHyperparameters
    rollout: RolloutConfig
    reward_config_path: str
    tracking: TrackingConfig
    smoke: Optional[SmokeConfig]
    # Kept so downstream code (checkpointing, W&B config logging) can
    # always recover exactly which file produced this config, per
    # PPO_PLAN.md SS8/SS10 ("git_sha, reward_version, seed, ... network_layers"
    # and the checkpoint's "complete PPO config (or a config snapshot)").
    source_path: str


@dataclasses.dataclass(frozen=True)
class RewardTerminalTable:
    success: float
    failure_collision: float
    failure_offroad: float
    truncation_horizon: float
    none: float


@dataclasses.dataclass(frozen=True)
class RewardDecisionCost:
    real_decision_step: float
    auto_execution_step: float


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    """One fully-parsed reward config (``configs/reward/*.yaml``)."""

    reward_version: str
    terminal: RewardTerminalTable
    decision_cost: RewardDecisionCost
    source_path: str


def load_ppo_config(path: str = "configs/ppo/ppo_base.yaml") -> PPOConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    network = raw["network"]
    hyperparameters = raw["hyperparameters"]
    rollout = raw["rollout"]
    tracking = raw["tracking"]
    smoke_raw = raw.get("smoke")

    return PPOConfig(
        seed=int(raw["seed"]),
        network=NetworkConfig(
            observation_dim=int(network["observation_dim"]),
            num_actions=int(network["num_actions"]),
            hidden_sizes=[int(h) for h in network["hidden_sizes"]],
            activation=str(network["activation"]),
        ),
        hyperparameters=PPOHyperparameters(
            learning_rate=float(hyperparameters["learning_rate"]),
            gamma=float(hyperparameters["gamma"]),
            gae_lambda=float(hyperparameters["gae_lambda"]),
            clip_epsilon=float(hyperparameters["clip_epsilon"]),
            value_coef=float(hyperparameters["value_coef"]),
            entropy_coef=float(hyperparameters["entropy_coef"]),
            max_grad_norm=float(hyperparameters["max_grad_norm"]),
            ppo_epochs=int(hyperparameters["ppo_epochs"]),
            num_minibatches=int(hyperparameters["num_minibatches"]),
        ),
        rollout=RolloutConfig(
            downstream_mode=str(rollout["downstream_mode"]),
        ),
        reward_config_path=str(raw["reward_config_path"]),
        tracking=TrackingConfig(
            wandb_project=str(tracking["wandb_project"]),
            wandb_mode=str(tracking["wandb_mode"]),
        ),
        smoke=(
            SmokeConfig(
                max_maneuvers=int(smoke_raw["max_maneuvers"]),
                num_updates=int(smoke_raw["num_updates"]),
                max_episode_steps=int(smoke_raw["max_episode_steps"]),
            )
            if smoke_raw is not None
            else None
        ),
        source_path=path,
    )


def load_reward_config(path: str = "configs/reward/merge_reward_v0.yaml") -> RewardConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    terminal = raw["terminal"]
    decision_cost = raw["decision_cost"]

    return RewardConfig(
        reward_version=str(raw["reward_version"]),
        terminal=RewardTerminalTable(
            success=float(terminal["success"]),
            failure_collision=float(terminal["failure_collision"]),
            failure_offroad=float(terminal["failure_offroad"]),
            truncation_horizon=float(terminal["truncation_horizon"]),
            none=float(terminal["none"]),
        ),
        decision_cost=RewardDecisionCost(
            real_decision_step=float(decision_cost["real_decision_step"]),
            auto_execution_step=float(decision_cost["auto_execution_step"]),
        ),
        source_path=path,
    )
