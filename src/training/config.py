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
    """PPO_PLAN.md SS6 baseline hyperparameters.

    ``value_coef`` (pre-P6 hardening Fix 4, deprecated-for-this-
    architecture note): the PPO paper's c1/``value_coef`` only affects
    training dynamics when the policy and value losses are combined
    into ONE shared gradient (i.e. shared or jointly-optimized
    parameters). This codebase's policy and value networks are fully
    INDEPENDENT ``flax.training.train_state.TrainState``s with separate
    ``apply_gradients`` calls (``src/policies/ppo/state.py``,
    SS7.2's Actor/Critic separation) -- ``src/training/trainer.py``'s
    ``run_update`` computes and applies the value loss's gradient on
    its own, entirely independent of ``value_coef``. The field is kept
    in this dataclass (not removed) purely for backward compatibility
    with existing configs/tests/checkpoints that already carry it, and
    because ``ppo_total_loss`` (``src/policies/ppo/loss.py``) -- a
    combined-loss helper validated in isolation by P3's own tests,
    never called by ``run_update`` -- still accepts a ``value_coef``
    argument for that standalone use. It has NO EFFECT on the actual
    training loop's dynamics; do not tune it expecting a training-time
    effect, and do not add it to a hyperparameter sweep's list of
    active dimensions.
    """

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
    """Legacy V0 per-decision-step cost. ``enabled=False`` (Reward V1's
    setting) makes this component contribute exactly 0.0 to the reward
    total regardless of ``real_decision_step``/``auto_execution_step``'s
    configured values -- V1 replaces this flat per-step cost with the
    ``RewardDecisionRegularizerConfig`` switching-only regularizer
    (outputs/reward_v1_spec/REWARD_V1_SPEC_FINAL.md Section 13, this
    session's correctness patch: V1's total must be exactly Terminal +
    Safety + Progress + Decision, never a fifth legacy component)."""

    real_decision_step: float
    auto_execution_step: float
    enabled: bool = True


@dataclasses.dataclass(frozen=True)
class RewardSafetyConfig:
    """Reward V1 Safety component (outputs/reward_v1_spec/
    REWARD_V1_SPEC_FINAL.md Section 5). Target-front TTC/Gap only in the
    current approved scope -- ``use_target_rear``/``use_source_front``
    are kept as explicit fields (not silently hardcoded) so a future
    config revision can opt back in without a code change, but
    ``src.rewards.merge_reward`` must raise if either is ever set
    ``true`` (not yet implemented -- see that module's docstring)."""

    weight: float
    ttc_danger_s: float
    ttc_safe_s: float
    gap_sufficient_m: float
    use_target_front: bool
    use_target_rear: bool
    use_source_front: bool


@dataclasses.dataclass(frozen=True)
class RewardProgressConfig:
    """Reward V1 Progress component. ``use_gamma=False`` selects the
    plain (non-discounted) potential-difference form
    ``Phi(s')-Phi(s)`` -- see REWARD_V1_SPEC_FINAL.md Section 6/17 for
    why this is a deliberate simplification, not the formal RL-theory
    potential-based-shaping form. A future ``use_gamma=True`` is
    reserved but not yet implemented (``src.rewards.merge_reward`` must
    raise if set)."""

    weight: float
    use_gamma: bool
    d_m_initial_epsilon_m: float


@dataclasses.dataclass(frozen=True)
class RewardDecisionRegularizerConfig:
    """Reward V1 Decision component -- an action-switching/chattering
    regularizer, distinct from (and additive with) the unchanged V0
    ``RewardDecisionCost``."""

    weight: float
    switching_only: bool


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    """One fully-parsed reward config (``configs/reward.yaml``).

    ``safety``/``progress``/``decision`` are ``Optional`` at the
    dataclass level (``src.rewards.merge_reward`` treats a ``None``
    component as contributing exactly 0.0) but the final canonical
    config always populates all three.
    """

    reward_version: str
    terminal: RewardTerminalTable
    decision_cost: RewardDecisionCost
    source_path: str
    safety: Optional[RewardSafetyConfig] = None
    progress: Optional[RewardProgressConfig] = None
    decision: Optional[RewardDecisionRegularizerConfig] = None


def load_ppo_config(path: str = "configs/ppo/train.yaml") -> PPOConfig:
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


def load_reward_config(path: str = "configs/reward.yaml") -> RewardConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    terminal = raw["terminal"]
    decision_cost = raw["decision_cost"]

    safety_raw = raw.get("safety")
    progress_raw = raw.get("progress")
    decision_raw = raw.get("decision")

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
            enabled=bool(decision_cost.get("enabled", True)),
        ),
        source_path=path,
        safety=(
            RewardSafetyConfig(
                weight=float(safety_raw["weight"]),
                ttc_danger_s=float(safety_raw["ttc_danger_s"]),
                ttc_safe_s=float(safety_raw["ttc_safe_s"]),
                gap_sufficient_m=float(safety_raw["gap_sufficient_m"]),
                use_target_front=bool(safety_raw["use_target_front"]),
                use_target_rear=bool(safety_raw["use_target_rear"]),
                use_source_front=bool(safety_raw["use_source_front"]),
            )
            if safety_raw is not None
            else None
        ),
        progress=(
            RewardProgressConfig(
                weight=float(progress_raw["weight"]),
                use_gamma=bool(progress_raw["use_gamma"]),
                d_m_initial_epsilon_m=float(progress_raw["d_m_initial_epsilon_m"]),
            )
            if progress_raw is not None
            else None
        ),
        decision=(
            RewardDecisionRegularizerConfig(
                weight=float(decision_raw["weight"]),
                switching_only=bool(decision_raw["switching_only"]),
            )
            if decision_raw is not None
            else None
        ),
    )
