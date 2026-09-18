"""PPO trainer loop: rollout -> GAE -> PPO update -> logging (docs/ppo/
PPO_PLAN.md SS0.1 P5).

P1 scope: structural skeleton only. Real end-to-end wiring (the loop
that ``scripts/train_ppo.py`` / ``scripts/smoke_train_ppo.py`` will
eventually drive) lands in P5, once P2 (reward/W&B), P3 (PPO core), and
P4 (rollout/GAE) all exist. This module intentionally does not import
``src.rewards`` / ``src.policies.ppo`` / ``src.training.rollout`` /
``src.training.gae`` real logic yet -- it only fixes where the trainer
entry point will live.
"""

from src.training.config import PPOConfig, RewardConfig


def run_training(ppo_config: PPOConfig, reward_config: RewardConfig, *args, **kwargs):
    """Runs the full PPO training loop (rollout -> GAE -> update ->
    checkpoint -> W&B logging) for one config.

    P1 skeleton: not yet implemented. Real training loop lands in P5
    per PPO_PLAN.md SS0.1/P5, once P2-P4 exist.
    """

    raise NotImplementedError(
        "The PPO training loop lands in P5 (docs/ppo/PPO_PLAN.md "
        "SS0.1/P5), once Reward V0 (P2), PPO core (P3), and rollout/GAE "
        "(P4) all exist. P1 only fixes this entry point's signature."
    )
