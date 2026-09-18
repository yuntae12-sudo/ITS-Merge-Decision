"""PPO clipped-surrogate loss (docs/ppo/PPO_PLAN.md SS0.1 P3, SS6, SS7.2).

P1 scope: structural skeleton only. Real implementation (PPO ratio,
clipped surrogate objective, value loss, entropy regularization,
gradient clipping) lands in P3. Per SS7.2, when implemented this module
must apply policy loss / entropy / approx-KL / clip-fraction /
advantage-normalization statistics over ``policy_mask == 1`` frames
only, while value loss uses the full physical trajectory.

Baseline (not-tuned) coefficients live in ``configs/ppo/ppo_base.yaml``
(SS6): ``clip_epsilon=0.2``, ``value_coef=0.5``, ``entropy_coef=0.01``.
"""


def ppo_clipped_surrogate_loss(*args, **kwargs):
    """Computes the PPO clipped surrogate policy loss.

    P1 skeleton: not yet implemented (lands in P3).
    """

    raise NotImplementedError(
        "PPO clipped surrogate loss lands in P3 (docs/ppo/PPO_PLAN.md "
        "SS0.1/P3)."
    )


def value_loss(*args, **kwargs):
    """Computes the PPO value-function loss.

    P1 skeleton: not yet implemented (lands in P3).
    """

    raise NotImplementedError(
        "PPO value loss lands in P3 (docs/ppo/PPO_PLAN.md SS0.1/P3)."
    )


def entropy_bonus(*args, **kwargs):
    """Computes the entropy regularization term.

    P1 skeleton: not yet implemented (lands in P3).
    """

    raise NotImplementedError(
        "PPO entropy bonus lands in P3 (docs/ppo/PPO_PLAN.md SS0.1/P3)."
    )
