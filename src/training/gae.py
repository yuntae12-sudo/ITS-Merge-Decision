"""Generalized Advantage Estimation (docs/ppo/PPO_PLAN.md SS0.1 P4,
SS6, SS7.2).

P1 scope: structural skeleton only. Real implementation lands in P4.
When implemented, GAE MUST be computed over the FULL physical
trajectory (all frames, regardless of ``policy_mask`` -- SS7.2), with
truncation and true termination distinguished for bootstrapping, using
the baseline (not-tuned) ``gamma=0.99`` / ``gae_lambda=0.95``
(SS6). Actor-side advantage normalization (mean/std) is computed only
over ``policy_mask == 1`` frames -- that masking happens in the
Actor/loss code (``src.policies.ppo.loss``), not here; this module's
job is Critic-side GAE/return over the whole trajectory.
"""

from typing import Any, Sequence


def compute_gae(*args, **kwargs) -> Any:
    """Computes GAE advantages and returns over a full physical
    trajectory.

    P1 skeleton: not yet implemented (lands in P4).
    """

    raise NotImplementedError(
        "GAE computation lands in P4 (docs/ppo/PPO_PLAN.md SS0.1/P4)."
    )


def normalize_advantages_masked(advantages: Sequence[float], policy_mask: Sequence[int]):
    """Normalizes advantages using only ``policy_mask == 1`` frames'
    statistics (Actor-side normalization per SS7.2).

    P1 skeleton: not yet implemented (lands in P4). Kept here as a
    named stub (rather than only inside ``loss.py``) since P4's GAE
    tests (SS7.2 tests A/B) need to exercise this exact masking
    behavior directly.
    """

    raise NotImplementedError(
        "Masked advantage normalization lands in P4 (docs/ppo/PPO_PLAN.md "
        "SS0.1/P4, SS7.2)."
    )
