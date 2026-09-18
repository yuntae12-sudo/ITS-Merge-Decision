"""Generalized Advantage Estimation (docs/ppo/PPO_PLAN.md SS0.1 P4,
SS6, SS7.2).

P4 implementation. GAE is computed over the FULL physical trajectory
(all frames, regardless of ``policy_mask`` -- SS7.2), with truncation
and true termination distinguished for bootstrapping, using the
baseline (not-tuned) ``gamma=0.99`` / ``gae_lambda=0.95`` (SS6).

Actor-side advantage normalization (mean/std) is computed only over
``policy_mask == 1`` frames -- that masking happens here via
``normalize_advantages_masked`` (kept in this module, not
``loss.py``, per P1's original placement decision, since GAE tests
SS7.2 A/B need to exercise this exact masking behavior directly), but
the GAE/return computation itself (``compute_gae``) always uses the
whole trajectory -- that is Critic-side and is never masked.

Standard backward-recursion GAE (Schulman et al. 2015, "High-
Dimensional Continuous Control Using Generalized Advantage
Estimation", eq. 16), matching the V-Max reference pattern
(PPO_PLAN.md SS3):

    delta_t   = r_t + gamma * V(s_{t+1}) * mask_t - V(s_t)
    A_t       = delta_t + gamma * lambda * mask_t * A_{t+1}
    return_t  = A_t + V(s_t)

where ``mask_t`` is the BOOTSTRAP mask for step t: ``0`` if step t
TERMINATED (true termination -- no value flows from beyond a terminal
state), ``1`` otherwise (including a TRUNCATED step, which still
bootstraps from ``next_value`` -- SS0.1/P4's required truncation
test). This bootstrap mask is unrelated to ``policy_mask``.
"""

from typing import Any, NamedTuple, Sequence

import numpy as np


class GAEResult(NamedTuple):
    advantages: np.ndarray
    returns: np.ndarray


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    next_values: Sequence[float],
    terminated: Sequence[bool],
    truncated: Sequence[bool],
    gamma: float,
    gae_lambda: float,
) -> GAEResult:
    """Computes GAE advantages and returns over one full physical
    trajectory (all frames, regardless of ``policy_mask`` -- SS7.2).

    Args:
        rewards: per-step reward, shape ``(T,)``.
        values: ``V(s_t)`` -- the value network's prediction for the
            observation BEFORE step t was taken, shape ``(T,)``.
        next_values: ``V(s_{t+1})`` -- the value network's prediction
            for the observation AFTER step t, shape ``(T,)``. For a
            terminated step this is not used for bootstrapping (masked
            to 0 internally) but must still be a finite number (e.g.
            whatever the value network happened to predict for the
            terminal observation, or 0.0) -- callers are not required
            to special-case it.
        terminated: per-step ``terminated`` flag from ``env.step``,
            shape ``(T,)``. A True at step t means step t reached a
            true terminal state (SUCCESS/COLLISION/OFFROAD) -- no
            bootstrap value flows past it.
        truncated: per-step ``truncated`` flag from ``env.step``,
            shape ``(T,)``. A True at step t means step t hit the
            episode horizon -- UNLIKE ``terminated``, this DOES
            bootstrap from ``next_values[t]`` (the trajectory was cut
            off, not actually over).
        gamma: discount factor (PPO_PLAN.md SS6: ``0.99``).
        gae_lambda: GAE lambda (PPO_PLAN.md SS6: ``0.95``).

    Returns:
        ``GAEResult(advantages, returns)``, each shape ``(T,)``.
    """

    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)

    num_steps = rewards.shape[0]
    if not (
        values.shape[0] == num_steps
        and next_values.shape[0] == num_steps
        and terminated.shape[0] == num_steps
        and truncated.shape[0] == num_steps
    ):
        raise ValueError(
            "compute_gae: rewards/values/next_values/terminated/truncated "
            "must all have the same length (one full physical trajectory). "
            f"Got lengths {rewards.shape[0]}, {values.shape[0]}, "
            f"{next_values.shape[0]}, {terminated.shape[0]}, "
            f"{truncated.shape[0]}."
        )

    # Bootstrap mask: 0.0 on true termination (no value beyond a
    # terminal state), 1.0 otherwise -- including truncation, which
    # DOES bootstrap from next_values (SS0.1/P4 required test).
    bootstrap_mask = np.where(terminated, 0.0, 1.0)

    advantages = np.zeros(num_steps, dtype=np.float64)
    gae_running = 0.0
    for t in reversed(range(num_steps)):
        delta = (
            rewards[t]
            + gamma * next_values[t] * bootstrap_mask[t]
            - values[t]
        )
        gae_running = delta + gamma * gae_lambda * bootstrap_mask[t] * gae_running
        advantages[t] = gae_running

    returns = advantages + values

    if not (np.all(np.isfinite(advantages)) and np.all(np.isfinite(returns))):
        raise ValueError(
            "compute_gae produced non-finite advantages/returns -- check "
            "input rewards/values for NaN/inf."
        )

    return GAEResult(advantages=advantages, returns=returns)


def normalize_advantages_masked(
    advantages: Sequence[float], policy_mask: Sequence[int], eps: float = 1e-8
) -> np.ndarray:
    """Normalizes advantages using only ``policy_mask == 1`` frames'
    mean/std (Actor-side normalization, SS7.2).

    Per SS7.2 test A: holding the set of ``policy_mask == 1``
    advantages fixed, changing ``policy_mask == 0`` frames' advantage
    VALUES must not change the computed mean/std, nor the resulting
    normalized values at ``policy_mask == 1`` positions. This is
    achieved by computing mean/std ONLY over the ``policy_mask == 1``
    subset -- masked-out entries never enter the reduction.

    Args:
        advantages: full-trajectory advantages, shape ``(T,)`` (as
            produced by ``compute_gae`` over the WHOLE trajectory).
        policy_mask: per-step ``policy_mask`` (1 = real decision step,
            0 = auto-executed MERGE-commitment step), shape ``(T,)``.
        eps: numerical-stability epsilon added to the std before
            dividing.

    Returns:
        An array shape ``(T,)`` where ``policy_mask == 1`` positions
        hold the normalized advantage (using the masked mean/std) and
        ``policy_mask == 0`` positions hold the ORIGINAL (unnormalized)
        advantage value unchanged (they are never consumed by any
        Actor-side computation anyway, per SS7.2, so their value here
        is a don't-care that we leave untouched rather than zeroing,
        to avoid implying any semantic meaning for a masked slot).
    """

    advantages = np.asarray(advantages, dtype=np.float64)
    policy_mask = np.asarray(policy_mask)

    if advantages.shape != policy_mask.shape:
        raise ValueError(
            "normalize_advantages_masked: advantages and policy_mask must "
            f"have the same shape, got {advantages.shape} vs "
            f"{policy_mask.shape}."
        )

    mask_bool = policy_mask.astype(bool)
    if not np.any(mask_bool):
        raise ValueError(
            "normalize_advantages_masked: policy_mask has no True entries "
            "-- cannot compute Actor-side normalization statistics over an "
            "empty set of decision frames."
        )

    decision_advantages = advantages[mask_bool]
    mean = float(np.mean(decision_advantages))
    std = float(np.std(decision_advantages))

    normalized = advantages.copy()
    normalized[mask_bool] = (decision_advantages - mean) / (std + eps)

    if not np.all(np.isfinite(normalized[mask_bool])):
        raise ValueError(
            "normalize_advantages_masked produced non-finite normalized "
            "advantages at a policy_mask==1 position."
        )

    return normalized


def masked_mean_std(advantages: Sequence[float], policy_mask: Sequence[int]):
    """Returns ``(mean, std)`` of ``advantages`` computed only over
    ``policy_mask == 1`` positions -- exposed standalone (in addition
    to being used internally by ``normalize_advantages_masked``) so
    SS7.2 test A can assert the statistic itself is unaffected by
    changes to masked-out frames, not just the final normalized
    output."""

    advantages = np.asarray(advantages, dtype=np.float64)
    policy_mask = np.asarray(policy_mask).astype(bool)
    decision_advantages = advantages[policy_mask]
    return float(np.mean(decision_advantages)), float(np.std(decision_advantages))
