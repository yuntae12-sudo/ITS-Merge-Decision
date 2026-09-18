"""PPO clipped-surrogate loss (docs/ppo/PPO_PLAN.md SS0.1 P3, SS6, SS7.2).

Implements the PPO ratio, clipped surrogate policy objective, value
loss, and entropy regularization, following the V-Max reference
pattern (``vmax/agents/learning/reinforcement/ppo/ppo_factory.py``'s
``_make_loss_fn``) for the math itself, adapted to this project's
discrete 4-way categorical action head (no continuous Gaussian/Beta
distribution is used anywhere).

Per SS7.2, when combined with a real rollout (P4+), policy loss /
entropy / approx-KL / clip-fraction / advantage-normalization
statistics must be computed over ``policy_mask == 1`` frames only,
while value loss uses the full physical trajectory. P3 validates the
loss math itself in isolation (synthetic batches, no ``policy_mask``
plumbing yet -- that lands with the real rollout in P4).

Baseline (not-tuned) coefficients live in ``configs/ppo/ppo_base.yaml``
(SS6): ``clip_epsilon=0.2``, ``value_coef=0.5``, ``entropy_coef=0.01``.
"""

import jax.numpy as jnp

from src.policies.ppo import distribution


def ppo_ratio(old_log_prob, new_log_prob):
    """PPO probability ratio: ``exp(new_log_prob - old_log_prob)``.

    Args:
        old_log_prob: log-probability of the taken action under the
            behavior policy that generated the rollout.
        new_log_prob: log-probability of the same action under the
            current policy parameters.

    Returns:
        The ratio ``r = pi_new(a|s) / pi_old(a|s)``.
    """

    return jnp.exp(new_log_prob - old_log_prob)


def ppo_clipped_surrogate_loss(
    old_log_prob,
    new_log_prob,
    advantages,
    clip_epsilon: float,
):
    """Computes the PPO clipped surrogate policy loss (to be
    minimized -- i.e. already negated from the objective to maximize).

    Args:
        old_log_prob: log-probability under the behavior policy,
            shape ``(batch,)``.
        new_log_prob: log-probability under the current policy,
            shape ``(batch,)``.
        advantages: advantage estimates, shape ``(batch,)``.
        clip_epsilon: clipping radius (PPO_PLAN.md SS6: ``0.2``).

    Returns:
        A tuple ``(loss, info)`` where ``loss`` is the scalar mean
        clipped-surrogate policy loss and ``info`` is a dict with
        diagnostic values (``ratio``, ``clip_fraction``, ``approx_kl``).
    """

    ratio = ppo_ratio(old_log_prob, new_log_prob)
    surrogate_1 = advantages * ratio
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate_2 = advantages * clipped_ratio

    loss = -jnp.mean(jnp.minimum(surrogate_1, surrogate_2))

    clip_fraction = jnp.mean(
        (jnp.abs(ratio - 1.0) > clip_epsilon).astype(jnp.float32)
    )
    # Standard approximate-KL diagnostic (Schulman et al.): E[old - new]
    approx_kl = jnp.mean(old_log_prob - new_log_prob)

    info = {
        "ratio": ratio,
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
    }
    return loss, info


def value_loss(values, returns):
    """Mean-squared-error value-function loss.

    Args:
        values: current value-network predictions, shape ``(batch,)``.
        returns: target returns (e.g. GAE-derived), shape ``(batch,)``.

    Returns:
        Scalar MSE loss (not yet multiplied by ``value_coef`` --
        callers apply the coefficient, matching PPO_PLAN.md SS6's
        separate ``value_coef`` hyperparameter).
    """

    error = returns - values
    return jnp.mean(error * error)


def entropy_bonus(logits):
    """Mean categorical entropy across a batch of logits, for use as
    the entropy regularization term.

    Args:
        logits: shape ``(batch, num_actions)`` or ``(num_actions,)``.

    Returns:
        Scalar mean entropy (nats). Callers subtract
        ``entropy_coef * entropy`` from the total loss (equivalently,
        add ``-entropy_coef * entropy`` -- see ``ppo_total_loss``).
    """

    return jnp.mean(distribution.entropy(logits))


def ppo_total_loss(
    old_log_prob,
    new_log_prob,
    advantages,
    values,
    returns,
    logits,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
):
    """Full PPO loss: clipped surrogate policy loss + value loss -
    entropy bonus.

    Args:
        old_log_prob: shape ``(batch,)``.
        new_log_prob: shape ``(batch,)``.
        advantages: shape ``(batch,)``.
        values: current value predictions, shape ``(batch,)``.
        returns: value targets, shape ``(batch,)``.
        logits: current policy logits, shape ``(batch, num_actions)``.
        clip_epsilon: PPO_PLAN.md SS6 ``0.2``.
        value_coef: PPO_PLAN.md SS6 ``0.5``.
        entropy_coef: PPO_PLAN.md SS6 ``0.01``.

    Returns:
        ``(total_loss, info)`` -- ``info`` carries the individual
        components (``policy_loss``, ``value_loss``, ``entropy``,
        ``entropy_loss``, ``ratio``, ``clip_fraction``, ``approx_kl``)
        for logging (PPO_PLAN.md SS8's ``ppo/*`` metrics).
    """

    policy_loss, surrogate_info = ppo_clipped_surrogate_loss(
        old_log_prob, new_log_prob, advantages, clip_epsilon
    )
    v_loss = value_loss(values, returns)
    entropy = entropy_bonus(logits)
    entropy_loss = -entropy_coef * entropy

    total = policy_loss + value_coef * v_loss + entropy_loss

    info = {
        "policy_loss": policy_loss,
        "value_loss": v_loss,
        "entropy": entropy,
        "entropy_loss": entropy_loss,
        "total_loss": total,
        **surrogate_info,
    }
    return total, info
