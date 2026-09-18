"""Categorical action distribution over ``BehaviorAction`` (docs/ppo/
PPO_PLAN.md SS0.1 P3, SS11).

Fixes the action-index mapping (SS11) and implements the categorical
distribution operations PPO needs: stochastic sampling, deterministic
(argmax) inference, log-probability, and entropy. Built directly on
``jax.nn``/``jax.random`` -- no continuous Gaussian/Beta action head is
implemented anywhere (PPO_PLAN.md SS3 explicitly excludes those from
this research).

    0 -> BehaviorAction.KEEP
    1 -> BehaviorAction.FOLLOW
    2 -> BehaviorAction.MERGE
    3 -> BehaviorAction.STOP
"""

import jax
import jax.numpy as jnp

from src.environment.behavior_action import BehaviorAction

# Fixed categorical action index -> BehaviorAction mapping (SS11). This
# must never drift from src.environment.behavior_action.BehaviorAction.
ACTION_INDEX_TO_BEHAVIOR = {
    0: BehaviorAction.KEEP,
    1: BehaviorAction.FOLLOW,
    2: BehaviorAction.MERGE,
    3: BehaviorAction.STOP,
}


def sample_action(logits: jax.Array, rng_key: jax.Array) -> jax.Array:
    """Samples a stochastic action index from the categorical
    distribution parameterized by ``logits``.

    Args:
        logits: unnormalized log-probabilities, shape ``(num_actions,)``
            or ``(batch, num_actions)``.
        rng_key: JAX PRNG key.

    Returns:
        Sampled action index (int32), shape ``()`` or ``(batch,)``.
    """

    return jax.random.categorical(rng_key, logits, axis=-1)


def deterministic_action(logits: jax.Array) -> jax.Array:
    """Returns the deterministic (argmax) action index for inference.

    Args:
        logits: unnormalized log-probabilities, shape ``(num_actions,)``
            or ``(batch, num_actions)``.

    Returns:
        Argmax action index (int32), shape ``()`` or ``(batch,)``.
    """

    return jnp.argmax(logits, axis=-1)


def log_prob(logits: jax.Array, action: jax.Array) -> jax.Array:
    """Log-probability of ``action`` under the categorical distribution
    parameterized by ``logits``.

    Args:
        logits: shape ``(..., num_actions)``.
        action: integer action index/indices, shape ``(...,)``
            (broadcast-compatible with ``logits``'s leading dims).

    Returns:
        Log-probability, shape ``(...,)``.
    """

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    action = jnp.asarray(action, dtype=jnp.int32)
    return jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)


def entropy(logits: jax.Array) -> jax.Array:
    """Entropy of the categorical distribution parameterized by
    ``logits``.

    Args:
        logits: shape ``(..., num_actions)``.

    Returns:
        Entropy (nats), shape ``(...,)``. Always finite and
        non-negative for finite logits.
    """

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


def probs(logits: jax.Array) -> jax.Array:
    """Softmax probabilities for ``logits`` (convenience helper for
    tests/diagnostics -- sums to 1 along the last axis)."""

    return jax.nn.softmax(logits, axis=-1)
