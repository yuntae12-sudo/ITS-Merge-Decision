"""PPO train state (policy/value params + optimizer) (docs/ppo/
PPO_PLAN.md SS0.1 P3, SS10).

Owns the concrete pytree structure referenced (as ``Any``) by
``src.training.checkpoint.CheckpointPayload.policy_params`` /
``value_params`` / ``optimizer_state``. Two independent
``flax.training.train_state.TrainState`` instances (policy, value),
each wrapping its own ``optax.adam`` optimizer with global-norm
gradient clipping -- matching the V-Max reference pattern
(``optax.chain(optax.clip_by_global_norm(...), optax.adam(...))``,
PPO_PLAN.md SS3) but with separate Actor/Critic parameters (SS7.2).
"""

import dataclasses

import jax
import optax
from flax.training import train_state

from src.policies.ppo.networks import (
    OBSERVATION_DIM,
    PolicyNetwork,
    ValueNetwork,
    build_policy_network,
    build_value_network,
)


class PPOTrainState(train_state.TrainState):
    """Thin alias of ``flax.training.train_state.TrainState``. Kept as
    a distinct name so PPO call sites are explicit about which
    train-state flavor they hold, without adding any extra fields
    beyond what ``TrainState`` already provides (``step``, ``params``,
    ``opt_state``, ``apply_fn``, ``tx``)."""


@dataclasses.dataclass(frozen=True)
class PPOTrainingState:
    """Bundles the policy (Actor) and value (Critic) train states, plus
    the network module definitions needed to re-apply them. This is
    the object ``src.training.checkpoint.CheckpointPayload.policy_params``/
    ``value_params``/``optimizer_state`` are populated from (P4/P5 call
    sites extract the respective ``.params``/``.opt_state`` fields)."""

    policy_network: PolicyNetwork
    value_network: ValueNetwork
    policy_state: PPOTrainState
    value_state: PPOTrainState


def _make_optimizer(learning_rate: float, max_grad_norm: float) -> optax.GradientTransformation:
    """Adam optimizer with global-norm gradient clipping, per
    PPO_PLAN.md SS3/SS6 (``learning_rate=3e-4``, clipping applied
    before Adam, matching the V-Max reference pattern)."""

    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate),
    )


def create_train_state(
    rng_key: jax.Array,
    learning_rate: float,
    max_grad_norm: float,
    policy_hidden_sizes=None,
    value_hidden_sizes=None,
    num_actions: int = 4,
    observation_dim: int = OBSERVATION_DIM,
) -> PPOTrainingState:
    """Builds the initial PPO train state: fresh policy/value network
    parameters and their respective Adam-with-clipping optimizer
    states, from one PRNG key and the run's hyperparameters.

    Args:
        rng_key: JAX PRNG key; split internally into a policy-init key
            and a value-init key so the two networks never share init
            randomness.
        learning_rate: Adam learning rate (PPO_PLAN.md SS6: ``3e-4``).
        max_grad_norm: global-norm gradient-clipping threshold
            (PPO_PLAN.md SS6: ``0.5``).
        policy_hidden_sizes: overrides the default policy hidden sizes
            (PPO_PLAN.md SS6: ``(256, 64, 32)``) if provided.
        value_hidden_sizes: overrides the default value hidden sizes
            if provided.
        num_actions: number of categorical actions (PPO_PLAN.md SS6: 4).
        observation_dim: input observation dimensionality
            (PPO_PLAN.md SS2: 14).

    Returns:
        A ``PPOTrainingState`` with freshly initialized policy and
        value ``PPOTrainState``s.
    """

    policy_key, value_key = jax.random.split(rng_key)

    policy_network = build_policy_network(
        hidden_sizes=policy_hidden_sizes or (256, 64, 32),
        num_actions=num_actions,
    )
    value_network = build_value_network(
        hidden_sizes=value_hidden_sizes or (256, 64, 32),
    )

    dummy_obs = jax.numpy.zeros((observation_dim,))
    policy_params = policy_network.init(policy_key, dummy_obs)
    value_params = value_network.init(value_key, dummy_obs)

    optimizer = _make_optimizer(learning_rate, max_grad_norm)

    policy_state = PPOTrainState.create(
        apply_fn=policy_network.apply,
        params=policy_params,
        tx=optimizer,
    )
    value_state = PPOTrainState.create(
        apply_fn=value_network.apply,
        params=value_params,
        tx=_make_optimizer(learning_rate, max_grad_norm),
    )

    return PPOTrainingState(
        policy_network=policy_network,
        value_network=value_network,
        policy_state=policy_state,
        value_state=value_state,
    )
