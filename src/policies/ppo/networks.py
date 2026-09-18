"""PPO policy/value networks (docs/ppo/PPO_PLAN.md SS6, SS0.1 P3).

Fixed architecture (not tuned in P0-P5):

    Policy: 14 -> 256 -> 64 -> 32 -> 4 logits, tanh activations
    Value:  14 -> 256 -> 64 -> 32 -> 1,        tanh activations

Policy and value networks are separate Flax modules with independent
parameters (no shared trunk) per SS7.2's Actor-vs-Critic separation of
concerns. Implemented with ``flax.linen`` (P0 confirmed ``flax==0.10.7``
already installed, compatible with the pinned ``jax==0.6.2``/
``jaxlib==0.6.2`` stack -- no upgrade performed).
"""

from typing import Sequence

import flax.linen as nn
import jax

# Fixed baseline architecture (PPO_PLAN.md SS6). Not tuned in P0-P5.
POLICY_HIDDEN_SIZES: Sequence[int] = (256, 64, 32)
VALUE_HIDDEN_SIZES: Sequence[int] = (256, 64, 32)
OBSERVATION_DIM = 14
NUM_ACTIONS = 4
ACTIVATION = "tanh"


class PolicyNetwork(nn.Module):
    """Actor network: 14D observation -> 4 categorical logits.

    ``hidden_sizes`` defaults to the fixed PPO_PLAN.md SS6 architecture
    (256, 64, 32) with tanh activations between every layer, including
    between the last hidden layer and the final logits layer (SS6
    specifies tanh activations throughout; the final logits layer
    itself has no activation applied to its output, matching a
    standard categorical-logits head).
    """

    hidden_sizes: Sequence[int] = POLICY_HIDDEN_SIZES
    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, observation):
        x = observation
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            x = nn.tanh(x)
        logits = nn.Dense(self.num_actions)(x)
        return logits


class ValueNetwork(nn.Module):
    """Critic network: 14D observation -> scalar state-value estimate."""

    hidden_sizes: Sequence[int] = VALUE_HIDDEN_SIZES

    @nn.compact
    def __call__(self, observation):
        x = observation
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            x = nn.tanh(x)
        value = nn.Dense(1)(x)
        # Squeeze the trailing size-1 dimension: (..., 1) -> (...,)
        return value.squeeze(-1)


def build_policy_network(
    hidden_sizes: Sequence[int] = POLICY_HIDDEN_SIZES,
    num_actions: int = NUM_ACTIONS,
) -> PolicyNetwork:
    """Builds the PPO policy (Actor) network module."""

    return PolicyNetwork(hidden_sizes=tuple(hidden_sizes), num_actions=num_actions)


def build_value_network(
    hidden_sizes: Sequence[int] = VALUE_HIDDEN_SIZES,
) -> ValueNetwork:
    """Builds the PPO value (Critic) network module."""

    return ValueNetwork(hidden_sizes=tuple(hidden_sizes))


def init_policy_params(
    policy_network: PolicyNetwork, rng_key: jax.Array, observation_dim: int = OBSERVATION_DIM
):
    """Initializes policy network parameters given a PRNG key."""

    dummy_obs = jax.numpy.zeros((observation_dim,))
    return policy_network.init(rng_key, dummy_obs)


def init_value_params(
    value_network: ValueNetwork, rng_key: jax.Array, observation_dim: int = OBSERVATION_DIM
):
    """Initializes value network parameters given a PRNG key."""

    dummy_obs = jax.numpy.zeros((observation_dim,))
    return value_network.init(rng_key, dummy_obs)
