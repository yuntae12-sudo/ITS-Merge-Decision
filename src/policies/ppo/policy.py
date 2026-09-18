"""PPO policy interface: observation -> action (docs/ppo/PPO_PLAN.md
SS0.1 P3, SS1).

This is the module PPO_PLAN.md SS1's architecture diagram calls "PPO
Policy": it consumes the 14D observation and produces a
``BehaviorAction`` via the categorical distribution in
``distribution.py``, backed by the network in ``networks.py``.

PPO never outputs acceleration/steering directly -- only the 4-way
categorical ``BehaviorAction`` (SS1). The environment must not know
whether the policy in control is FSM or PPO: this class has no
dependency on ``src.environment`` beyond consuming the frozen
``BehaviorAction`` enum (via ``distribution.ACTION_INDEX_TO_BEHAVIOR``),
which is the required PPO -> Environment direction, never the reverse.
"""

import jax

from src.policies.ppo import distribution
from src.policies.ppo.networks import PolicyNetwork


class PPOPolicy:
    """Stochastic/deterministic PPO policy over the 14D observation."""

    def __init__(self, policy_network: PolicyNetwork, policy_params):
        """Args:
        policy_network: a ``networks.PolicyNetwork`` (or any object
            exposing a Flax-style ``.apply(params, observation)``
            method returning logits of shape ``(num_actions,)`` /
            ``(batch, num_actions)``).
        policy_params: the network's current parameters (a Flax
            variable pytree, as returned by ``.init(...)`` or a
            ``PPOTrainState.params``).
        """

        self._policy_network = policy_network
        self._policy_params = policy_params

    @property
    def params(self):
        return self._policy_params

    def logits(self, observation):
        """Raw categorical logits for ``observation``."""

        return self._policy_network.apply(self._policy_params, observation)

    def act(self, observation, rng_key: jax.Array):
        """Stochastic action selection (training-time).

        Args:
            observation: 14D observation vector (or batch thereof).
            rng_key: JAX PRNG key.

        Returns:
            ``(action_index, log_prob)`` -- ``action_index`` is the
            sampled categorical index in ``0..3`` (§11 mapping applies
            downstream via ``distribution.ACTION_INDEX_TO_BEHAVIOR``),
            ``log_prob`` is its log-probability under ``logits``.
        """

        logits = self.logits(observation)
        action = distribution.sample_action(logits, rng_key)
        log_prob = distribution.log_prob(logits, action)
        return action, log_prob

    def act_deterministic(self, observation):
        """Deterministic (argmax) action selection (inference-time).

        Returns:
            The argmax action index in ``0..3``.
        """

        logits = self.logits(observation)
        return distribution.deterministic_action(logits)
