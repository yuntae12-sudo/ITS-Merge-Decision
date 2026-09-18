"""PPO policy interface: observation -> action (docs/ppo/PPO_PLAN.md
SS0.1 P3, SS1).

P1 scope: structural skeleton only. This is the module PPO_PLAN.md
SS1's architecture diagram calls "PPO Policy": it consumes the 14D
observation and produces a ``BehaviorAction`` via the categorical
distribution in ``distribution.py``, backed by the networks in
``networks.py``. Real implementation lands in P3.

PPO never outputs acceleration/steering directly -- only the 4-way
categorical ``BehaviorAction`` (SS1). The environment must not know
whether the policy in control is FSM or PPO.
"""


class PPOPolicy:
    """Stochastic/deterministic PPO policy over the 14D observation.

    P1 skeleton: constructor and ``act``/``act_deterministic`` are not
    yet implemented. Real network + distribution wiring lands in P3.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "PPOPolicy construction lands in P3 (docs/ppo/PPO_PLAN.md "
            "SS0.1/P3). P1 only defines this module's structure."
        )

    def act(self, observation, rng_key):
        """Stochastic action selection (training-time)."""

        raise NotImplementedError(
            "PPOPolicy.act lands in P3 (docs/ppo/PPO_PLAN.md SS0.1/P3)."
        )

    def act_deterministic(self, observation):
        """Deterministic (argmax) action selection (inference-time)."""

        raise NotImplementedError(
            "PPOPolicy.act_deterministic lands in P3 (docs/ppo/PPO_PLAN.md "
            "SS0.1/P3)."
        )
