"""PPO train state (policy/value params + optimizer) (docs/ppo/
PPO_PLAN.md SS0.1 P3, SS10).

P1 scope: structural skeleton only. This is the module that will own
the concrete pytree structure referenced (as ``Any``) by
``src.training.checkpoint.CheckpointPayload.policy_params`` /
``value_params`` / ``optimizer_state`` -- real ``flax``/``optax``
train-state construction lands in P3.
"""


def create_train_state(*args, **kwargs):
    """Builds the initial PPO train state (policy params, value params,
    optimizer state) from a seed and network/optimizer config.

    P1 skeleton: not yet implemented (lands in P3).
    """

    raise NotImplementedError(
        "PPO train state construction lands in P3 (docs/ppo/PPO_PLAN.md "
        "SS0.1/P3)."
    )
