"""Categorical action distribution over ``BehaviorAction`` (docs/ppo/
PPO_PLAN.md SS0.1 P3, SS11).

P1 scope: structural skeleton only. Fixes the action-index mapping
regression (SS11) so it can be imported and asserted from P1 onward,
even though sampling/log_prob/entropy land in P3.

    0 -> BehaviorAction.KEEP
    1 -> BehaviorAction.FOLLOW
    2 -> BehaviorAction.MERGE
    3 -> BehaviorAction.STOP
"""

from src.environment.behavior_action import BehaviorAction

# Fixed categorical action index -> BehaviorAction mapping (SS11). This
# must never drift from src.environment.behavior_action.BehaviorAction.
ACTION_INDEX_TO_BEHAVIOR = {
    0: BehaviorAction.KEEP,
    1: BehaviorAction.FOLLOW,
    2: BehaviorAction.MERGE,
    3: BehaviorAction.STOP,
}


def sample_action(*args, **kwargs):
    """Samples a stochastic action from the PPO categorical
    distribution.

    P1 skeleton: not yet implemented. Real sampling/log_prob/entropy
    logic lands in P3 per PPO_PLAN.md SS0.1/P3.
    """

    raise NotImplementedError(
        "Categorical action sampling lands in P3 (docs/ppo/PPO_PLAN.md "
        "SS0.1/P3). P1 only fixes the action-index mapping."
    )


def deterministic_action(*args, **kwargs):
    """Returns the deterministic (argmax) action for inference.

    P1 skeleton: not yet implemented (lands in P3).
    """

    raise NotImplementedError(
        "Deterministic inference lands in P3 (docs/ppo/PPO_PLAN.md "
        "SS0.1/P3). P1 only fixes the action-index mapping."
    )
