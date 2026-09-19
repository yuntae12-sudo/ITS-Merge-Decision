"""Import/structure checks for the PPO policy package (docs/ppo/
PPO_PLAN.md SS0.1/P1, P3). As of P3 the real algorithm is implemented
(see ``tests/policies/test_ppo_core.py`` for behavioral coverage) --
this file only confirms module structure/imports and the fixed
action-index mapping regression (SS11), which must hold unconditionally
since it is a pure constant, not algorithm logic.
"""

from src.environment.behavior_action import BehaviorAction
from src.policies.ppo import distribution, loss, networks, policy, state


def test_networks_module_imports():
    assert networks.OBSERVATION_DIM == 14
    assert networks.NUM_ACTIONS == 4
    assert tuple(networks.POLICY_HIDDEN_SIZES) == (256, 64, 32)
    assert tuple(networks.VALUE_HIDDEN_SIZES) == (256, 64, 32)
    assert hasattr(networks, "build_policy_network")
    assert hasattr(networks, "build_value_network")


def test_distribution_module_imports():
    assert hasattr(distribution, "sample_action")
    assert hasattr(distribution, "deterministic_action")
    assert hasattr(distribution, "log_prob")
    assert hasattr(distribution, "entropy")


def test_action_index_mapping_regression():
    """docs/ppo/PPO_PLAN.md SS11: the categorical action index ->
    BehaviorAction mapping must never drift."""

    assert distribution.ACTION_INDEX_TO_BEHAVIOR == {
        0: BehaviorAction.KEEP,
        1: BehaviorAction.FOLLOW,
        2: BehaviorAction.MERGE,
        3: BehaviorAction.STOP,
    }


def test_loss_module_imports():
    assert hasattr(loss, "ppo_clipped_surrogate_loss")
    assert hasattr(loss, "value_loss")
    assert hasattr(loss, "entropy_bonus")
    assert hasattr(loss, "ppo_ratio")
    assert hasattr(loss, "ppo_total_loss")


def test_policy_module_imports():
    assert hasattr(policy, "PPOPolicy")


def test_state_module_imports():
    assert hasattr(state, "create_train_state")
    assert hasattr(state, "PPOTrainingState")
