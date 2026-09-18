"""P1 import checks for the PPO policy package (docs/ppo/PPO_PLAN.md
SS0.1/P1). Real PPO algorithm code lands in P3 -- these tests only
confirm module structure/imports and the fixed action-index mapping
regression (SS11), which must hold from P1 onward since it is a pure
constant, not algorithm logic.
"""

import pytest

from src.environment.behavior_action import BehaviorAction
from src.policies.ppo import distribution, loss, networks, policy, state


def test_networks_module_imports():
    assert networks.OBSERVATION_DIM == 14
    assert networks.NUM_ACTIONS == 4
    assert tuple(networks.POLICY_HIDDEN_SIZES) == (256, 64, 32)
    assert tuple(networks.VALUE_HIDDEN_SIZES) == (256, 64, 32)


def test_distribution_module_imports():
    assert hasattr(distribution, "sample_action")
    assert hasattr(distribution, "deterministic_action")


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


def test_policy_module_imports():
    assert hasattr(policy, "PPOPolicy")


def test_state_module_imports():
    assert hasattr(state, "create_train_state")


def test_ppo_core_stubs_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        networks.build_policy_network()
    with pytest.raises(NotImplementedError):
        networks.build_value_network()
    with pytest.raises(NotImplementedError):
        distribution.sample_action()
    with pytest.raises(NotImplementedError):
        loss.ppo_clipped_surrogate_loss()
    with pytest.raises(NotImplementedError):
        state.create_train_state()
    with pytest.raises(NotImplementedError):
        policy.PPOPolicy()
