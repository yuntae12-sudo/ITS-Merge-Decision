"""Import/structure checks for the reward package (docs/ppo/
PPO_PLAN.md SS0.1/P1, P2). Real reward computation landed in P2 --
behavioral tests for that live in ``tests/rewards/test_merge_reward.py``.
This file only confirms module structure/imports.
"""

from src.rewards import merge_reward, reward_wrapper
from src.training.config import load_reward_config


def test_merge_reward_module_imports():
    assert hasattr(merge_reward, "compute_reward")


def test_reward_wrapper_module_imports():
    assert hasattr(reward_wrapper, "MergeRewardWrapper")


def test_compute_reward_is_implemented():
    reward_config = load_reward_config("configs/reward.yaml")
    value = merge_reward.compute_reward(
        reward_config=reward_config,
        termination_reason="success",
        is_policy_step=True,
    )
    # Legacy decision_cost is disabled in the final reward config, so a
    # bare success/is_policy_step case is exactly the terminal reward.
    assert value == 1.0


def test_reward_wrapper_compute_is_implemented():
    reward_config = load_reward_config("configs/reward.yaml")
    wrapper = reward_wrapper.MergeRewardWrapper(reward_config)
    value = wrapper.compute(termination_reason="success", is_policy_step=True)
    assert value == 1.0
