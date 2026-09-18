"""P1 import checks for the reward package (docs/ppo/PPO_PLAN.md
SS0.1/P1). Real reward computation is P2's job -- these tests only
confirm the module structure exists and is importable, and that the
P1-stub functions raise ``NotImplementedError`` rather than silently
returning a fake value.
"""

import pytest

from src.rewards import merge_reward, reward_wrapper
from src.training.config import load_reward_config


def test_merge_reward_module_imports():
    assert hasattr(merge_reward, "compute_reward")


def test_reward_wrapper_module_imports():
    assert hasattr(reward_wrapper, "MergeRewardWrapper")


def test_compute_reward_is_not_yet_implemented():
    reward_config = load_reward_config("configs/reward/merge_reward_v0.yaml")
    with pytest.raises(NotImplementedError):
        merge_reward.compute_reward(
            reward_config=reward_config,
            termination_reason="success",
            is_policy_step=True,
        )


def test_reward_wrapper_compute_is_not_yet_implemented():
    reward_config = load_reward_config("configs/reward/merge_reward_v0.yaml")
    wrapper = reward_wrapper.MergeRewardWrapper(reward_config)
    with pytest.raises(NotImplementedError):
        wrapper.compute(termination_reason="success", is_policy_step=True)
