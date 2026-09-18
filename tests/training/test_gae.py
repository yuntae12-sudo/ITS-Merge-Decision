"""P4 tests for src.training.gae (docs/ppo/PPO_PLAN.md SS0.1/P4, SS7.2).

Covers: handcrafted-trajectory GAE matching a hand-computed value,
terminal-bootstrap handling, truncation handling, masked advantage
normalization invariance (SS7.2 test A), and no-NaN/inf.
"""

import math

import numpy as np
import pytest

from src.training.gae import compute_gae, masked_mean_std, normalize_advantages_masked

GAMMA = 0.99
LAMBDA = 0.95


def test_handcrafted_two_step_gae_matches_hand_computation():
    """2-step trajectory, hand-computed by the GAE backward recursion:

    rewards = [1.0, 2.0], values = [0.5, 0.5], next_values = [0.5, 0.0]
    Step 1 (last): terminated=True -> bootstrap_mask=0
        delta_1 = r_1 + gamma*V(s_2)*0 - V(s_1) = 2.0 + 0 - 0.5 = 1.5
        A_1 = delta_1 = 1.5
    Step 0: terminated=False -> bootstrap_mask=1
        delta_0 = r_0 + gamma*V(s_1) - V(s_0) = 1.0 + 0.99*0.5 - 0.5 = 0.995
        A_0 = delta_0 + gamma*lambda*1*A_1 = 0.995 + 0.99*0.95*1.5
            = 0.995 + 1.41075 = 2.40575
    returns = advantages + values -> [2.90575, 2.0]
    """

    rewards = [1.0, 2.0]
    values = [0.5, 0.5]
    next_values = [0.5, 0.0]
    terminated = [False, True]
    truncated = [False, False]

    result = compute_gae(
        rewards, values, next_values, terminated, truncated, GAMMA, LAMBDA
    )

    expected_advantage_1 = 1.5
    expected_advantage_0 = 0.995 + GAMMA * LAMBDA * expected_advantage_1
    expected_returns = [
        expected_advantage_0 + values[0],
        expected_advantage_1 + values[1],
    ]

    assert math.isclose(result.advantages[1], expected_advantage_1, abs_tol=1e-9)
    assert math.isclose(result.advantages[0], expected_advantage_0, abs_tol=1e-9)
    assert math.isclose(result.returns[0], expected_returns[0], abs_tol=1e-9)
    assert math.isclose(result.returns[1], expected_returns[1], abs_tol=1e-9)


def test_handcrafted_three_step_gae_matches_hand_computation():
    """3-step trajectory, all non-terminal/non-truncated except the
    last, which truncates (still bootstraps).

    rewards = [1.0, 1.0, 1.0], values = [0.0, 0.0, 0.0],
    next_values = [0.0, 0.0, 10.0], truncated=[F, F, T]

    Step 2 (last, truncated -> bootstrap_mask=1 since NOT terminated):
        delta_2 = 1.0 + 0.99*10.0 - 0.0 = 10.9
        A_2 = 10.9
    Step 1:
        delta_1 = 1.0 + 0.99*0.0 - 0.0 = 1.0
        A_1 = 1.0 + 0.99*0.95*10.9 = 1.0 + 10.25595 = 11.25595
    Step 0:
        delta_0 = 1.0 + 0.99*0.0 - 0.0 = 1.0
        A_0 = 1.0 + 0.99*0.95*11.25595 = 1.0 + 10.58810025 = 11.58810025
    """

    rewards = [1.0, 1.0, 1.0]
    values = [0.0, 0.0, 0.0]
    next_values = [0.0, 0.0, 10.0]
    terminated = [False, False, False]
    truncated = [False, False, True]

    result = compute_gae(
        rewards, values, next_values, terminated, truncated, GAMMA, LAMBDA
    )

    a2 = 1.0 + GAMMA * 10.0
    a1 = 1.0 + GAMMA * LAMBDA * a2
    a0 = 1.0 + GAMMA * LAMBDA * a1

    assert math.isclose(result.advantages[2], a2, abs_tol=1e-9)
    assert math.isclose(result.advantages[1], a1, abs_tol=1e-9)
    assert math.isclose(result.advantages[0], a0, abs_tol=1e-9)


def test_terminal_step_does_not_bootstrap():
    """A terminated step's advantage must not depend on next_value at
    all -- changing next_value on a terminated final step changes
    nothing (bootstrap_mask=0 zeroes it out)."""

    rewards = [5.0]
    values = [1.0]
    terminated = [True]
    truncated = [False]

    result_a = compute_gae(rewards, values, [999.0], terminated, truncated, GAMMA, LAMBDA)
    result_b = compute_gae(rewards, values, [-999.0], terminated, truncated, GAMMA, LAMBDA)

    assert math.isclose(result_a.advantages[0], result_b.advantages[0], abs_tol=1e-9)
    # delta = reward - value (no bootstrap term) = 5.0 - 1.0 = 4.0
    assert math.isclose(result_a.advantages[0], 4.0, abs_tol=1e-9)


def test_truncated_step_does_bootstrap():
    """A truncated (but not terminated) step's advantage MUST depend
    on next_value -- this is the required truncation-vs-termination
    distinction (SS0.1/P4)."""

    rewards = [5.0]
    values = [1.0]
    terminated = [False]
    truncated = [True]

    result_a = compute_gae(rewards, values, [10.0], terminated, truncated, GAMMA, LAMBDA)
    result_b = compute_gae(rewards, values, [0.0], terminated, truncated, GAMMA, LAMBDA)

    assert not math.isclose(result_a.advantages[0], result_b.advantages[0], abs_tol=1e-9)
    expected_a = 5.0 + GAMMA * 10.0 - 1.0
    expected_b = 5.0 + GAMMA * 0.0 - 1.0
    assert math.isclose(result_a.advantages[0], expected_a, abs_tol=1e-9)
    assert math.isclose(result_b.advantages[0], expected_b, abs_tol=1e-9)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        compute_gae([1.0, 2.0], [0.0], [0.0, 0.0], [False, False], [False, False], GAMMA, LAMBDA)


def test_no_nan_inf_in_gae_output():
    rng = np.random.RandomState(0)
    n = 50
    rewards = rng.uniform(-1, 1, size=n)
    values = rng.uniform(-1, 1, size=n)
    next_values = rng.uniform(-1, 1, size=n)
    terminated = np.zeros(n, dtype=bool)
    terminated[-1] = True
    truncated = np.zeros(n, dtype=bool)

    result = compute_gae(rewards, values, next_values, terminated, truncated, GAMMA, LAMBDA)
    assert np.all(np.isfinite(result.advantages))
    assert np.all(np.isfinite(result.returns))


# ======================================================================
# SS7.2 test A: Actor-side normalization invariance to masked frames
# ======================================================================


def test_masked_normalization_invariant_to_masked_frame_values():
    """Holding policy_mask==1 advantages fixed, changing policy_mask==0
    frames' advantage VALUES must not change the normalization mean/
    std, nor the resulting normalized decision-frame advantages."""

    advantages = np.array([1.0, -2.0, 0.5, 3.0, -1.0])
    policy_mask = np.array([1, 0, 1, 0, 1])

    mean_a, std_a = masked_mean_std(advantages, policy_mask)
    normalized_a = normalize_advantages_masked(advantages, policy_mask)

    # Change ONLY the policy_mask == 0 frames' advantage values.
    advantages_perturbed = advantages.copy()
    advantages_perturbed[1] = 999.0
    advantages_perturbed[3] = -999.0

    mean_b, std_b = masked_mean_std(advantages_perturbed, policy_mask)
    normalized_b = normalize_advantages_masked(advantages_perturbed, policy_mask)

    assert math.isclose(mean_a, mean_b, abs_tol=1e-12)
    assert math.isclose(std_a, std_b, abs_tol=1e-12)

    decision_indices = np.where(policy_mask == 1)[0]
    for idx in decision_indices:
        assert math.isclose(normalized_a[idx], normalized_b[idx], abs_tol=1e-9)


def test_masked_normalization_changes_when_decision_frames_change():
    """Sanity counterpart: the statistic DOES change when the
    policy_mask==1 set's own values change (confirms the invariance
    test above isn't trivially passing because nothing ever changes)."""

    advantages = np.array([1.0, -2.0, 0.5, 3.0, -1.0])
    policy_mask = np.array([1, 0, 1, 0, 1])

    mean_a, _ = masked_mean_std(advantages, policy_mask)

    advantages_changed = advantages.copy()
    advantages_changed[0] = 100.0  # a policy_mask==1 frame
    mean_b, _ = masked_mean_std(advantages_changed, policy_mask)

    assert not math.isclose(mean_a, mean_b, abs_tol=1e-6)


def test_masked_normalization_matches_manual_computation():
    advantages = np.array([2.0, 4.0, 6.0])
    policy_mask = np.array([1, 1, 0])

    decision_values = advantages[:2]
    expected_mean = np.mean(decision_values)
    expected_std = np.std(decision_values)

    normalized = normalize_advantages_masked(advantages, policy_mask, eps=1e-8)
    assert math.isclose(
        normalized[0], (2.0 - expected_mean) / (expected_std + 1e-8), abs_tol=1e-9
    )
    assert math.isclose(
        normalized[1], (4.0 - expected_mean) / (expected_std + 1e-8), abs_tol=1e-9
    )


def test_normalize_advantages_all_masked_out_raises():
    advantages = np.array([1.0, 2.0])
    policy_mask = np.array([0, 0])
    with pytest.raises(ValueError):
        normalize_advantages_masked(advantages, policy_mask)


def test_normalize_advantages_shape_mismatch_raises():
    with pytest.raises(ValueError):
        normalize_advantages_masked([1.0, 2.0], [1, 0, 1])
