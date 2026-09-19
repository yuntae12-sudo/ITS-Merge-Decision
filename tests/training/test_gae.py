"""P4 tests for src.training.gae (docs/ppo/PPO_PLAN.md SS0.1/P4, SS7.2).

Covers: handcrafted-trajectory GAE matching a hand-computed value,
terminal-bootstrap handling, truncation handling, masked advantage
normalization invariance (SS7.2 test A), and no-NaN/inf.
"""

import math

import numpy as np
import pytest

from src.training.gae import (
    compute_gae,
    compute_gae_segmented,
    masked_mean_std,
    normalize_advantages_masked,
)

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


# ======================================================================
# Fix 1 (pre-P6 hardening): episode-aware GAE correctness via
# compute_gae_segmented. collect_rollout concatenates multiple
# episodes' transitions into one flat list; a naive single compute_gae
# call over that flat list lets a later episode's advantage leak
# backward into an earlier truncated/cutoff episode purely because the
# backward recursion has no notion of an episode boundary other than
# `terminated`. compute_gae_segmented fixes this by running compute_gae
# independently per contiguous episode_ids run.
# ======================================================================


def test_segmented_gae_truncated_episode_a_unaffected_by_episode_b_rewards():
    """Required test 1: Episode A truncated + Episode B follows --
    changing B's rewards must not change A's advantages."""

    # Episode A: 2 steps, truncated at the end (bootstraps from
    # next_value but must NOT see episode B's rewards/values at all).
    rewards_a = [1.0, 2.0]
    values_a = [0.5, 0.5]
    next_values_a = [0.5, 3.0]
    terminated_a = [False, False]
    truncated_a = [False, True]

    # Episode B: 2 steps, terminates normally.
    rewards_b = [10.0, -10.0]
    values_b = [1.0, 1.0]
    next_values_b = [1.0, 0.0]
    terminated_b = [False, True]
    truncated_b = [False, False]

    episode_ids = ["A", "A", "B", "B"]

    def _run(rewards_b_variant):
        rewards = rewards_a + rewards_b_variant
        values = values_a + values_b
        next_values = next_values_a + next_values_b
        terminated = terminated_a + terminated_b
        truncated = truncated_a + truncated_b
        return compute_gae_segmented(
            rewards, values, next_values, terminated, truncated,
            episode_ids, GAMMA, LAMBDA,
        )

    result_1 = _run(rewards_b)
    result_2 = _run([999.0, -999.0])  # drastically different Episode B rewards

    # Episode A's advantages/returns (indices 0, 1) must be byte-identical
    # regardless of Episode B's rewards.
    assert math.isclose(result_1.advantages[0], result_2.advantages[0], abs_tol=1e-12)
    assert math.isclose(result_1.advantages[1], result_2.advantages[1], abs_tol=1e-12)
    assert math.isclose(result_1.returns[0], result_2.returns[0], abs_tol=1e-12)
    assert math.isclose(result_1.returns[1], result_2.returns[1], abs_tol=1e-12)

    # Episode A's own result must equal what an independent compute_gae
    # call on ONLY episode A's sub-trajectory would produce (the direct
    # correctness statement, not just an invariance check).
    standalone_a = compute_gae(
        rewards_a, values_a, next_values_a, terminated_a, truncated_a, GAMMA, LAMBDA
    )
    assert math.isclose(result_1.advantages[0], standalone_a.advantages[0], abs_tol=1e-9)
    assert math.isclose(result_1.advantages[1], standalone_a.advantages[1], abs_tol=1e-9)

    # Sanity: Episode B's OWN advantages DO change with its own rewards
    # (confirms the test isn't vacuous).
    assert not math.isclose(result_1.advantages[2], result_2.advantages[2], abs_tol=1e-6)


def test_segmented_gae_truncated_final_step_next_value_matters():
    """Required test 2: on the truncated final step of an episode
    segment, changing next_value DOES change that episode's last
    delta/return (bootstrap applies right up to the segment boundary)."""

    rewards = [1.0, 2.0]
    values = [0.5, 0.5]
    terminated = [False, False]
    truncated = [False, True]
    episode_ids = ["A", "A"]

    result_a = compute_gae_segmented(
        rewards, values, [0.5, 10.0], terminated, truncated, episode_ids, GAMMA, LAMBDA
    )
    result_b = compute_gae_segmented(
        rewards, values, [0.5, -10.0], terminated, truncated, episode_ids, GAMMA, LAMBDA
    )

    assert not math.isclose(result_a.advantages[1], result_b.advantages[1], abs_tol=1e-6)
    assert not math.isclose(result_a.returns[1], result_b.returns[1], abs_tol=1e-6)


def test_segmented_gae_true_terminated_step_next_value_does_not_matter():
    """Required test 3: on a TRUE terminated step, changing next_value
    must not change the advantage/return at all (no bootstrap)."""

    rewards = [1.0, 2.0]
    values = [0.5, 0.5]
    terminated = [False, True]
    truncated = [False, False]
    episode_ids = ["A", "A"]

    result_a = compute_gae_segmented(
        rewards, values, [0.5, 999.0], terminated, truncated, episode_ids, GAMMA, LAMBDA
    )
    result_b = compute_gae_segmented(
        rewards, values, [0.5, -999.0], terminated, truncated, episode_ids, GAMMA, LAMBDA
    )

    assert math.isclose(result_a.advantages[1], result_b.advantages[1], abs_tol=1e-9)
    assert math.isclose(result_a.returns[1], result_b.returns[1], abs_tol=1e-9)


def test_segmented_gae_multi_episode_batch_equals_independent_concatenation():
    """Required test 4: a multi-episode batch's segmented GAE output
    must equal the concatenation of INDEPENDENTLY computed per-episode
    compute_gae results, for every episode in the batch."""

    rng = np.random.RandomState(3)

    def _make_episode(n, final_terminated):
        rewards = list(rng.uniform(-1, 1, size=n))
        values = list(rng.uniform(-1, 1, size=n))
        next_values = list(rng.uniform(-1, 1, size=n))
        terminated = [False] * (n - 1) + [final_terminated]
        truncated = [False] * (n - 1) + [not final_terminated]
        return rewards, values, next_values, terminated, truncated

    episodes = [
        _make_episode(3, final_terminated=True),
        _make_episode(2, final_terminated=False),  # truncated
        _make_episode(4, final_terminated=True),
    ]

    all_rewards, all_values, all_next_values = [], [], []
    all_terminated, all_truncated, all_episode_ids = [], [], []
    expected_advantages, expected_returns = [], []

    for i, (rewards, values, next_values, terminated, truncated) in enumerate(episodes):
        all_rewards += rewards
        all_values += values
        all_next_values += next_values
        all_terminated += terminated
        all_truncated += truncated
        all_episode_ids += [f"ep{i}"] * len(rewards)

        standalone = compute_gae(
            rewards, values, next_values, terminated, truncated, GAMMA, LAMBDA
        )
        expected_advantages.extend(standalone.advantages.tolist())
        expected_returns.extend(standalone.returns.tolist())

    result = compute_gae_segmented(
        all_rewards, all_values, all_next_values, all_terminated, all_truncated,
        all_episode_ids, GAMMA, LAMBDA,
    )

    np.testing.assert_allclose(result.advantages, expected_advantages, atol=1e-9)
    np.testing.assert_allclose(result.returns, expected_returns, atol=1e-9)


def test_segmented_gae_artificial_cutoff_bootstraps_but_does_not_leak():
    """Required test 5: an artificial max-steps rollout_cutoff (neither
    terminated nor truncated) must still bootstrap from next_value
    (like truncation), and must not leak its trace into the NEXT
    episode in the same flat batch."""

    # Episode A: 2 steps, ends via an ARTIFICIAL rollout cutoff (the
    # trainer's own max_steps ran out) -- terminated=False,
    # truncated=False, rollout_cutoff=True on the last step.
    rewards_a = [1.0, 2.0]
    values_a = [0.5, 0.5]
    terminated_a = [False, False]
    truncated_a = [False, False]
    rollout_cutoff_a = [False, True]

    rewards_b = [5.0, -5.0]
    values_b = [1.0, 1.0]
    terminated_b = [False, True]
    truncated_b = [False, False]
    rollout_cutoff_b = [False, False]

    episode_ids = ["A", "A", "B", "B"]

    def _run(next_value_last_a, rewards_b_variant):
        next_values = [0.5, next_value_last_a, 1.0, 0.0]
        rewards = rewards_a + rewards_b_variant
        values = values_a + values_b
        terminated = terminated_a + terminated_b
        truncated = truncated_a + truncated_b
        rollout_cutoff = rollout_cutoff_a + rollout_cutoff_b
        return compute_gae_segmented(
            rewards, values, next_values, terminated, truncated, episode_ids,
            GAMMA, LAMBDA, rollout_cutoff=rollout_cutoff,
        )

    # (a) Bootstraps: changing next_value on the cutoff step changes A's
    # last-step advantage/return (same treatment as truncation).
    result_hi = _run(next_value_last_a=100.0, rewards_b_variant=rewards_b)
    result_lo = _run(next_value_last_a=-100.0, rewards_b_variant=rewards_b)
    assert not math.isclose(
        result_hi.advantages[1], result_lo.advantages[1], abs_tol=1e-3
    )

    # (b) No leakage: changing Episode B's rewards must not change
    # Episode A's advantages/returns at all.
    result_b_variant = _run(next_value_last_a=100.0, rewards_b_variant=[999.0, -999.0])
    assert math.isclose(
        result_hi.advantages[0], result_b_variant.advantages[0], abs_tol=1e-12
    )
    assert math.isclose(
        result_hi.advantages[1], result_b_variant.advantages[1], abs_tol=1e-12
    )
    assert math.isclose(
        result_hi.returns[1], result_b_variant.returns[1], abs_tol=1e-12
    )


# ======================================================================
# Pre-P6 hardening Fix 1: episode-aware GAE correctness
# (compute_gae_segmented) -- the 5 required tests.
# ======================================================================


def test_segmented_truncated_episode_a_unaffected_by_episode_b_rewards():
    """Test 1: Episode A truncated, Episode B follows in the flat
    transition list -- changing Episode B's rewards/values arbitrarily
    must NOT change Episode A's computed advantages/returns at all."""

    # Episode A: 2 steps, truncated at the end (bootstraps).
    rewards_a = [1.0, 2.0]
    values_a = [0.5, 0.5]
    next_values_a = [0.5, 3.0]
    terminated_a = [False, False]
    truncated_a = [False, True]

    # Episode B: 2 steps, ends in true termination.
    rewards_b = [10.0, -5.0]
    values_b = [1.0, 1.0]
    next_values_b = [1.0, 0.0]
    terminated_b = [False, True]
    truncated_b = [False, False]

    episode_ids = ["A", "A", "B", "B"]

    def _run(rewards_b_variant, values_b_variant, next_values_b_variant):
        return compute_gae_segmented(
            rewards=rewards_a + rewards_b_variant,
            values=values_a + values_b_variant,
            next_values=next_values_a + next_values_b_variant,
            terminated=terminated_a + terminated_b,
            truncated=truncated_a + truncated_b,
            episode_ids=episode_ids,
            gamma=GAMMA,
            gae_lambda=LAMBDA,
        )

    result_1 = _run(rewards_b, values_b, next_values_b)
    # Arbitrarily perturb EVERY one of Episode B's rewards/values/next_values.
    result_2 = _run([999.0, -888.0], [777.0, -666.0], [555.0, -444.0])

    # Episode A's advantages/returns (indices 0, 1) must be byte-identical.
    np.testing.assert_array_equal(result_1.advantages[:2], result_2.advantages[:2])
    np.testing.assert_array_equal(result_1.returns[:2], result_2.returns[:2])

    # Sanity: Episode B's OWN advantages did in fact change (the test
    # isn't vacuously passing because nothing ever changes).
    assert not np.allclose(result_1.advantages[2:], result_2.advantages[2:])

    # Cross-check against independently calling compute_gae on Episode
    # A alone.
    standalone_a = compute_gae(
        rewards_a, values_a, next_values_a, terminated_a, truncated_a, GAMMA, LAMBDA
    )
    np.testing.assert_array_equal(result_1.advantages[:2], standalone_a.advantages)
    np.testing.assert_array_equal(result_1.returns[:2], standalone_a.returns)


def test_segmented_truncated_final_step_bootstrap_alive():
    """Test 2: a truncated final step's next_value DOES change that
    episode's own last delta/return (bootstrap alive) -- same guarantee
    as compute_gae's own truncation test, now through the segmented
    multi-episode entry point."""

    episode_ids = ["A", "A"]
    rewards = [1.0, 5.0]
    values = [0.5, 1.0]
    terminated = [False, False]
    truncated = [False, True]

    result_a = compute_gae_segmented(
        rewards, values, [0.5, 10.0], terminated, truncated, episode_ids, GAMMA, LAMBDA
    )
    result_b = compute_gae_segmented(
        rewards, values, [0.5, 0.0], terminated, truncated, episode_ids, GAMMA, LAMBDA
    )

    assert not math.isclose(result_a.advantages[1], result_b.advantages[1], abs_tol=1e-9)
    assert not math.isclose(result_a.returns[1], result_b.returns[1], abs_tol=1e-9)


def test_segmented_true_terminated_step_no_bootstrap():
    """Test 3: a TRUE terminated step's advantage does NOT depend on
    next_value -- no bootstrap past a true terminal state, even through
    the segmented multi-episode entry point."""

    episode_ids = ["A", "A"]
    rewards = [1.0, 5.0]
    values = [0.5, 1.0]
    terminated = [False, True]
    truncated = [False, False]

    result_a = compute_gae_segmented(
        rewards, values, [0.5, 999.0], terminated, truncated, episode_ids, GAMMA, LAMBDA
    )
    result_b = compute_gae_segmented(
        rewards, values, [0.5, -999.0], terminated, truncated, episode_ids, GAMMA, LAMBDA
    )

    assert math.isclose(result_a.advantages[1], result_b.advantages[1], abs_tol=1e-9)
    assert math.isclose(result_a.returns[1], result_b.returns[1], abs_tol=1e-9)


def test_segmented_multi_episode_batch_equals_concatenation_of_independent_results():
    """Test 4: a multi-episode concatenated batch's compute_gae_segmented
    result must equal exactly the concatenation of independently
    computing compute_gae on each episode's own sub-trajectory alone."""

    rng = np.random.RandomState(3)

    def _random_episode(n, final_terminated, final_truncated):
        rewards = rng.uniform(-1, 1, size=n).tolist()
        values = rng.uniform(-1, 1, size=n).tolist()
        next_values = rng.uniform(-1, 1, size=n).tolist()
        terminated = [False] * (n - 1) + [final_terminated]
        truncated = [False] * (n - 1) + [final_truncated]
        return rewards, values, next_values, terminated, truncated

    ep_a = _random_episode(4, final_terminated=True, final_truncated=False)
    ep_b = _random_episode(3, final_terminated=False, final_truncated=True)
    ep_c = _random_episode(5, final_terminated=True, final_truncated=False)

    all_rewards = ep_a[0] + ep_b[0] + ep_c[0]
    all_values = ep_a[1] + ep_b[1] + ep_c[1]
    all_next_values = ep_a[2] + ep_b[2] + ep_c[2]
    all_terminated = ep_a[3] + ep_b[3] + ep_c[3]
    all_truncated = ep_a[4] + ep_b[4] + ep_c[4]
    episode_ids = ["A"] * 4 + ["B"] * 3 + ["C"] * 5

    combined = compute_gae_segmented(
        all_rewards, all_values, all_next_values, all_terminated, all_truncated,
        episode_ids, GAMMA, LAMBDA,
    )

    independent_results = [
        compute_gae(*ep[:5], GAMMA, LAMBDA) for ep in (ep_a, ep_b, ep_c)
    ]
    expected_advantages = np.concatenate([r.advantages for r in independent_results])
    expected_returns = np.concatenate([r.returns for r in independent_results])

    np.testing.assert_array_equal(combined.advantages, expected_advantages)
    np.testing.assert_array_equal(combined.returns, expected_returns)


def test_segmented_artificial_cutoff_bootstraps_but_does_not_leak():
    """Test 5: an artificial max-steps rollout_cutoff (NOT a real
    terminated/truncated environment signal) still bootstraps from
    next_value for its own episode (like truncation), AND does not let
    the following episode's rewards/values leak backward into it."""

    # Episode A: 2 steps, the SECOND is an artificial cutoff (the
    # trainer's max_steps loop ran out -- terminated=truncated=False,
    # rollout_cutoff=True).
    rewards_a = [1.0, 2.0]
    values_a = [0.5, 0.5]
    next_values_a = [0.5, 4.0]
    terminated_a = [False, False]
    truncated_a = [False, False]
    rollout_cutoff_a = [False, True]

    # Episode B follows immediately in the flat list.
    rewards_b = [7.0]
    values_b = [1.0]
    next_values_b = [1.0]
    terminated_b = [True]
    truncated_b = [False]
    rollout_cutoff_b = [False]

    episode_ids = ["A", "A", "B"]

    def _run(next_value_a_last):
        return compute_gae_segmented(
            rewards=rewards_a[:1] + [rewards_a[1]] + rewards_b,
            values=values_a + values_b,
            next_values=[next_values_a[0], next_value_a_last] + next_values_b,
            terminated=terminated_a + terminated_b,
            truncated=truncated_a + truncated_b,
            episode_ids=episode_ids,
            gamma=GAMMA,
            gae_lambda=LAMBDA,
            rollout_cutoff=rollout_cutoff_a + rollout_cutoff_b,
        )

    # Bootstrap alive: changing the cutoff step's next_value changes
    # its own advantage/return (same as a truncated step would).
    result_1 = _run(4.0)
    result_2 = _run(-100.0)
    assert not math.isclose(result_1.advantages[1], result_2.advantages[1], abs_tol=1e-9)

    # No leakage: Episode A's advantages (indices 0, 1) must be
    # unaffected by Episode B's own reward, regardless of what Episode
    # A's cutoff step's next_value is.
    result_b_perturbed = compute_gae_segmented(
        rewards=rewards_a + [999.0],
        values=values_a + values_b,
        next_values=next_values_a + next_values_b,
        terminated=terminated_a + terminated_b,
        truncated=truncated_a + truncated_b,
        episode_ids=episode_ids,
        gamma=GAMMA,
        gae_lambda=LAMBDA,
        rollout_cutoff=rollout_cutoff_a + rollout_cutoff_b,
    )
    result_b_original = compute_gae_segmented(
        rewards=rewards_a + rewards_b,
        values=values_a + values_b,
        next_values=next_values_a + next_values_b,
        terminated=terminated_a + terminated_b,
        truncated=truncated_a + truncated_b,
        episode_ids=episode_ids,
        gamma=GAMMA,
        gae_lambda=LAMBDA,
        rollout_cutoff=rollout_cutoff_a + rollout_cutoff_b,
    )
    np.testing.assert_array_equal(
        result_b_perturbed.advantages[:2], result_b_original.advantages[:2]
    )
    np.testing.assert_array_equal(
        result_b_perturbed.returns[:2], result_b_original.returns[:2]
    )

    # Cross-check against calling compute_gae directly on Episode A
    # alone with the SAME rollout_cutoff flag.
    standalone_a = compute_gae(
        rewards_a, values_a, next_values_a, terminated_a, truncated_a,
        GAMMA, LAMBDA, rollout_cutoff=rollout_cutoff_a,
    )
    np.testing.assert_array_equal(result_b_original.advantages[:2], standalone_a.advantages)
    np.testing.assert_array_equal(result_b_original.returns[:2], standalone_a.returns)


def test_compute_gae_rollout_cutoff_defaults_to_false_backward_compatible():
    """compute_gae's new rollout_cutoff parameter defaults to all-False
    -- an existing caller that never passes it gets EXACTLY the P4
    behavior (bootstrap determined only by terminated/truncated)."""

    rewards = [1.0, 2.0, 3.0]
    values = [0.1, 0.2, 0.3]
    next_values = [0.2, 0.3, 0.0]
    terminated = [False, False, True]
    truncated = [False, False, False]

    with_default = compute_gae(rewards, values, next_values, terminated, truncated, GAMMA, LAMBDA)
    with_explicit_false = compute_gae(
        rewards, values, next_values, terminated, truncated, GAMMA, LAMBDA,
        rollout_cutoff=[False, False, False],
    )
    np.testing.assert_array_equal(with_default.advantages, with_explicit_false.advantages)
    np.testing.assert_array_equal(with_default.returns, with_explicit_false.returns)
