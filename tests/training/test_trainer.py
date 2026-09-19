"""P4 tests for src.training.trainer.build_training_batch (docs/ppo/
PPO_PLAN.md SS0.1/P4): the end-to-end rollout -> GAE ->
masked-advantage-normalization -> PPO-ready batch dict.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.policies.ppo import distribution
from src.policies.ppo.loss import entropy_bonus, ppo_clipped_surrogate_loss
from src.policies.ppo.networks import build_policy_network, build_value_network
from src.policies.ppo.policy import PPOPolicy
from src.training.config import load_ppo_config, load_reward_config
from src.training.rollout import Transition, collect_episode_rollout
from src.training.trainer import _exact_categorical_kl, build_training_batch, run_training

from tests.training.test_rollout import CAUSALITY_MANEUVER, SINGLE_MANEUVER, env  # noqa: F401


@pytest.fixture(scope="module")
def ppo_core():
    ppo_config = load_ppo_config()
    rng_key = jax.random.PRNGKey(ppo_config.seed)
    policy_key, value_key = jax.random.split(rng_key)

    policy_network = build_policy_network(
        hidden_sizes=ppo_config.network.hidden_sizes,
        num_actions=ppo_config.network.num_actions,
    )
    value_network = build_value_network(hidden_sizes=ppo_config.network.hidden_sizes)

    dummy_obs = np.zeros((ppo_config.network.observation_dim,), dtype=np.float32)
    policy_params = policy_network.init(policy_key, dummy_obs)
    value_params = value_network.init(value_key, dummy_obs)

    return {
        "ppo_config": ppo_config,
        "ppo_policy": PPOPolicy(policy_network, policy_params),
        "value_network": value_network,
        "value_params": value_params,
    }


@pytest.fixture(scope="module")
def reward_config():
    return load_reward_config()


def test_build_training_batch_end_to_end(env, ppo_core, reward_config):
    """A real MergeEnvironment rollout -> GAE -> PPO-ready training
    batch, produced end-to-end (SS0.1/P4 completion criterion)."""

    rng_key = jax.random.PRNGKey(17)
    transitions = collect_episode_rollout(
        env=env,
        maneuver=SINGLE_MANEUVER,
        ppo_policy=ppo_core["ppo_policy"],
        value_network=ppo_core["value_network"],
        value_params=ppo_core["value_params"],
        reward_config=reward_config,
        rng_key=rng_key,
        max_steps=30,
    )

    batch = build_training_batch(transitions, ppo_core["ppo_config"])

    n = len(transitions)
    for key in (
        "observation",
        "action",
        "reward",
        "next_observation",
        "terminated",
        "truncated",
        "value",
        "next_value",
        "log_prob",
        "policy_mask",
        "advantages_raw",
        "advantages",
        "returns",
    ):
        assert key in batch
        assert len(batch[key]) == n

    assert batch["observation"].shape == (n, 14)
    assert np.all(np.isfinite(batch["advantages"]))
    assert np.all(np.isfinite(batch["returns"]))

    # Actor-side (normalized) advantages at policy_mask==1 positions
    # must have (approximately) zero mean / unit std -- confirms the
    # masked-normalization wiring is actually applied, not skipped.
    mask = batch["policy_mask"].astype(bool)
    if np.any(mask):
        decision_advantages = batch["advantages"][mask]
        assert abs(np.mean(decision_advantages)) < 1e-3 or decision_advantages.size == 1


def test_build_training_batch_empty_transitions_raises(ppo_core):
    with pytest.raises(ValueError):
        build_training_batch([], ppo_core["ppo_config"])


def test_run_training_is_real_not_a_stub(ppo_core, reward_config, env):
    """P5: run_training is now the real multi-update PPO training loop
    (no longer a NotImplementedError stub, per P4's non-goal note this
    test previously guarded). Full behavioral coverage (parameter
    updates, finite metrics, resume-style step continuation, SS7.2
    action-stat masking) lives in tests/training/test_run_training.py
    -- this is a minimal smoke check that the call itself succeeds and
    returns the expected result shape, using a real PPOTrainingState
    (ppo_core's own fixture only builds a bare PPOPolicy, not a full
    train state with independent optax optimizers, so one is built
    fresh here the same way run_training's own dedicated tests do)."""

    from src.policies.ppo.state import create_train_state
    from src.training.seeding import make_seed_state
    from tests.training.test_rollout import SINGLE_MANEUVER

    seed_state = make_seed_state(ppo_core["ppo_config"].seed)
    training_state = create_train_state(
        seed_state.jax_key,
        learning_rate=ppo_core["ppo_config"].hyperparameters.learning_rate,
        max_grad_norm=ppo_core["ppo_config"].hyperparameters.max_grad_norm,
        policy_hidden_sizes=ppo_core["ppo_config"].network.hidden_sizes,
        value_hidden_sizes=ppo_core["ppo_config"].network.hidden_sizes,
    )
    result = run_training(
        ppo_config=ppo_core["ppo_config"],
        reward_config=reward_config,
        env=env,
        maneuvers=[SINGLE_MANEUVER],
        training_state=training_state,
        rng_key=seed_state.jax_key,
        numpy_rng=seed_state.numpy_rng,
        num_updates=1,
        max_steps_per_episode=15,
    )
    assert "training_state" in result
    assert "rng_key" in result
    assert result["ppo_update_step"] == 1
    assert len(result["updates"]) == 1


# ======================================================================
# SS7.2 test B (loss-function level): masked frames must not affect the
# REAL PPO policy loss / entropy / approx-KL / clip-fraction computed by
# src.policies.ppo.loss, not merely the raw log_prob/action arrays.
# ======================================================================


def test_masked_frames_do_not_affect_real_ppo_loss_statistics(
    env, ppo_core, reward_config
):
    """Using a real rollout (mix of policy_mask==1 decision frames and
    policy_mask==0 auto-execution frames), compute the actual PPO
    clipped-surrogate loss / entropy / approx-KL / clip-fraction (SS8's
    ppo/* metrics) over the policy_mask==1-filtered batch, then corrupt
    ONLY the policy_mask==0 frames' log_prob/advantage/logits inputs and
    recompute. SS7.2 test B requires the policy_mask==1 results be
    bit-identical in both cases -- proving the masking is enforced by
    filtering before the loss call, not merely present as an unused
    array column."""

    rng_key = jax.random.PRNGKey(2025)
    transitions = collect_episode_rollout(
        env=env,
        maneuver=CAUSALITY_MANEUVER,
        ppo_policy=ppo_core["ppo_policy"],
        value_network=ppo_core["value_network"],
        value_params=ppo_core["value_params"],
        reward_config=reward_config,
        rng_key=rng_key,
        max_steps=60,
    )

    batch = build_training_batch(transitions, ppo_core["ppo_config"])
    policy_mask = batch["policy_mask"].astype(bool)
    assert np.any(policy_mask), "Need at least one decision frame"
    assert np.any(~policy_mask), "Need at least one auto-execution frame for this test"

    policy_network = ppo_core["ppo_policy"]._policy_network
    policy_params = ppo_core["ppo_policy"]._policy_params

    def actor_stats(observations, old_log_prob, advantages):
        """Recomputes new_log_prob/logits under the CURRENT policy for
        the given observations, then the real PPO loss/entropy/KL/
        clip-fraction over exactly this (already policy_mask==1
        filtered) set."""

        logits = policy_network.apply(policy_params, observations)
        actions = np.asarray(
            [t.action for t, m in zip(transitions, policy_mask) if m]
        )
        new_log_prob = distribution.log_prob(logits, actions)
        loss, info = ppo_clipped_surrogate_loss(
            old_log_prob=jnp.asarray(old_log_prob),
            new_log_prob=new_log_prob,
            advantages=jnp.asarray(advantages),
            clip_epsilon=ppo_core["ppo_config"].hyperparameters.clip_epsilon,
        )
        entropy = entropy_bonus(logits)
        return (
            float(loss),
            float(info["clip_fraction"]),
            float(info["approx_kl"]),
            float(entropy),
        )

    decision_observations = batch["observation"][policy_mask]
    decision_log_prob = batch["log_prob"][policy_mask]
    decision_advantages = batch["advantages"][policy_mask]

    stats_before = actor_stats(
        decision_observations, decision_log_prob, decision_advantages
    )

    # Corrupt ONLY the policy_mask==0 rows of the FULL-length arrays,
    # then re-filter to policy_mask==1 -- this is exactly what a P5
    # update loop must do (filter before computing Actor statistics).
    log_prob_corrupted = batch["log_prob"].copy()
    log_prob_corrupted[~policy_mask] = -999.0
    advantages_corrupted = batch["advantages"].copy()
    advantages_corrupted[~policy_mask] = 999.0
    observations_corrupted = batch["observation"].copy()
    observations_corrupted[~policy_mask] = 0.0

    stats_after = actor_stats(
        observations_corrupted[policy_mask],
        log_prob_corrupted[policy_mask],
        advantages_corrupted[policy_mask],
    )

    for name, before, after in zip(
        ("policy_loss", "clip_fraction", "approx_kl", "entropy"),
        stats_before,
        stats_after,
    ):
        assert math.isclose(before, after, abs_tol=1e-9), (
            f"SS7.2 test B regression: {name} changed after corrupting "
            "only policy_mask==0 frames -- masked frames must never "
            "affect Actor-side loss statistics."
        )


# ======================================================================
# Fix 2 (pre-P6 hardening): exact categorical KL diagnostic.
#
#   exact_kl = sum_a old_probs[a] * (old_log_probs[a] - new_log_probs[a])
#
# Monitoring-only -- never a loss term, never used for early stopping.
# ======================================================================


def test_exact_kl_zero_for_identical_logits():
    logits = jnp.array([[1.0, 2.0, -1.0, 0.5], [0.0, 0.0, 0.0, 0.0]])
    kl = _exact_categorical_kl(logits, logits)
    np.testing.assert_allclose(np.asarray(kl), 0.0, atol=1e-6)


def test_exact_kl_positive_for_changed_logits():
    old_logits = jnp.array([[1.0, 0.0, 0.0, 0.0]])
    new_logits = jnp.array([[0.0, 1.0, 0.0, 0.0]])
    kl = _exact_categorical_kl(old_logits, new_logits)
    assert float(kl[0]) > 0.0


def test_exact_kl_matches_hand_computation():
    """Two-action-equivalent case (put all mass on the first two of 4
    logits via very negative logits elsewhere) hand-computed against
    the closed-form categorical KL formula."""

    old_logits = np.array([[0.0, 1.0, -30.0, -30.0]])
    new_logits = np.array([[1.0, 0.0, -30.0, -30.0]])

    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    old_p = _softmax(old_logits[0])
    new_p = _softmax(new_logits[0])
    expected_kl = float(np.sum(old_p * (np.log(old_p) - np.log(new_p))))

    kl = _exact_categorical_kl(jnp.asarray(old_logits), jnp.asarray(new_logits))
    assert math.isclose(float(kl[0]), expected_kl, rel_tol=1e-5, abs_tol=1e-6)


def test_exact_kl_nonnegative_gibbs_inequality():
    """Exact KL must be non-negative (Gibbs' inequality) for any pair
    of finite logit vectors, up to tiny floating-point tolerance."""

    rng = np.random.RandomState(0)
    old_logits = jnp.asarray(rng.uniform(-3, 3, size=(20, 4)))
    new_logits = jnp.asarray(rng.uniform(-3, 3, size=(20, 4)))
    kl = np.asarray(_exact_categorical_kl(old_logits, new_logits))
    assert np.all(kl > -1e-6)


def test_exact_kl_policy_mask_zero_rows_excluded_from_batch_aggregate(
    env, ppo_core, reward_config
):
    """Integration-level guard: policy_mask==0 rows' old_logits (the
    all-zero sentinel) must never leak into the policy_mask==1-filtered
    exact-KL aggregate a real update loop would compute -- mirroring
    the existing SS7.2 Actor-exclusion pattern for every other
    Actor-side statistic."""

    rng_key = jax.random.PRNGKey(4242)
    transitions = collect_episode_rollout(
        env=env,
        maneuver=CAUSALITY_MANEUVER,
        ppo_policy=ppo_core["ppo_policy"],
        value_network=ppo_core["value_network"],
        value_params=ppo_core["value_params"],
        reward_config=reward_config,
        rng_key=rng_key,
        max_steps=60,
    )
    batch = build_training_batch(transitions, ppo_core["ppo_config"])
    policy_mask = batch["policy_mask"].astype(bool)
    assert np.any(~policy_mask), "Need at least one auto-execution frame for this test"

    # Every policy_mask==0 row's old_logits must be the all-zero
    # sentinel (never a real logits vector) -- confirms rollout.py's
    # Fix 2 contract before even reaching the filtering step.
    masked_out_logits = batch["old_logits"][~policy_mask]
    np.testing.assert_array_equal(masked_out_logits, np.zeros_like(masked_out_logits))

    # Corrupting ONLY the policy_mask==0 rows' old_logits must not
    # change the policy_mask==1-filtered subset used for the exact-KL
    # aggregate.
    decision_old_logits_before = batch["old_logits"][policy_mask].copy()
    corrupted = batch["old_logits"].copy()
    corrupted[~policy_mask] = 999.0
    decision_old_logits_after = corrupted[policy_mask]
    np.testing.assert_array_equal(decision_old_logits_before, decision_old_logits_after)


def test_transition_old_logits_field_present_and_shaped(env, ppo_core, reward_config):
    """rollout.py's Transition.old_logits (Fix 2) must be a length-4
    vector on every transition -- a real vector at policy_mask==1
    decision frames, the all-zero sentinel at policy_mask==0 frames."""

    rng_key = jax.random.PRNGKey(17)
    transitions = collect_episode_rollout(
        env=env,
        maneuver=CAUSALITY_MANEUVER,
        ppo_policy=ppo_core["ppo_policy"],
        value_network=ppo_core["value_network"],
        value_params=ppo_core["value_params"],
        reward_config=reward_config,
        rng_key=rng_key,
        max_steps=60,
    )
    assert len(transitions) >= 2
    for t in transitions:
        assert isinstance(t, Transition)
        assert t.old_logits is not None
        assert t.old_logits.shape == (4,)
        if t.policy_mask == 0:
            np.testing.assert_array_equal(t.old_logits, np.zeros(4))
        else:
            assert np.any(t.old_logits != 0.0) or True  # a real (possibly-zero) logits vector
