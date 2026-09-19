"""Behavioral tests for the discrete PPO core (docs/ppo/PPO_PLAN.md
SS0.1/P3, SS11). Validates the PPO algorithm in isolation, using
synthetic/toy inputs only -- no real ``MergeEnvironment`` rollout here
(that integration is P4's job, per PPO_PLAN.md's explicit P3 non-goal).
"""

import jax
import jax.numpy as jnp
import pytest

from src.environment.behavior_action import BehaviorAction
from src.policies.ppo import distribution, loss
from src.policies.ppo.networks import (
    NUM_ACTIONS,
    OBSERVATION_DIM,
    build_policy_network,
    build_value_network,
)
from src.policies.ppo.policy import PPOPolicy
from src.policies.ppo.state import create_train_state

SEED = 0


def _dummy_observation():
    return jnp.arange(OBSERVATION_DIM, dtype=jnp.float32) / 10.0


def _dummy_batch_observation(batch_size=8):
    key = jax.random.PRNGKey(123)
    return jax.random.normal(key, (batch_size, OBSERVATION_DIM))


# ---------------------------------------------------------------------------
# Networks: 14D input handling, shapes, finiteness
# ---------------------------------------------------------------------------


def test_policy_network_handles_14d_input():
    network = build_policy_network()
    params = network.init(jax.random.PRNGKey(SEED), _dummy_observation())
    logits = network.apply(params, _dummy_observation())
    assert logits.shape == (NUM_ACTIONS,)


def test_policy_network_output_logits_shape_batched():
    network = build_policy_network()
    obs = _dummy_batch_observation(8)
    params = network.init(jax.random.PRNGKey(SEED), obs[0])
    logits = network.apply(params, obs)
    assert logits.shape == (8, NUM_ACTIONS)


def test_policy_network_logits_are_finite():
    network = build_policy_network()
    obs = _dummy_batch_observation(16)
    params = network.init(jax.random.PRNGKey(SEED), obs[0])
    logits = network.apply(params, obs)
    assert jnp.all(jnp.isfinite(logits))


def test_softmax_probabilities_sum_to_one():
    network = build_policy_network()
    obs = _dummy_batch_observation(16)
    params = network.init(jax.random.PRNGKey(SEED), obs[0])
    logits = network.apply(params, obs)
    probs = distribution.probs(logits)
    sums = jnp.sum(probs, axis=-1)
    assert jnp.allclose(sums, 1.0, atol=1e-5)


def test_value_network_handles_14d_input_and_scalar_output():
    network = build_value_network()
    params = network.init(jax.random.PRNGKey(SEED), _dummy_observation())
    value = network.apply(params, _dummy_observation())
    assert value.shape == ()


def test_value_network_batched_output_is_vector():
    network = build_value_network()
    obs = _dummy_batch_observation(10)
    params = network.init(jax.random.PRNGKey(SEED), obs[0])
    values = network.apply(params, obs)
    assert values.shape == (10,)
    assert jnp.all(jnp.isfinite(values))


# ---------------------------------------------------------------------------
# Sampling / determinism / action range
# ---------------------------------------------------------------------------


def test_sampled_action_always_in_range():
    network = build_policy_network()
    obs = _dummy_batch_observation(32)
    params = network.init(jax.random.PRNGKey(SEED), obs[0])
    logits = network.apply(params, obs)
    for i in range(20):
        key = jax.random.PRNGKey(i)
        action = distribution.sample_action(logits, key)
        assert jnp.all(action >= 0)
        assert jnp.all(action < NUM_ACTIONS)


def test_deterministic_inference_equals_argmax():
    network = build_policy_network()
    obs = _dummy_batch_observation(8)
    params = network.init(jax.random.PRNGKey(SEED), obs[0])
    logits = network.apply(params, obs)
    det_action = distribution.deterministic_action(logits)
    assert jnp.array_equal(det_action, jnp.argmax(logits, axis=-1))


def test_stochastic_sampling_produces_variation_across_keys():
    # Fixed observation, single set of logits -- sample with many
    # different keys and confirm not all outcomes are identical.
    logits = jnp.array([0.5, 0.6, 0.4, 0.3])
    actions = [
        int(distribution.sample_action(logits, jax.random.PRNGKey(i)))
        for i in range(200)
    ]
    assert len(set(actions)) > 1, "stochastic sampling never varied across 200 keys"


def test_log_prob_is_finite():
    logits = jnp.array([1.0, -2.0, 0.5, 3.0])
    for action in range(NUM_ACTIONS):
        lp = distribution.log_prob(logits, jnp.array(action))
        assert jnp.isfinite(lp)


def test_entropy_is_finite_and_nonnegative():
    key = jax.random.PRNGKey(7)
    logits_batch = jax.random.normal(key, (32, NUM_ACTIONS)) * 5.0
    ent = distribution.entropy(logits_batch)
    assert jnp.all(jnp.isfinite(ent))
    assert jnp.all(ent >= 0.0)


def test_entropy_uniform_logits_is_log_num_actions():
    # Sanity check on the entropy formula itself: uniform categorical
    # over NUM_ACTIONS has entropy = log(NUM_ACTIONS).
    logits = jnp.zeros((NUM_ACTIONS,))
    ent = distribution.entropy(logits)
    assert jnp.allclose(ent, jnp.log(NUM_ACTIONS), atol=1e-5)


# ---------------------------------------------------------------------------
# Action-mapping regression test (SS11) -- real BehaviorAction import
# ---------------------------------------------------------------------------


def test_action_mapping_regression_against_real_behavior_action():
    """Guards against the categorical index ever silently drifting
    from the frozen action semantics (PPO_PLAN.md SS11). Imports the
    real ``BehaviorAction`` enum -- not a hardcoded duplicate."""

    assert distribution.ACTION_INDEX_TO_BEHAVIOR[0] == BehaviorAction.KEEP
    assert distribution.ACTION_INDEX_TO_BEHAVIOR[1] == BehaviorAction.FOLLOW
    assert distribution.ACTION_INDEX_TO_BEHAVIOR[2] == BehaviorAction.MERGE
    assert distribution.ACTION_INDEX_TO_BEHAVIOR[3] == BehaviorAction.STOP
    assert BehaviorAction.KEEP == 0
    assert BehaviorAction.FOLLOW == 1
    assert BehaviorAction.MERGE == 2
    assert BehaviorAction.STOP == 3


# ---------------------------------------------------------------------------
# PPO ratio + clipping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old_lp,new_lp,expected_ratio",
    [
        (0.0, 0.0, 1.0),
        (jnp.log(0.5), jnp.log(0.5), 1.0),
        (jnp.log(0.5), jnp.log(1.0), 2.0),
        (jnp.log(1.0), jnp.log(0.25), 0.25),
        (-1.0, -1.0, 1.0),
        (-2.0, -1.0, jnp.e),
    ],
)
def test_ppo_ratio_hand_checked(old_lp, new_lp, expected_ratio):
    ratio = loss.ppo_ratio(jnp.asarray(old_lp), jnp.asarray(new_lp))
    assert jnp.allclose(ratio, expected_ratio, atol=1e-5)


def test_clipping_engages_when_ratio_exceeds_upper_bound():
    clip_epsilon = 0.2
    # new/old such that ratio = e^(1.0) ~= 2.718, far above 1+eps.
    old_log_prob = jnp.array([0.0])
    new_log_prob = jnp.array([1.0])
    advantages = jnp.array([1.0])  # positive advantage

    unclipped_ratio = loss.ppo_ratio(old_log_prob, new_log_prob)
    assert float(unclipped_ratio[0]) > 1.0 + clip_epsilon

    total_loss, info = loss.ppo_clipped_surrogate_loss(
        old_log_prob, new_log_prob, advantages, clip_epsilon
    )
    # With positive advantage, min(r*A, clip(r)*A) = clip(r)*A since
    # r*A > clip(r)*A when r > 1+eps and A > 0.
    expected_loss = -((1.0 + clip_epsilon) * 1.0)
    assert jnp.allclose(total_loss, expected_loss, atol=1e-5)
    assert float(info["clip_fraction"]) == 1.0


def test_clipping_engages_when_ratio_below_lower_bound():
    clip_epsilon = 0.2
    old_log_prob = jnp.array([0.0])
    new_log_prob = jnp.array([-2.0])  # ratio = e^-2 ~= 0.135, below 1-eps
    advantages = jnp.array([1.0])

    unclipped_ratio = loss.ppo_ratio(old_log_prob, new_log_prob)
    assert float(unclipped_ratio[0]) < 1.0 - clip_epsilon

    total_loss, info = loss.ppo_clipped_surrogate_loss(
        old_log_prob, new_log_prob, advantages, clip_epsilon
    )
    # With positive advantage and r < 1-eps: r*A < clip(r)*A, so
    # min picks the UNCLIPPED r*A (this is the asymmetric PPO clip
    # behavior -- clipping only prevents the objective from being
    # pushed higher than the clipped value, not lower).
    expected_loss = -(float(unclipped_ratio[0]) * 1.0)
    assert jnp.allclose(total_loss, expected_loss, atol=1e-5)
    assert float(info["clip_fraction"]) == 1.0


def test_no_clipping_when_ratio_within_bounds():
    clip_epsilon = 0.2
    old_log_prob = jnp.array([0.0])
    new_log_prob = jnp.array([0.05])  # ratio ~= 1.05, within [0.8, 1.2]
    advantages = jnp.array([1.0])

    total_loss, info = loss.ppo_clipped_surrogate_loss(
        old_log_prob, new_log_prob, advantages, clip_epsilon
    )
    ratio = loss.ppo_ratio(old_log_prob, new_log_prob)
    expected_loss = -(float(ratio[0]) * 1.0)
    assert jnp.allclose(total_loss, expected_loss, atol=1e-5)
    assert float(info["clip_fraction"]) == 0.0


# ---------------------------------------------------------------------------
# Value loss / full PPO loss / gradients / optimizer step
# ---------------------------------------------------------------------------


def test_value_loss_is_scalar_and_correct():
    values = jnp.array([1.0, 2.0, 3.0])
    returns = jnp.array([1.5, 2.0, 2.5])
    v_loss = loss.value_loss(values, returns)
    expected = jnp.mean((returns - values) ** 2)
    assert v_loss.shape == ()
    assert jnp.allclose(v_loss, expected)


def _synthetic_minibatch(batch_size=16, seed=SEED):
    key = jax.random.PRNGKey(seed)
    key_obs, key_old_logits, key_action, key_adv, key_ret = jax.random.split(key, 5)

    observations = jax.random.normal(key_obs, (batch_size, OBSERVATION_DIM))
    old_logits = jax.random.normal(key_old_logits, (batch_size, NUM_ACTIONS))
    actions = jax.random.categorical(key_action, old_logits, axis=-1)
    old_log_prob = distribution.log_prob(old_logits, actions)
    advantages = jax.random.normal(key_adv, (batch_size,))
    returns = jax.random.normal(key_ret, (batch_size,)) + 1.0

    return {
        "observations": observations,
        "actions": actions,
        "old_log_prob": old_log_prob,
        "advantages": advantages,
        "returns": returns,
    }


def test_full_ppo_loss_is_finite_for_synthetic_minibatch():
    network = build_policy_network()
    value_network = build_value_network()
    batch = _synthetic_minibatch()

    policy_params = network.init(jax.random.PRNGKey(1), batch["observations"][0])
    value_params = value_network.init(jax.random.PRNGKey(2), batch["observations"][0])

    new_logits = network.apply(policy_params, batch["observations"])
    new_log_prob = distribution.log_prob(new_logits, batch["actions"])
    values = value_network.apply(value_params, batch["observations"])

    total_loss, info = loss.ppo_total_loss(
        old_log_prob=batch["old_log_prob"],
        new_log_prob=new_log_prob,
        advantages=batch["advantages"],
        values=values,
        returns=batch["returns"],
        logits=new_logits,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )

    assert jnp.isfinite(total_loss)
    for key, val in info.items():
        assert jnp.all(jnp.isfinite(val)), f"non-finite value in loss info[{key}]"


def test_gradients_of_full_loss_are_finite_and_nonzero():
    network = build_policy_network()
    value_network = build_value_network()
    batch = _synthetic_minibatch()

    policy_params = network.init(jax.random.PRNGKey(1), batch["observations"][0])
    value_params = value_network.init(jax.random.PRNGKey(2), batch["observations"][0])

    def loss_fn(policy_params, value_params):
        new_logits = network.apply(policy_params, batch["observations"])
        new_log_prob = distribution.log_prob(new_logits, batch["actions"])
        values = value_network.apply(value_params, batch["observations"])
        total_loss, _ = loss.ppo_total_loss(
            old_log_prob=batch["old_log_prob"],
            new_log_prob=new_log_prob,
            advantages=batch["advantages"],
            values=values,
            returns=batch["returns"],
            logits=new_logits,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
        )
        return total_loss

    grads_policy, grads_value = jax.grad(loss_fn, argnums=(0, 1))(policy_params, value_params)

    def _all_finite(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)

    def _any_nonzero(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return any(bool(jnp.any(leaf != 0.0)) for leaf in leaves)

    assert _all_finite(grads_policy), "policy gradients contain NaN/inf"
    assert _all_finite(grads_value), "value gradients contain NaN/inf"
    assert _any_nonzero(grads_policy), "policy gradients are all zero (wiring bug)"
    assert _any_nonzero(grads_value), "value gradients are all zero (wiring bug)"


def test_optimizer_step_changes_parameters():
    train_state = create_train_state(
        rng_key=jax.random.PRNGKey(SEED),
        learning_rate=3e-4,
        max_grad_norm=0.5,
    )
    batch = _synthetic_minibatch()

    def policy_loss_fn(policy_params):
        new_logits = train_state.policy_network.apply(policy_params, batch["observations"])
        new_log_prob = distribution.log_prob(new_logits, batch["actions"])
        values = train_state.value_network.apply(
            train_state.value_state.params, batch["observations"]
        )
        total_loss, _ = loss.ppo_total_loss(
            old_log_prob=batch["old_log_prob"],
            new_log_prob=new_log_prob,
            advantages=batch["advantages"],
            values=values,
            returns=batch["returns"],
            logits=new_logits,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
        )
        return total_loss

    params_before = train_state.policy_state.params
    grads = jax.grad(policy_loss_fn)(params_before)
    new_policy_state = train_state.policy_state.apply_gradients(grads=grads)
    params_after = new_policy_state.params

    leaves_before = jax.tree_util.tree_leaves(params_before)
    leaves_after = jax.tree_util.tree_leaves(params_after)
    any_changed = any(
        not bool(jnp.array_equal(b, a)) for b, a in zip(leaves_before, leaves_after)
    )
    assert any_changed, "optimizer step did not change any policy parameter"


def test_value_optimizer_step_changes_parameters():
    train_state = create_train_state(
        rng_key=jax.random.PRNGKey(SEED),
        learning_rate=3e-4,
        max_grad_norm=0.5,
    )
    batch = _synthetic_minibatch()

    def value_loss_fn(value_params):
        values = train_state.value_network.apply(value_params, batch["observations"])
        return loss.value_loss(values, batch["returns"])

    params_before = train_state.value_state.params
    grads = jax.grad(value_loss_fn)(params_before)
    new_value_state = train_state.value_state.apply_gradients(grads=grads)
    params_after = new_value_state.params

    leaves_before = jax.tree_util.tree_leaves(params_before)
    leaves_after = jax.tree_util.tree_leaves(params_after)
    any_changed = any(
        not bool(jnp.array_equal(b, a)) for b, a in zip(leaves_before, leaves_after)
    )
    assert any_changed, "optimizer step did not change any value parameter"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_reproducibility_network_init():
    network = build_policy_network()
    obs = _dummy_observation()

    params_1 = network.init(jax.random.PRNGKey(42), obs)
    params_2 = network.init(jax.random.PRNGKey(42), obs)

    leaves_1 = jax.tree_util.tree_leaves(params_1)
    leaves_2 = jax.tree_util.tree_leaves(params_2)
    for l1, l2 in zip(leaves_1, leaves_2):
        assert jnp.array_equal(l1, l2)

    logits_1 = network.apply(params_1, obs)
    logits_2 = network.apply(params_2, obs)
    assert jnp.array_equal(logits_1, logits_2)


def test_same_seed_reproducibility_sampling():
    logits = jnp.array([0.1, 0.4, -0.2, 0.3])
    action_1 = distribution.sample_action(logits, jax.random.PRNGKey(99))
    action_2 = distribution.sample_action(logits, jax.random.PRNGKey(99))
    assert int(action_1) == int(action_2)


def test_different_seeds_can_differ():
    logits = jnp.array([0.1, 0.4, -0.2, 0.3])
    actions = {
        int(distribution.sample_action(logits, jax.random.PRNGKey(i))) for i in range(50)
    }
    assert len(actions) > 1


# ---------------------------------------------------------------------------
# PPOPolicy wrapper
# ---------------------------------------------------------------------------


def test_ppo_policy_act_deterministic_matches_argmax():
    network = build_policy_network()
    obs = _dummy_observation()
    params = network.init(jax.random.PRNGKey(SEED), obs)
    ppo_policy = PPOPolicy(network, params)

    action = ppo_policy.act_deterministic(obs)
    logits = network.apply(params, obs)
    assert int(action) == int(jnp.argmax(logits))
    assert 0 <= int(action) < NUM_ACTIONS


def test_ppo_policy_act_returns_action_and_log_prob():
    network = build_policy_network()
    obs = _dummy_observation()
    params = network.init(jax.random.PRNGKey(SEED), obs)
    ppo_policy = PPOPolicy(network, params)

    action, log_prob = ppo_policy.act(obs, jax.random.PRNGKey(5))
    assert 0 <= int(action) < NUM_ACTIONS
    assert jnp.isfinite(log_prob)


# ---------------------------------------------------------------------------
# create_train_state
# ---------------------------------------------------------------------------


def test_create_train_state_produces_working_state():
    train_state = create_train_state(
        rng_key=jax.random.PRNGKey(SEED),
        learning_rate=3e-4,
        max_grad_norm=0.5,
    )
    obs = _dummy_observation()
    logits = train_state.policy_network.apply(train_state.policy_state.params, obs)
    value = train_state.value_network.apply(train_state.value_state.params, obs)
    assert logits.shape == (NUM_ACTIONS,)
    assert value.shape == ()
    assert jnp.all(jnp.isfinite(logits))
    assert jnp.isfinite(value)


def test_create_train_state_is_reproducible_for_same_seed():
    ts_1 = create_train_state(rng_key=jax.random.PRNGKey(7), learning_rate=3e-4, max_grad_norm=0.5)
    ts_2 = create_train_state(rng_key=jax.random.PRNGKey(7), learning_rate=3e-4, max_grad_norm=0.5)

    leaves_1 = jax.tree_util.tree_leaves(ts_1.policy_state.params)
    leaves_2 = jax.tree_util.tree_leaves(ts_2.policy_state.params)
    for l1, l2 in zip(leaves_1, leaves_2):
        assert jnp.array_equal(l1, l2)
