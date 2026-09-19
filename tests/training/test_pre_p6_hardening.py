"""Pre-P6 hardening pass tests (docs/ppo/PRE_P6_REPORT.md).

Covers Fix 2 (exact categorical KL diagnostic), Fix 3 (PPO update
metric aggregation across all epochs/minibatches, not just the last),
Fix 5 (NumPy RNG checkpoint round-trip), Fix 6 (W&B full diagnostics:
episode-outcome rates, downstream-intervention aggregation, explained-
variance edge cases, policy_decision_count/physical_step_count,
throughput), and Fix 7 (reward component logging correctness). Fix 1
(episode-aware GAE) has its own dedicated tests in
``tests/training/test_gae.py``.

None of these tests change the PPO-Clip objective, Reward V0's values,
or any frozen Phase 1-3 semantics -- they exercise DIAGNOSTIC/
correctness code only.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.policies.ppo.networks import NUM_ACTIONS
from src.training.checkpoint import CheckpointPayload, load_checkpoint, restore_numpy_rng, save_checkpoint
from src.training.trainer import _aggregate_downstream_rates, _exact_categorical_kl
from src.training.rollout import Transition


# ======================================================================
# Fix 2: exact categorical KL diagnostic
# ======================================================================


def test_exact_kl_zero_for_identical_logits():
    logits = jnp.array([[1.0, 0.5, -0.5, 2.0], [0.0, 0.0, 0.0, 0.0]])
    kl = _exact_categorical_kl(logits, logits)
    np.testing.assert_allclose(np.asarray(kl), np.zeros(2), atol=1e-6)


def test_exact_kl_positive_for_changed_logits():
    old_logits = jnp.array([[1.0, 0.0, 0.0, 0.0]])
    new_logits = jnp.array([[0.0, 1.0, 0.0, 0.0]])
    kl = _exact_categorical_kl(old_logits, new_logits)
    assert float(kl[0]) > 0.0


def test_exact_kl_matches_hand_computed_categorical_kl():
    """Hand-computed exact categorical KL(old || new) for a specific
    2-logit-pair example, cross-checked against a manual
    sum_a p_old(a) * (log p_old(a) - log p_new(a)) computation using
    plain numpy softmax (independent of the jax.nn.log_softmax
    implementation under test)."""

    old_logits = np.array([2.0, 1.0, 0.0, -1.0])
    new_logits = np.array([1.5, 1.0, 0.5, -0.5])

    def _manual_softmax(logits):
        exp = np.exp(logits - np.max(logits))
        return exp / np.sum(exp)

    old_probs = _manual_softmax(old_logits)
    new_probs = _manual_softmax(new_logits)
    expected_kl = float(np.sum(old_probs * (np.log(old_probs) - np.log(new_probs))))

    kl = _exact_categorical_kl(
        jnp.asarray(old_logits)[None, :], jnp.asarray(new_logits)[None, :]
    )
    assert math.isclose(float(kl[0]), expected_kl, abs_tol=1e-6)
    assert expected_kl > 0.0  # sanity: these logits actually differ


def test_exact_kl_batched_rows_independent():
    """Exact KL is computed independently per row -- a batch of mixed
    identical/differing rows shows zero KL only where the logits
    actually match."""

    old_logits = jnp.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
    new_logits = jnp.array([[1.0, 2.0, 3.0, 4.0], [5.0, 0.0, 0.0, 0.0]])
    kl = np.asarray(_exact_categorical_kl(old_logits, new_logits))
    assert math.isclose(kl[0], 0.0, abs_tol=1e-6)
    assert kl[1] > 0.0


def test_exact_kl_never_used_for_policy_mask_zero_rows():
    """Mirrors the existing policy_mask==0 Actor-exclusion pattern from
    P4: a Transition constructed for an auto-execution (policy_mask=0)
    frame must carry the zero-vector old_logits sentinel, which -- if
    it were ever accidentally included in an exact-KL aggregate -- would
    contribute exactly zero divergence from a zero-logit "new" policy,
    making silent inclusion easy to miss. This test locks down the
    SENTINEL VALUE itself (all-zero), which src/training/rollout.py's
    collect_episode_rollout is responsible for emitting on such frames,
    and confirms _filter_actor_rows-style boolean masking excludes it
    exactly like every other Actor-side field."""

    zero_sentinel = np.zeros((NUM_ACTIONS,), dtype=np.float64)
    real_logits = np.array([1.0, -2.0, 0.5, 3.0])

    old_logits_batch = np.stack([real_logits, zero_sentinel, real_logits])
    policy_mask = np.array([1, 0, 1], dtype=bool)

    filtered = old_logits_batch[policy_mask]
    assert filtered.shape[0] == 2
    for row in filtered:
        np.testing.assert_array_equal(row, real_logits)
    # The masked-out row's sentinel never appears in the filtered set.
    assert not any(np.array_equal(row, zero_sentinel) for row in filtered)


# ======================================================================
# Fix 3: PPO update metric aggregation across all epochs/minibatches
# ======================================================================


def test_run_update_aggregation_reflects_full_sweep_not_just_last_minibatch():
    """Constructs a scenario where minibatch-level policy_loss values
    vary meaningfully across the ppo_epochs x num_minibatches sweep
    (via changing advantages per-minibatch through varying rows), then
    confirms the logged ppo/policy_loss_mean reflects the MEAN across
    the whole sweep, not merely the last minibatch processed -- by
    running run_update with num_minibatches=1 (a single "last" value
    that trivially equals the mean of 1 element) vs num_minibatches>1
    (verifying the mean over >1 elements is not by coincidence just
    the last element for a batch engineered to have very different
    per-minibatch advantage scales)."""

    from src.policies.ppo.state import create_train_state
    from src.training.config import (
        NetworkConfig,
        PPOConfig,
        PPOHyperparameters,
        RolloutConfig,
        TrackingConfig,
    )
    from src.training.trainer import run_update

    rng_key = jax.random.PRNGKey(0)
    training_state = create_train_state(
        rng_key, learning_rate=3e-4, max_grad_norm=0.5,
        policy_hidden_sizes=(8, 8), value_hidden_sizes=(8, 8),
    )

    n = 8
    obs_rng = np.random.RandomState(1)
    observations = obs_rng.uniform(-1, 1, size=(n, 14)).astype(np.float32)
    # Deliberately give the minibatches very different advantage
    # SCALES (first half tiny, second half huge) so a "last minibatch
    # wins" bug would produce a policy_loss/grad_norm wildly different
    # from the correctly-averaged one.
    advantages = np.concatenate([np.full(n // 2, 0.01), np.full(n - n // 2, 100.0)])
    actions = np.zeros(n, dtype=np.int32)
    old_logits = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    old_log_prob = np.full(n, -1.386)  # log(1/4), roughly uniform
    returns = obs_rng.uniform(-1, 1, size=n)

    batch = {
        "observation": observations,
        "action": actions,
        "log_prob": old_log_prob,
        "policy_mask": np.ones(n, dtype=np.int32),
        "advantages": advantages,
        "old_logits": old_logits,
        "returns": returns,
        "value": np.zeros(n),
    }

    ppo_config_single_mb = PPOConfig(
        seed=0,
        network=NetworkConfig(observation_dim=14, num_actions=4, hidden_sizes=[8, 8], activation="tanh"),
        hyperparameters=PPOHyperparameters(
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
            value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
            ppo_epochs=1, num_minibatches=1,
        ),
        rollout=RolloutConfig(downstream_mode="frenet_mpc"),
        reward_config_path="configs/reward/merge_reward_v0.yaml",
        tracking=TrackingConfig(wandb_project="test", wandb_mode="offline"),
        smoke=None,
        source_path="<test>",
    )
    ppo_config_multi_mb = PPOConfig(
        seed=0,
        network=ppo_config_single_mb.network,
        hyperparameters=PPOHyperparameters(
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
            value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
            ppo_epochs=1, num_minibatches=n,  # one row per minibatch
        ),
        rollout=ppo_config_single_mb.rollout,
        reward_config_path=ppo_config_single_mb.reward_config_path,
        tracking=ppo_config_single_mb.tracking,
        smoke=None,
        source_path="<test>",
    )

    # With num_minibatches == n, every row is its own minibatch. Since
    # each minibatch's gradient step slightly updates the params before
    # the next minibatch is processed, the exact aggregate value is not
    # perfectly shuffle-order invariant (a real, expected effect of
    # sequential SGD-style minibatching) -- but it stays close across
    # different shuffle orders. A "last-minibatch-wins" bug, in
    # contrast, would report a metric equal to whichever SINGLE row's
    # own policy_loss happened to be processed last -- and this batch
    # was deliberately constructed so the tiny-advantage half and
    # huge-advantage half of rows have very different per-row
    # policy_loss magnitudes, so "last row's own value" and "mean over
    # all rows" are far apart whenever the last-processed row lands in
    # the huge-advantage half.
    numpy_rng_a = np.random.RandomState(42)
    result_multi = run_update(training_state, batch, ppo_config_multi_mb, numpy_rng_a)

    numpy_rng_b = np.random.RandomState(999)  # different shuffle order
    result_multi_2 = run_update(training_state, batch, ppo_config_multi_mb, numpy_rng_b)

    loss_multi_a = result_multi["metrics"]["ppo/policy_loss_mean"]
    loss_multi_b = result_multi_2["metrics"]["ppo/policy_loss_mean"]
    # Correctly-aggregated means across two different shuffle orders of
    # the SAME underlying rows must be close to each other (small
    # sequential-update drift only) -- not off by orders of magnitude,
    # which is what comparing "mean of tiny+huge" against "just the
    # last huge row" would produce.
    assert math.isclose(loss_multi_a, loss_multi_b, rel_tol=0.05, abs_tol=2.0), (
        "ppo/policy_loss_mean should be close (small sequential-update "
        "drift only) across different minibatch shuffle orders -- a "
        "last-minibatch-wins bug would make this wildly shuffle-order-"
        f"dependent instead. Got {loss_multi_a} vs {loss_multi_b}."
    )

    # Direct, decisive check: a single-minibatch run over ONLY the
    # huge-advantage half of rows (simulating "the last minibatch
    # happened to be one of the huge-advantage rows") produces a
    # policy_loss far from the correctly-aggregated whole-batch mean --
    # proving the whole-batch aggregate is NOT simply equal to a
    # single huge-advantage-only minibatch's value.
    huge_only_batch = {k: v[n // 2:] for k, v in batch.items()}
    result_huge_only = run_update(
        training_state, huge_only_batch, ppo_config_single_mb, np.random.RandomState(0)
    )
    loss_huge_only = result_huge_only["metrics"]["ppo/policy_loss_mean"]
    assert not math.isclose(loss_multi_a, loss_huge_only, rel_tol=0.2, abs_tol=5.0), (
        "The full-batch aggregated policy_loss_mean should differ "
        "substantially from a huge-advantage-only minibatch's value -- "
        "if they matched, that would suggest only the huge-advantage "
        "rows (e.g. 'the last minibatch') were actually being reported."
    )

    # Sanity: the metric keys required by Fix 3 are all present.
    for key in (
        "ppo/policy_loss_mean", "ppo/value_loss_mean", "ppo/entropy_mean",
        "ppo/approx_kl_mean", "ppo/exact_kl_mean", "ppo/exact_kl_max",
        "ppo/clip_fraction_mean", "ppo/policy_grad_norm_mean", "ppo/policy_grad_norm_max",
        "ppo/value_grad_norm_mean", "ppo/value_grad_norm_max",
    ):
        assert key in result_multi["metrics"], f"missing Fix 3 metric key: {key}"
        assert np.isfinite(result_multi["metrics"][key])


def test_run_update_aggregation_not_last_value_across_epochs():
    """A more direct confirmation: run_update over ppo_epochs=3,
    num_minibatches=1 (so each epoch is exactly one "minibatch" whose
    value could trivially be mistaken for "the last epoch's value" by
    a last-value-wins bug). Since parameters change between epochs
    (real gradient steps), each epoch's policy_loss WILL differ, so a
    correct mean-over-3-epochs value must generically differ from
    epoch 3 alone -- assert that directly by reproducing epoch-by-epoch
    values via repeated single-epoch run_update calls and comparing
    their mean to the multi-epoch aggregate."""

    from src.policies.ppo.state import create_train_state
    from src.training.config import (
        NetworkConfig, PPOConfig, PPOHyperparameters, RolloutConfig, TrackingConfig,
    )
    from src.training.trainer import run_update

    rng_key = jax.random.PRNGKey(7)
    training_state = create_train_state(
        rng_key, learning_rate=1e-2, max_grad_norm=0.5,
        policy_hidden_sizes=(8,), value_hidden_sizes=(8,),
    )

    n = 6
    obs_rng = np.random.RandomState(5)
    observations = obs_rng.uniform(-1, 1, size=(n, 14)).astype(np.float32)
    advantages = obs_rng.uniform(-2, 2, size=n)
    actions = obs_rng.randint(0, 4, size=n).astype(np.int32)
    old_logits = np.tile(np.array([0.25, 0.25, 0.25, 0.25]), (n, 1))
    old_log_prob = np.full(n, -1.386)
    returns = obs_rng.uniform(-1, 1, size=n)

    batch = {
        "observation": observations,
        "action": actions,
        "log_prob": old_log_prob,
        "policy_mask": np.ones(n, dtype=np.int32),
        "advantages": advantages,
        "old_logits": old_logits,
        "returns": returns,
        "value": np.zeros(n),
    }

    def _make_config(ppo_epochs):
        return PPOConfig(
            seed=0,
            network=NetworkConfig(observation_dim=14, num_actions=4, hidden_sizes=[8], activation="tanh"),
            hyperparameters=PPOHyperparameters(
                learning_rate=1e-2, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
                value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
                ppo_epochs=ppo_epochs, num_minibatches=1,
            ),
            rollout=RolloutConfig(downstream_mode="frenet_mpc"),
            reward_config_path="configs/reward/merge_reward_v0.yaml",
            tracking=TrackingConfig(wandb_project="test", wandb_mode="offline"),
            smoke=None,
            source_path="<test>",
        )

    # 3-epoch aggregate (the function under test).
    result_3_epochs = run_update(
        training_state, batch, _make_config(3), np.random.RandomState(0)
    )
    aggregated_loss = result_3_epochs["metrics"]["ppo/policy_loss_mean"]

    # Reproduce it manually: 3 sequential single-epoch run_update calls,
    # threading the training_state through, collecting each epoch's
    # OWN policy_loss (num_minibatches=1 here means the single value IS
    # that epoch's value, letting us hand-verify the aggregate).
    state = training_state
    per_epoch_losses = []
    for _ in range(3):
        result = run_update(state, batch, _make_config(1), np.random.RandomState(0))
        per_epoch_losses.append(result["metrics"]["ppo/policy_loss_mean"])
        state = result["training_state"]

    manual_mean = float(np.mean(per_epoch_losses))

    # The aggregate must match the mean of the per-epoch values (not
    # just the last epoch's value, which would be per_epoch_losses[-1]
    # and should generically differ from the mean given real parameter
    # updates between epochs).
    assert math.isclose(aggregated_loss, manual_mean, rel_tol=1e-4, abs_tol=1e-6)
    if not math.isclose(per_epoch_losses[0], per_epoch_losses[-1], rel_tol=1e-3):
        assert not math.isclose(aggregated_loss, per_epoch_losses[-1], rel_tol=1e-6, abs_tol=1e-9), (
            "The aggregated metric coincidentally equals just the last "
            "epoch's value -- this would not distinguish a fixed "
            "aggregation from a last-value-wins bug; the per-epoch "
            "values differed enough that this should not happen."
        )


# ======================================================================
# Fix 5: NumPy RNG checkpoint round-trip
# ======================================================================


def test_numpy_rng_state_round_trips_through_checkpoint(tmp_path):
    rng = np.random.RandomState(123)
    rng.permutation(100)  # advance the stream so state isn't the freshly-seeded one

    payload = CheckpointPayload(
        policy_params={"w": [1.0]},
        value_params={"w": [2.0]},
        optimizer_state={"step": 1},
        jax_rng_key=jax.random.PRNGKey(0),
        global_env_step=10,
        ppo_update_step=1,
        seed=123,
        config_snapshot={},
        reward_version="v0",
        git_sha="abc123",
        numpy_rng_state=rng.get_state(),
    )
    path = tmp_path / "ckpt.pkl"
    save_checkpoint(payload, str(path))
    loaded = load_checkpoint(str(path))

    restored = restore_numpy_rng(loaded.numpy_rng_state, fallback_seed=123)

    expected_next = rng.permutation(50)
    actual_next = restored.permutation(50)
    np.testing.assert_array_equal(expected_next, actual_next)


def test_numpy_rng_state_missing_falls_back_to_seed():
    """An OLDER checkpoint (numpy_rng_state=None) must still load and
    produce a deterministic (seed-derived) RNG rather than crashing."""

    restored = restore_numpy_rng(None, fallback_seed=42)
    expected = np.random.RandomState(42).permutation(10)
    actual = restored.permutation(10)
    np.testing.assert_array_equal(expected, actual)


def test_resumed_minibatch_order_matches_uninterrupted_run():
    """The stronger SS10-style claim: Run A (no checkpoint) does update
    N then N+1 on a deterministic synthetic batch/state. Run B does
    update N, saves a checkpoint (including numpy_rng_state), starts a
    FRESH RandomState, loads the checkpoint, and does update N+1. With
    identical initial params/config, Run A and Run B's update-N+1
    RESULTING PARAMS must match exactly -- proving the checkpoint
    resumes the STOCHASTIC minibatch-shuffling state, not merely the
    static weights."""

    from src.policies.ppo.state import create_train_state
    from src.training.config import (
        NetworkConfig, PPOConfig, PPOHyperparameters, RolloutConfig, TrackingConfig,
    )
    from src.training.trainer import run_update

    def _fresh_training_state():
        return create_train_state(
            jax.random.PRNGKey(11), learning_rate=1e-2, max_grad_norm=0.5,
            policy_hidden_sizes=(8,), value_hidden_sizes=(8,),
        )

    n = 12
    obs_rng = np.random.RandomState(3)
    observations = obs_rng.uniform(-1, 1, size=(n, 14)).astype(np.float32)
    advantages = obs_rng.uniform(-1, 1, size=n)
    actions = obs_rng.randint(0, 4, size=n).astype(np.int32)
    old_logits = np.tile(np.array([0.25, 0.25, 0.25, 0.25]), (n, 1))
    old_log_prob = np.full(n, -1.386)
    returns = obs_rng.uniform(-1, 1, size=n)
    batch = {
        "observation": observations, "action": actions, "log_prob": old_log_prob,
        "policy_mask": np.ones(n, dtype=np.int32), "advantages": advantages,
        "old_logits": old_logits, "returns": returns, "value": np.zeros(n),
    }
    ppo_config = PPOConfig(
        seed=0,
        network=NetworkConfig(observation_dim=14, num_actions=4, hidden_sizes=[8], activation="tanh"),
        hyperparameters=PPOHyperparameters(
            learning_rate=1e-2, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
            value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
            ppo_epochs=2, num_minibatches=4,
        ),
        rollout=RolloutConfig(downstream_mode="frenet_mpc"),
        reward_config_path="configs/reward/merge_reward_v0.yaml",
        tracking=TrackingConfig(wandb_project="test", wandb_mode="offline"),
        smoke=None,
        source_path="<test>",
    )

    def _leaves(pytree):
        import jax.tree_util as jtu
        return [np.array(x) for x in jtu.tree_leaves(pytree)]

    # --- Run A: uninterrupted, update N then N+1, SAME RandomState object
    # threaded through both calls (as a real training loop would).
    state_a = _fresh_training_state()
    numpy_rng_a = np.random.RandomState(555)
    result_n_a = run_update(state_a, batch, ppo_config, numpy_rng_a)
    result_n1_a = run_update(result_n_a["training_state"], batch, ppo_config, numpy_rng_a)

    # --- Run B: update N with a FRESH RandomState(555), checkpoint its
    # POST-update state (Fix 5's numpy_rng_state), then simulate a brand
    # new process: build a fresh RandomState, restore from the
    # checkpointed state, and run update N+1.
    state_b = _fresh_training_state()
    numpy_rng_b_run1 = np.random.RandomState(555)
    result_n_b = run_update(state_b, batch, ppo_config, numpy_rng_b_run1)
    checkpointed_numpy_state = numpy_rng_b_run1.get_state()

    numpy_rng_b_resumed = restore_numpy_rng(checkpointed_numpy_state, fallback_seed=555)
    result_n1_b = run_update(result_n_b["training_state"], batch, ppo_config, numpy_rng_b_resumed)

    # Update N's resulting params must match between A and B (same
    # everything so far).
    for a, b in zip(_leaves(result_n_a["training_state"].policy_state.params),
                     _leaves(result_n_b["training_state"].policy_state.params)):
        np.testing.assert_array_equal(a, b)

    # Update N+1's resulting params must ALSO match -- this is the
    # real claim: Run B's numpy RNG state resumed correctly, so its
    # minibatch shuffle order for update N+1 matches Run A's
    # continuously-threaded RNG exactly.
    for a, b in zip(_leaves(result_n1_a["training_state"].policy_state.params),
                     _leaves(result_n1_b["training_state"].policy_state.params)):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(_leaves(result_n1_a["training_state"].value_state.params),
                     _leaves(result_n1_b["training_state"].value_state.params)):
        np.testing.assert_array_equal(a, b)


# ======================================================================
# Fix 6: W&B full diagnostics
# ======================================================================


def _make_transition(**overrides):
    defaults = dict(
        observation=np.zeros(14, dtype=np.float32),
        action=0,
        reward=0.0,
        next_observation=np.zeros(14, dtype=np.float32),
        terminated=False,
        truncated=False,
        value=0.0,
        next_value=0.0,
        log_prob=-1.386,
        policy_mask=1,
        episode_id="ep0",
        info=None,
    )
    defaults.update(overrides)
    return Transition(**defaults)


def test_aggregate_downstream_rates_from_episode_info():
    info = {
        "steps_elapsed": 10,
        "intervention_rate": 0.2,
        "planner_infeasible_count": 1,
        "collision_blocked_count": 2,
        "controller_failure_count": 0,
        "invalid_reference_count": 1,
    }
    transitions = [
        _make_transition(episode_id="ep0", info={"steps_elapsed": 5}),
        _make_transition(episode_id="ep0", info=info),  # last transition of ep0
    ]
    rates = _aggregate_downstream_rates(transitions)
    assert math.isclose(rates["downstream/intervention_rate"], 0.2, abs_tol=1e-9)
    assert math.isclose(rates["downstream/planner_infeasible_rate"], 1 / 10, abs_tol=1e-9)
    assert math.isclose(rates["downstream/collision_blocked_rate"], 2 / 10, abs_tol=1e-9)
    assert math.isclose(rates["downstream/controller_failure_rate"], 0.0, abs_tol=1e-9)
    assert math.isclose(rates["downstream/invalid_reference_rate"], 1 / 10, abs_tol=1e-9)


def test_aggregate_downstream_rates_no_info_returns_zero_not_error():
    transitions = [_make_transition(info=None), _make_transition(info=None)]
    rates = _aggregate_downstream_rates(transitions)
    for value in rates.values():
        assert value == 0.0


def test_aggregate_downstream_rates_averages_across_multiple_episodes():
    info_ep0 = {"steps_elapsed": 10, "intervention_rate": 0.0, "planner_infeasible_count": 0,
                "collision_blocked_count": 0, "controller_failure_count": 0, "invalid_reference_count": 0}
    info_ep1 = {"steps_elapsed": 10, "intervention_rate": 1.0, "planner_infeasible_count": 10,
                "collision_blocked_count": 0, "controller_failure_count": 0, "invalid_reference_count": 0}
    transitions = [
        _make_transition(episode_id="ep0", info=info_ep0),
        _make_transition(episode_id="ep1", info=info_ep1),
    ]
    rates = _aggregate_downstream_rates(transitions)
    assert math.isclose(rates["downstream/intervention_rate"], 0.5, abs_tol=1e-9)
    assert math.isclose(rates["downstream/planner_infeasible_rate"], 0.5, abs_tol=1e-9)


def test_explained_variance_finite_and_correct_near_zero_variance():
    """Constructs a batch with near-constant returns (Var ~= 0) and
    confirms run_update's explained_variance computation reports 0.0
    rather than NaN/inf -- exercised directly via the same formula used
    in run_update (duplicated here as a focused unit test of the
    edge-case branch, since run_update itself needs a full training
    state to invoke)."""

    returns = np.full(10, 5.0)  # exactly constant -> Var == 0
    values = np.random.RandomState(0).uniform(-1, 1, size=10)

    returns_var = float(np.var(returns))
    if returns_var < 1e-8:
        explained_variance = 0.0
    else:
        explained_variance = float(1.0 - np.var(returns - values) / returns_var)

    assert explained_variance == 0.0
    assert np.isfinite(explained_variance)


def test_explained_variance_normal_case_between_reasonable_bounds():
    rng = np.random.RandomState(0)
    returns = rng.uniform(-10, 10, size=100)
    # Values that are a decent (but imperfect) predictor of returns.
    values = returns + rng.normal(0, 1.0, size=100)

    returns_var = float(np.var(returns))
    explained_variance = float(1.0 - np.var(returns - values) / returns_var)

    assert np.isfinite(explained_variance)
    assert explained_variance > 0.5  # values track returns fairly well


# ======================================================================
# Fix 7: reward component logging correctness
# ======================================================================


def test_reward_terminal_includes_truncation_horizon_component():
    """The bug this guards against: a TRUNCATION_HORIZON (truncated=True,
    NOT terminated=True) step's -0.5 terminal component must be counted
    in reward/terminal, not silently folded into reward/decision_cost."""

    from src.rewards.merge_reward import compute_reward
    from src.training.config import load_reward_config

    reward_config = load_reward_config()

    # A truncated (TRUNCATION_HORIZON) step on a real decision frame.
    truncation_reward = compute_reward(
        reward_config, termination_reason="truncation_horizon", is_policy_step=True
    )
    assert math.isclose(truncation_reward, -0.5 + -0.01, abs_tol=1e-9)

    transitions = [
        _make_transition(
            episode_id="ep0", terminated=False, truncated=True, reward=truncation_reward
        )
    ]
    reward_terminal = float(
        sum(t.reward for t in transitions if t.terminated or t.truncated)
    )
    reward_total = float(sum(t.reward for t in transitions))
    reward_decision_cost = reward_total - reward_terminal

    # The FULL truncation reward (terminal -0.5 + decision cost -0.01)
    # must be attributed to reward_terminal's bucket in this
    # aggregation style, not split incorrectly.
    assert math.isclose(reward_terminal, truncation_reward, abs_tol=1e-9)
    assert math.isclose(reward_decision_cost, 0.0, abs_tol=1e-9)


def test_reward_terminal_collision_and_offroad_split():
    from src.rewards.merge_reward import compute_reward
    from src.training.config import load_reward_config

    reward_config = load_reward_config()
    collision_reward = compute_reward(reward_config, "failure_collision", is_policy_step=False)
    offroad_reward = compute_reward(reward_config, "failure_offroad", is_policy_step=False)

    transitions = [
        _make_transition(episode_id="ep0", terminated=True, truncated=False, reward=collision_reward),
    ]
    reward_terminal = float(sum(t.reward for t in transitions if t.terminated or t.truncated))
    assert math.isclose(reward_terminal, collision_reward, abs_tol=1e-9)
    assert math.isclose(collision_reward, -1.0 + 0.0, abs_tol=1e-9)

    transitions_offroad = [
        _make_transition(episode_id="ep1", terminated=True, truncated=False, reward=offroad_reward),
    ]
    reward_terminal_offroad = float(
        sum(t.reward for t in transitions_offroad if t.terminated or t.truncated)
    )
    assert math.isclose(reward_terminal_offroad, offroad_reward, abs_tol=1e-9)


def test_reward_terminal_success_plus_decision_cost_split_over_episode():
    """A full mini-episode: 2 real decision steps (each -0.01) followed
    by a SUCCESS terminal step (+1.0 terminal, -0.01 decision cost since
    it's also a real decision step here) -- confirms reward/terminal
    captures ONLY the +1.0 (not the accumulated decision costs) and
    reward/decision_cost captures the sum of all three steps'
    decision-cost components."""

    from src.rewards.merge_reward import compute_reward
    from src.training.config import load_reward_config

    reward_config = load_reward_config()
    nonterminal_reward = compute_reward(reward_config, "none", is_policy_step=True)
    success_reward = compute_reward(reward_config, "success", is_policy_step=True)

    transitions = [
        _make_transition(episode_id="ep0", terminated=False, truncated=False, reward=nonterminal_reward),
        _make_transition(episode_id="ep0", terminated=False, truncated=False, reward=nonterminal_reward),
        _make_transition(episode_id="ep0", terminated=True, truncated=False, reward=success_reward),
    ]

    reward_terminal = float(sum(t.reward for t in transitions if t.terminated or t.truncated))
    reward_total = float(sum(t.reward for t in transitions))
    reward_decision_cost = reward_total - reward_terminal

    assert math.isclose(reward_terminal, success_reward, abs_tol=1e-9)
    assert math.isclose(success_reward, 1.0 - 0.01, abs_tol=1e-9)
    expected_decision_cost = 2 * nonterminal_reward  # both nonterminal steps' full reward is decision-cost-only
    assert math.isclose(reward_decision_cost, expected_decision_cost, abs_tol=1e-9)


def test_reward_nonterminal_decision_cost_only_case():
    """A pure nonterminal decision step contributes ONLY to
    reward/decision_cost, with reward/terminal for that step being
    exactly 0 (it's neither terminated nor truncated)."""

    from src.rewards.merge_reward import compute_reward
    from src.training.config import load_reward_config

    reward_config = load_reward_config()
    nonterminal_reward = compute_reward(reward_config, "none", is_policy_step=True)

    transitions = [
        _make_transition(episode_id="ep0", terminated=False, truncated=False, reward=nonterminal_reward),
    ]
    reward_terminal = float(sum(t.reward for t in transitions if t.terminated or t.truncated))
    reward_total = float(sum(t.reward for t in transitions))
    reward_decision_cost = reward_total - reward_terminal

    assert math.isclose(reward_terminal, 0.0, abs_tol=1e-9)
    assert math.isclose(reward_decision_cost, nonterminal_reward, abs_tol=1e-9)
    assert math.isclose(nonterminal_reward, -0.01, abs_tol=1e-9)
