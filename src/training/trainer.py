"""PPO trainer loop: rollout -> GAE -> PPO update -> logging (docs/ppo/
PPO_PLAN.md SS0.1 P5).

P4 built ``build_training_batch``: rollout (``src.training.rollout``)
-> Reward V0 (``src.rewards``) -> GAE (``src.training.gae``) -> masked
Actor-side advantage normalization (SS7.2) -> a flat batch dict ready
for a PPO update.

P5 implements ``run_training``: the real multi-update PPO training
loop against the real ``MergeEnvironment``. Each update:

  1. collect a small rollout (``src.training.rollout.collect_rollout``)
     over the given maneuvers
  2. build a training batch (``build_training_batch``: rollout -> GAE
     -> masked-normalized advantages)
  3. filter every ACTOR-side quantity (observation, action, old
     log_prob, advantages, policy_mask) to ``policy_mask == 1`` rows
     ONLY (SS7.2 -- ``build_training_batch`` does not do this itself)
  4. run ``ppo_epochs`` inner-epoch passes over ``num_minibatches``
     minibatches of the masked Actor rows, each minibatch:
       - policy loss + entropy computed via ``jax.grad`` of
         ``ppo_clipped_surrogate_loss``/``entropy_bonus`` w.r.t. the
         POLICY train state's params only, applied via
         ``policy_state.apply_gradients``
       - value loss computed over the FULL (unmasked) trajectory
         (SS7.2 Critic scope) via ``jax.grad`` of ``value_loss``
         w.r.t. the VALUE train state's params only, applied via
         ``value_state.apply_gradients``
  5. log metrics (``src.tracking.wandb_logger``) and advance the
     global env-step / PPO-update-step counters

This is pipeline-verification only (PPO_PLAN.md SS0.1/P5): no
hyperparameter tuning, no long training, no TRAIN/TUNE split.
"""

import dataclasses
import time
from typing import Any, Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.environment.merge_environment import ManeuverSpec, MergeEnvironment
from src.policies.ppo import distribution
from src.policies.ppo.loss import entropy_bonus, ppo_clipped_surrogate_loss, value_loss
from src.policies.ppo.policy import PPOPolicy
from src.policies.ppo.state import PPOTrainingState
from src.training.config import PPOConfig, RewardConfig
from src.training.gae import compute_gae_segmented, normalize_advantages_masked
from src.training.rollout import Transition, collect_rollout


def build_training_batch(
    transitions: List[Transition], ppo_config: PPOConfig
) -> Dict[str, np.ndarray]:
    """Turns one rollout's flat ``Transition`` list into a PPO-ready
    training batch dict.

    Critic-side quantities (``returns``, ``values``, GAE's own
    ``advantages_raw``) are computed over the FULL physical trajectory,
    regardless of ``policy_mask`` (SS7.2). The batch's ``advantages``
    field is the Actor-side MASKED-NORMALIZED version (SS7.2: mean/std
    computed only over ``policy_mask == 1`` frames via
    ``normalize_advantages_masked``) -- a P5 update loop must further
    filter every Actor-side quantity (this batch's ``policy_mask``
    field, ``log_prob``, ``action``, ``observation``) down to
    ``policy_mask == 1`` rows before computing policy loss/entropy/KL/
    clip-fraction (SS7.2); this function does not do that filtering
    itself since the Critic-side arrays in the SAME batch must stay
    full-length.

    Returns a dict with keys: ``observation``, ``action``, ``reward``,
    ``next_observation``, ``terminated``, ``truncated``, ``value``,
    ``next_value``, ``log_prob``, ``policy_mask``, ``advantages_raw``
    (unmasked, full-trajectory GAE output -- kept for diagnostics/
    tests), ``advantages`` (masked-normalized, SS7.2), ``returns``
    (full-trajectory GAE returns, Critic target).
    """

    if len(transitions) == 0:
        raise ValueError("build_training_batch: got an empty transition list.")

    observations = np.stack([t.observation for t in transitions])
    next_observations = np.stack([t.next_observation for t in transitions])
    actions = np.asarray([t.action for t in transitions], dtype=np.int32)
    rewards = np.asarray([t.reward for t in transitions], dtype=np.float64)
    terminated = np.asarray([t.terminated for t in transitions], dtype=bool)
    truncated = np.asarray([t.truncated for t in transitions], dtype=bool)
    values = np.asarray([t.value for t in transitions], dtype=np.float64)
    next_values = np.asarray([t.next_value for t in transitions], dtype=np.float64)
    log_probs = np.asarray([t.log_prob for t in transitions], dtype=np.float64)
    policy_mask = np.asarray([t.policy_mask for t in transitions], dtype=np.int32)
    rollout_cutoff = np.asarray([t.rollout_cutoff for t in transitions], dtype=bool)
    old_logits = np.stack(
        [
            t.old_logits if t.old_logits is not None else np.zeros(4, dtype=np.float64)
            for t in transitions
        ]
    )
    # episode_id may legitimately be None (a caller that never set it,
    # e.g. some synthetic unit tests) -- fall back to a single implicit
    # segment (index 0 for every transition) rather than crashing, so
    # this Fix-1 segmentation is a strict superset of the old flat
    # compute_gae behavior whenever episode boundaries are unknown/unset.
    episode_ids = [t.episode_id for t in transitions]
    if all(eid is None for eid in episode_ids):
        episode_ids = [0] * len(transitions)

    hp = ppo_config.hyperparameters
    gae_result = compute_gae_segmented(
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminated=terminated,
        truncated=truncated,
        episode_ids=episode_ids,
        gamma=hp.gamma,
        gae_lambda=hp.gae_lambda,
        rollout_cutoff=rollout_cutoff,
    )

    advantages_normalized = normalize_advantages_masked(
        gae_result.advantages, policy_mask
    )

    batch = {
        "observation": observations,
        "action": actions,
        "reward": rewards,
        "next_observation": next_observations,
        "terminated": terminated,
        "truncated": truncated,
        "value": values,
        "next_value": next_values,
        "log_prob": log_probs,
        "policy_mask": policy_mask,
        "advantages_raw": gae_result.advantages,
        "advantages": advantages_normalized,
        "returns": gae_result.returns,
        # Fix 1 diagnostic (not consumed by GAE itself downstream --
        # segmentation already happened above): lets a caller/test
        # confirm which rows were artificial trainer-side cutoffs.
        "rollout_cutoff": rollout_cutoff,
        # Fix 2: full old-policy logits per row (zero sentinel at
        # policy_mask==0 rows), for the exact-KL diagnostic in run_update.
        "old_logits": old_logits,
    }

    for key, array in batch.items():
        if key == "rollout_cutoff":
            continue  # boolean diagnostic flag, not a numeric field
        if not np.all(np.isfinite(np.asarray(array, dtype=np.float64))):
            raise ValueError(
                f"build_training_batch: batch field {key!r} contains "
                "non-finite (NaN/inf) values."
            )

    return batch


def _aggregate_downstream_rates(transitions: List[Transition]) -> Dict[str, float]:
    """Fix 6 (W&B full diagnostics): downstream intervention-diagnostic
    RATES for this rollout, sourced exclusively from the frozen
    ``MergeEnvironment``'s own per-step ``info`` dict (carried through
    verbatim on ``Transition.info`` -- never re-derived here).

    ``MergeEnvironment._build_info`` already accumulates
    ``planner_infeasible_count``/``collision_blocked_count``/
    ``controller_failure_count``/``invalid_reference_count`` and
    ``intervention_rate`` as EPISODE-CUMULATIVE tallies (Stage 3-H SS3),
    reset at each ``env.reset()``. To get a per-rollout RATE (rather
    than double-counting across episodes, since each episode's count
    already includes every prior step of THAT episode), this function
    reads only each episode's LAST transition's ``info`` snapshot (the
    final cumulative count for that episode) and divides by that
    episode's own ``steps_elapsed`` -- then averages across the
    episodes present in this rollout, matching how ``intervention_rate``
    itself is already defined by the environment
    (count / steps_elapsed).

    Returns 0.0 for every field if no transition carries a non-None
    ``info`` (e.g. an older/synthetic transition list) rather than
    raising -- these are diagnostic-only metrics, never required for a
    PPO update to proceed.
    """

    episode_last_info: Dict[Any, dict] = {}
    for t in transitions:
        if t.info is not None:
            episode_last_info[t.episode_id] = t.info

    if not episode_last_info:
        return {
            "downstream/intervention_rate": 0.0,
            "downstream/planner_infeasible_rate": 0.0,
            "downstream/collision_blocked_rate": 0.0,
            "downstream/controller_failure_rate": 0.0,
            "downstream/invalid_reference_rate": 0.0,
        }

    intervention_rates = []
    planner_infeasible_rates = []
    collision_blocked_rates = []
    controller_failure_rates = []
    invalid_reference_rates = []
    for info in episode_last_info.values():
        steps_elapsed = max(1, int(info.get("steps_elapsed", 0) or 0))
        intervention_rates.append(float(info.get("intervention_rate", 0.0) or 0.0))
        planner_infeasible_rates.append(
            float(info.get("planner_infeasible_count", 0) or 0) / steps_elapsed
        )
        collision_blocked_rates.append(
            float(info.get("collision_blocked_count", 0) or 0) / steps_elapsed
        )
        controller_failure_rates.append(
            float(info.get("controller_failure_count", 0) or 0) / steps_elapsed
        )
        invalid_reference_rates.append(
            float(info.get("invalid_reference_count", 0) or 0) / steps_elapsed
        )

    return {
        "downstream/intervention_rate": float(np.mean(intervention_rates)),
        "downstream/planner_infeasible_rate": float(np.mean(planner_infeasible_rates)),
        "downstream/collision_blocked_rate": float(np.mean(collision_blocked_rates)),
        "downstream/controller_failure_rate": float(np.mean(controller_failure_rates)),
        "downstream/invalid_reference_rate": float(np.mean(invalid_reference_rates)),
    }


def _filter_actor_rows(batch: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Filters a training batch's Actor-relevant fields down to
    ``policy_mask == 1`` rows only (SS7.2). ``build_training_batch``
    deliberately does NOT do this itself (its Critic-side arrays must
    stay full-length in the same dict), so callers filter here.

    ``old_logits`` (Fix 2) is included here alongside the other
    already-Actor-scoped fields -- it is exclusively used for the
    exact-KL diagnostic, itself an Actor-side (policy_mask==1-only)
    computation per PPO_PLAN.md SS7.2.
    """

    return {
        "observation": batch["observation"][mask],
        "action": batch["action"][mask],
        "log_prob": batch["log_prob"][mask],
        "advantages": batch["advantages"][mask],
        "old_logits": batch["old_logits"][mask],
    }


def _exact_categorical_kl(old_logits: jnp.ndarray, new_logits: jnp.ndarray) -> jnp.ndarray:
    """Fix 2: exact full-distribution categorical KL(old || new),
    per-row, for a 4-action categorical policy -- monitoring-only
    diagnostic, never used as a loss term or for early stopping.

        old_log_probs = log_softmax(old_logits)
        new_log_probs = log_softmax(new_logits)
        old_probs      = exp(old_log_probs)
        exact_kl        = sum_a old_probs[a] * (old_log_probs[a] - new_log_probs[a])

    Args:
        old_logits: shape ``(batch, num_actions)``.
        new_logits: shape ``(batch, num_actions)``.

    Returns:
        Per-row exact KL, shape ``(batch,)``. Mathematically
        non-negative (Gibbs' inequality); may read as a tiny negative
        value (e.g. ``-1e-9``) purely from floating-point rounding when
        ``old_logits == new_logits`` -- callers must not treat that as
        an error, only clip/tolerate it for reporting.
    """

    old_log_probs = jax.nn.log_softmax(old_logits, axis=-1)
    new_log_probs = jax.nn.log_softmax(new_logits, axis=-1)
    old_probs = jnp.exp(old_log_probs)
    return jnp.sum(old_probs * (old_log_probs - new_log_probs), axis=-1)


def _minibatch_indices(num_rows: int, num_minibatches: int, rng: np.random.RandomState):
    """Shuffles ``num_rows`` row indices and splits them into
    ``num_minibatches`` (nearly) equal chunks. ``num_minibatches`` is
    clamped to ``num_rows`` so a tiny smoke batch never produces an
    empty minibatch."""

    num_minibatches = max(1, min(num_minibatches, num_rows))
    order = rng.permutation(num_rows)
    return np.array_split(order, num_minibatches)


def _policy_loss_fn(
    policy_params,
    policy_apply_fn,
    observations,
    old_log_prob,
    actions,
    advantages,
    old_logits,
    clip_epsilon,
    entropy_coef,
):
    logits = policy_apply_fn(policy_params, observations)
    new_log_prob = distribution.log_prob(logits, actions)
    policy_loss, surrogate_info = ppo_clipped_surrogate_loss(
        old_log_prob=old_log_prob,
        new_log_prob=new_log_prob,
        advantages=advantages,
        clip_epsilon=clip_epsilon,
    )
    entropy = entropy_bonus(logits)
    total = policy_loss - entropy_coef * entropy
    # Fix 2: exact categorical KL diagnostic -- monitoring only, never
    # added to `total` (the PPO-Clip objective itself is unchanged).
    exact_kl_per_row = _exact_categorical_kl(old_logits, logits)
    info = {
        "policy_loss": policy_loss,
        "entropy": entropy,
        "exact_kl_mean": jnp.mean(exact_kl_per_row),
        "exact_kl_max": jnp.max(exact_kl_per_row),
        **surrogate_info,
    }
    return total, info


def _value_loss_fn(value_params, value_apply_fn, observations, returns):
    values = value_apply_fn(value_params, observations)
    v_loss = value_loss(values, returns)
    return v_loss, {"value_loss": v_loss, "values": values}


def run_update(
    training_state: PPOTrainingState,
    batch: Dict[str, np.ndarray],
    ppo_config: PPOConfig,
    numpy_rng: np.random.RandomState,
) -> Dict[str, Any]:
    """Runs ONE PPO update (``ppo_epochs`` inner-epoch passes over
    ``num_minibatches`` minibatches) against one already-built training
    batch, returning ``(new_training_state, metrics)``.

    Per SS7.2: Actor-side quantities (policy loss, entropy, approx-KL,
    exact-KL, clip-fraction, action-distribution stats) are computed
    ONLY over ``policy_mask == 1`` rows; the value loss uses the FULL
    trajectory (every row, regardless of ``policy_mask``).

    Fix 3 (pre-P6 hardening): every ``ppo/*``/``policy_grad_norm``/
    ``value_grad_norm`` metric below is aggregated (mean, and for
    grad-norms also max) across the FULL ``ppo_epochs x
    num_minibatches`` sweep of this update -- not just the last
    minibatch/epoch's value. Explicit ``_mean``/``_max`` suffixes are
    used throughout (rather than silently redefining what a bare
    ``ppo/policy_loss`` key means) so this is purely additive to the
    existing metric-name contract; see ``run_training`` for how these
    are merged into the final per-update metrics dict actually logged
    to W&B.

    Fix 4 (value_coef): the policy and value networks/optimizers are
    fully INDEPENDENT ``TrainState``s with separate ``apply_gradients``
    calls (see ``src/policies/ppo/state.py`` -- no shared parameters,
    per SS7.2's Actor/Critic separation). The value loss below is never
    summed with the policy loss into one combined gradient, so
    ``ppo_config.hyperparameters.value_coef`` has NO EFFECT on training
    dynamics in this architecture (PPO paper SS6.1: c1/value_coef only
    matters when policy and value losses are combined into one shared
    gradient, which requires shared or jointly-optimized parameters).
    ``value_coef`` is intentionally never referenced in this function's
    body -- see ``PPOHyperparameters.value_coef``'s docstring in
    ``src/training/config.py`` for the full deprecation note.
    """

    hp = ppo_config.hyperparameters
    policy_mask = batch["policy_mask"].astype(bool)
    if not np.any(policy_mask):
        raise ValueError(
            "run_update: batch has no policy_mask==1 rows -- cannot run "
            "any Actor-side PPO update."
        )

    actor_batch = _filter_actor_rows(batch, policy_mask)
    num_actor_rows = actor_batch["observation"].shape[0]
    num_critic_rows = batch["observation"].shape[0]

    policy_state = training_state.policy_state
    value_state = training_state.value_state

    policy_grad_fn = jax.value_and_grad(_policy_loss_fn, has_aux=True)
    value_grad_fn = jax.value_and_grad(_value_loss_fn, has_aux=True)

    # Fix 3: accumulate EVERY minibatch's diagnostic values across the
    # whole ppo_epochs x num_minibatches sweep, not just the last one.
    policy_loss_values: List[float] = []
    value_loss_values: List[float] = []
    entropy_values: List[float] = []
    approx_kl_values: List[float] = []
    exact_kl_mean_values: List[float] = []
    exact_kl_max_values: List[float] = []
    clip_fraction_values: List[float] = []
    policy_grad_norms: List[float] = []
    value_grad_norms: List[float] = []

    for _epoch in range(hp.ppo_epochs):
        # --- Actor update: policy_mask==1 rows only, minibatched.
        for idx in _minibatch_indices(num_actor_rows, hp.num_minibatches, numpy_rng):
            mb_obs = jnp.asarray(actor_batch["observation"][idx])
            mb_old_log_prob = jnp.asarray(actor_batch["log_prob"][idx])
            mb_actions = jnp.asarray(actor_batch["action"][idx])
            mb_advantages = jnp.asarray(actor_batch["advantages"][idx])
            mb_old_logits = jnp.asarray(actor_batch["old_logits"][idx])

            (_loss, policy_info), grads = policy_grad_fn(
                policy_state.params,
                policy_state.apply_fn,
                mb_obs,
                mb_old_log_prob,
                mb_actions,
                mb_advantages,
                mb_old_logits,
                hp.clip_epsilon,
                hp.entropy_coef,
            )
            policy_grad_norm = float(optax.global_norm(grads))
            policy_grad_norms.append(policy_grad_norm)
            policy_state = policy_state.apply_gradients(grads=grads)

            policy_loss_values.append(float(policy_info["policy_loss"]))
            entropy_values.append(float(policy_info["entropy"]))
            approx_kl_values.append(float(policy_info["approx_kl"]))
            exact_kl_mean_values.append(float(policy_info["exact_kl_mean"]))
            exact_kl_max_values.append(float(policy_info["exact_kl_max"]))
            clip_fraction_values.append(float(policy_info["clip_fraction"]))

        # --- Critic update: FULL trajectory, minibatched.
        for idx in _minibatch_indices(num_critic_rows, hp.num_minibatches, numpy_rng):
            mb_obs = jnp.asarray(batch["observation"][idx])
            mb_returns = jnp.asarray(batch["returns"][idx])

            (_loss, value_info), grads = value_grad_fn(
                value_state.params,
                value_state.apply_fn,
                mb_obs,
                mb_returns,
            )
            value_grad_norm = float(optax.global_norm(grads))
            value_grad_norms.append(value_grad_norm)
            value_state = value_state.apply_gradients(grads=grads)

            value_loss_values.append(float(value_info["value_loss"]))

    new_training_state = dataclasses.replace(
        training_state, policy_state=policy_state, value_state=value_state
    )

    # Action-distribution stats: policy_mask==1 rows only (SS7.2/SS8).
    actor_actions = actor_batch["action"]
    action_counts = {i: int(np.sum(actor_actions == i)) for i in range(4)}
    total_actions = max(1, int(actor_actions.shape[0]))

    # exact_kl can read as a tiny negative value purely from
    # floating-point rounding (see _exact_categorical_kl's docstring) --
    # clip only for reporting, never treat as an error.
    exact_kl_mean_values = [max(0.0, v) for v in exact_kl_mean_values]
    exact_kl_max_values = [max(0.0, v) for v in exact_kl_max_values]

    # Fix 6: ppo/explained_variance -- standard
    # 1 - Var(returns - values) / Var(returns), using the Critic's
    # PRE-update value predictions (batch["value"], already collected
    # during rollout) against the GAE returns, over the FULL trajectory
    # (Critic scope, SS7.2 -- never masked to policy_mask==1). Handles
    # Var(returns) ~= 0 safely (a degenerate/near-constant-return batch,
    # e.g. a single-step smoke rollout) by reporting 0.0 rather than
    # dividing by ~0 into a NaN/inf.
    returns_full = batch["returns"]
    values_full = batch["value"]
    returns_var = float(np.var(returns_full))
    if returns_var < 1e-8:
        explained_variance = 0.0
    else:
        explained_variance = float(
            1.0 - np.var(returns_full - values_full) / returns_var
        )

    metrics = {
        "ppo/policy_loss_mean": float(np.mean(policy_loss_values)),
        "ppo/value_loss_mean": float(np.mean(value_loss_values)),
        "ppo/entropy_mean": float(np.mean(entropy_values)),
        "ppo/approx_kl_mean": float(np.mean(approx_kl_values)),
        "ppo/exact_kl_mean": float(np.mean(exact_kl_mean_values)),
        "ppo/exact_kl_max": float(np.max(exact_kl_max_values)),
        "ppo/clip_fraction_mean": float(np.mean(clip_fraction_values)),
        "ppo/policy_grad_norm_mean": float(np.mean(policy_grad_norms)),
        "ppo/policy_grad_norm_max": float(np.max(policy_grad_norms)),
        "ppo/value_grad_norm_mean": float(np.mean(value_grad_norms)),
        "ppo/value_grad_norm_max": float(np.max(value_grad_norms)),
        # Backward-compatible bare keys (mean-of-full-sweep, same
        # aggregation fix as the _mean-suffixed keys above) -- kept so
        # any existing caller/test/dashboard reading the original P5
        # bare-key names still sees a CORRECTLY aggregated value rather
        # than losing the key outright.
        "ppo/policy_loss": float(np.mean(policy_loss_values)),
        "ppo/value_loss": float(np.mean(value_loss_values)),
        "ppo/entropy": float(np.mean(entropy_values)),
        "ppo/approx_kl": float(np.mean(approx_kl_values)),
        "ppo/clip_fraction": float(np.mean(clip_fraction_values)),
        "ppo/grad_norm": float(np.mean(policy_grad_norms + value_grad_norms)),
        "action/keep_ratio": action_counts[0] / total_actions,
        "action/follow_ratio": action_counts[1] / total_actions,
        "action/merge_ratio": action_counts[2] / total_actions,
        "action/stop_ratio": action_counts[3] / total_actions,
        "ppo/explained_variance": explained_variance,
    }

    if not all(np.isfinite(v) for v in metrics.values()):
        raise ValueError(f"run_update produced a non-finite metric: {metrics}")

    return {"training_state": new_training_state, "metrics": metrics}


def run_training(
    ppo_config: PPOConfig,
    reward_config: RewardConfig,
    env: MergeEnvironment,
    maneuvers: List[ManeuverSpec],
    training_state: PPOTrainingState,
    rng_key: jax.Array,
    numpy_rng: np.random.RandomState,
    num_updates: int,
    max_steps_per_episode: int,
    global_env_step: int = 0,
    ppo_update_step: int = 0,
    wandb_logger: Optional[Any] = None,
    on_update: Optional[Callable[[int, PPOTrainingState, jax.Array, int, int], None]] = None,
) -> Dict[str, Any]:
    """Runs ``num_updates`` real PPO updates against the real
    ``MergeEnvironment`` (docs/ppo/PPO_PLAN.md SS0.1/P5).

    Each update: collect one rollout over ``maneuvers`` -> build a
    training batch (rollout -> GAE -> masked-normalized advantages) ->
    filter Actor-side quantities to ``policy_mask == 1`` -> run
    ``ppo_epochs``/``num_minibatches`` gradient steps for both the
    policy and value train states (independent parameters, SS7.2) ->
    log metrics.

    ``global_env_step``/``ppo_update_step`` are the counters to CONTINUE
    from (0 for a fresh run, or whatever a resumed checkpoint recorded)
    -- this is what makes ``--resume`` a real continuation rather than a
    restart (PPO_PLAN.md SS10). ``on_update`` is an optional callback
    invoked after every update with
    ``(update_index, training_state, global_env_step, ppo_update_step)``
    -- used by callers (e.g. ``scripts/train_ppo.py``) to save periodic
    checkpoints without this function needing to know about the
    checkpoint contract itself.

    Returns a dict with the final ``training_state``, the final
    ``rng_key`` (the advanced PRNG state after all rollouts -- callers
    should persist THIS into a checkpoint, never the key they passed
    in, so a resumed run continues the RNG stream rather than replaying
    it), the final ``global_env_step``/``ppo_update_step``, and a list
    of per-update metrics dicts (``updates``).
    """

    update_metrics: List[Dict[str, Any]] = []

    for update_index in range(num_updates):
        rng_key, rollout_key = jax.random.split(rng_key)

        ppo_policy = PPOPolicy(training_state.policy_network, training_state.policy_state.params)
        # Fix 6 (runtime/env_steps_per_sec): wall-clock throughput of
        # ROLLOUT COLLECTION specifically (the LTV-MPC solve inside
        # env.step dominates per-step cost, per the P0 audit) -- timed
        # around collect_rollout only, never including the PPO update
        # itself, so this metric reflects environment-interaction
        # throughput, not training throughput.
        rollout_start_time = time.time()
        transitions: List[Transition] = collect_rollout(
            env=env,
            maneuvers=maneuvers,
            ppo_policy=ppo_policy,
            value_network=training_state.value_network,
            value_params=training_state.value_state.params,
            reward_config=reward_config,
            rng_key=rollout_key,
            max_steps_per_episode=max_steps_per_episode,
        )
        rollout_elapsed = max(1e-9, time.time() - rollout_start_time)
        env_steps_per_sec = float(len(transitions) / rollout_elapsed)
        global_env_step += len(transitions)

        batch = build_training_batch(transitions, ppo_config)

        update_result = run_update(training_state, batch, ppo_config, numpy_rng)
        training_state = update_result["training_state"]
        ppo_update_step += 1

        episode_returns = {}
        for t in transitions:
            episode_returns.setdefault(t.episode_id, 0.0)
            episode_returns[t.episode_id] += t.reward
        mean_episode_return = float(np.mean(list(episode_returns.values()))) if episode_returns else 0.0

        # Fix 7 follow-up (reward component logging correctness): the
        # PRIOR version of this aggregation approximated reward/terminal
        # by summing the FULL `t.reward` of every terminated/truncated
        # row -- but `t.reward` on such a row is terminal_component +
        # decision_cost_component combined (see
        # src/rewards/merge_reward.py), so e.g. a SUCCESS (+1.0) landing
        # on a real policy-decision step (-0.01 decision cost) dumped
        # the whole +0.99 into reward/terminal instead of splitting it
        # into +1.0 terminal / -0.01 decision cost. Each `Transition`
        # now carries its own exact `reward_terminal_component`/
        # `reward_decision_cost_component`, propagated verbatim from
        # `MergeRewardWrapper.compute`'s per-step
        # `last_terminal_component`/`last_decision_cost_component`
        # (never re-derived from `terminated`/`truncated` here) --
        # summing those directly is now exact, including on an
        # artificial rollout_cutoff row (which the wrapper already
        # correctly gave a 0.0 terminal component, since the
        # environment's own termination_reason was non-terminal there).
        reward_terminal = float(
            sum(t.reward_terminal_component for t in transitions)
        )
        reward_decision_cost = float(
            sum(t.reward_decision_cost_component for t in transitions)
        )
        reward_total = float(np.sum(batch["reward"]))

        # Fix 6 (W&B full diagnostics): episode-outcome rates, sourced
        # exclusively from the frozen environment's own
        # `info["termination_reason"]` (Fix 6's additive
        # ``Transition.info`` pass-through of the real post-step info
        # dict -- SS5.1: never re-derived, only read verbatim) on each
        # episode's LAST transition. Falls back to a terminated-sign
        # heuristic (positive terminal reward => success-like,
        # non-positive => failure-like) ONLY for a `Transition` that
        # carries no `info` (e.g. an older/synthetic transition list
        # predating this field) so this aggregation degrades gracefully
        # rather than crashing.
        episode_last_transition: Dict[Any, Transition] = {}
        for t in transitions:
            episode_last_transition[t.episode_id] = t
        num_episodes = max(1, len(episode_last_transition))
        num_success = 0
        num_collision = 0
        num_offroad = 0
        num_timeout = 0
        for t in episode_last_transition.values():
            reason = t.info.get("termination_reason") if t.info is not None else None
            if reason == "success":
                num_success += 1
            elif reason == "failure_collision":
                num_collision += 1
            elif reason == "failure_offroad":
                num_offroad += 1
            elif reason == "truncation_horizon" or t.truncated:
                num_timeout += 1
            elif t.terminated:
                # info unavailable -- fall back to the reward's sign.
                if t.reward > 0:
                    num_success += 1
                else:
                    num_collision += 1
        train_success_rate = num_success / num_episodes
        train_collision_rate = num_collision / num_episodes
        train_offroad_rate = num_offroad / num_episodes
        train_timeout_rate = num_timeout / num_episodes

        # policy_decision_count / physical_step_count (Fix 6): raw
        # counts of policy_mask==1 rows vs. the full physical trajectory
        # -- directly from Transition.policy_mask, never re-derived.
        policy_decision_count = int(np.sum(batch["policy_mask"]))
        physical_step_count = int(len(transitions))

        # ppo/explained_variance (Fix 6): already computed inside
        # run_update (using the same batch["returns"]/batch["value"]
        # Critic-scope inputs, Var(returns)~=0 handled safely there) --
        # not recomputed here to avoid two slightly-differently-tuned
        # near-zero-variance thresholds disagreeing; it flows through
        # via `**update_result["metrics"]` below.

        # downstream/* diagnostics (Fix 6): sourced from the frozen
        # environment's own per-step `info` fields (intervention_rate,
        # planner_infeasible/collision_blocked/controller_failure/
        # invalid_reference counts) -- carried through via each
        # episode's LAST transition's `info_after` snapshot, which is
        # already an episode-CUMULATIVE tally per
        # MergeEnvironment._build_info's own docstring (Stage 3-H SS3),
        # never re-derived here. Transition does not itself carry the
        # raw `info` dict (only the fixed SS0.1/P4 field set does), so
        # this reads it from `rollout.Transition`'s optional `info`
        # snapshot when present (Fix 6 additive field on Transition;
        # falls back to 0.0/rate-unavailable if a caller's
        # Transition predates that field, e.g. a synthetic test batch).
        downstream_rates = _aggregate_downstream_rates(transitions)

        metrics = {
            "train/episode_return": mean_episode_return,
            "train/episode_length": float(len(transitions) / max(1, len(episode_returns))),
            "train/success_rate": train_success_rate,
            "train/collision_rate": train_collision_rate,
            "train/offroad_rate": train_offroad_rate,
            "train/timeout_rate": train_timeout_rate,
            "reward/terminal": reward_terminal,
            "reward/decision_cost": reward_decision_cost,
            "reward/total": reward_total,
            "train/policy_decision_count": float(policy_decision_count),
            "train/physical_step_count": float(physical_step_count),
            "runtime/env_steps_per_sec": env_steps_per_sec,
            **downstream_rates,
            **update_result["metrics"],
        }

        if not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError(f"run_training: non-finite metric at update {update_index}: {metrics}")

        if wandb_logger is not None:
            wandb_logger.log_metrics(metrics, step=ppo_update_step)

        update_metrics.append(metrics)

        if on_update is not None:
            on_update(update_index, training_state, rng_key, global_env_step, ppo_update_step)

    return {
        "training_state": training_state,
        "rng_key": rng_key,
        "global_env_step": global_env_step,
        "ppo_update_step": ppo_update_step,
        "updates": update_metrics,
    }
