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
from src.training.gae import compute_gae, normalize_advantages_masked
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

    hp = ppo_config.hyperparameters
    gae_result = compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminated=terminated,
        truncated=truncated,
        gamma=hp.gamma,
        gae_lambda=hp.gae_lambda,
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
    }

    for key, array in batch.items():
        if not np.all(np.isfinite(np.asarray(array, dtype=np.float64))):
            raise ValueError(
                f"build_training_batch: batch field {key!r} contains "
                "non-finite (NaN/inf) values."
            )

    return batch


def _filter_actor_rows(batch: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Filters a training batch's Actor-relevant fields down to
    ``policy_mask == 1`` rows only (SS7.2). ``build_training_batch``
    deliberately does NOT do this itself (its Critic-side arrays must
    stay full-length in the same dict), so callers filter here."""

    return {
        "observation": batch["observation"][mask],
        "action": batch["action"][mask],
        "log_prob": batch["log_prob"][mask],
        "advantages": batch["advantages"][mask],
    }


def _minibatch_indices(num_rows: int, num_minibatches: int, rng: np.random.RandomState):
    """Shuffles ``num_rows`` row indices and splits them into
    ``num_minibatches`` (nearly) equal chunks. ``num_minibatches`` is
    clamped to ``num_rows`` so a tiny smoke batch never produces an
    empty minibatch."""

    num_minibatches = max(1, min(num_minibatches, num_rows))
    order = rng.permutation(num_rows)
    return np.array_split(order, num_minibatches)


def _policy_loss_fn(policy_params, policy_apply_fn, observations, old_log_prob, actions, advantages, clip_epsilon, entropy_coef):
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
    info = {
        "policy_loss": policy_loss,
        "entropy": entropy,
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
    clip-fraction, action-distribution stats) are computed ONLY over
    ``policy_mask == 1`` rows; the value loss uses the FULL trajectory
    (every row, regardless of ``policy_mask``).
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

    last_policy_info: Dict[str, Any] = {}
    last_value_info: Dict[str, Any] = {}
    policy_grad_norms: List[float] = []
    value_grad_norms: List[float] = []

    for _epoch in range(hp.ppo_epochs):
        # --- Actor update: policy_mask==1 rows only, minibatched.
        for idx in _minibatch_indices(num_actor_rows, hp.num_minibatches, numpy_rng):
            mb_obs = jnp.asarray(actor_batch["observation"][idx])
            mb_old_log_prob = jnp.asarray(actor_batch["log_prob"][idx])
            mb_actions = jnp.asarray(actor_batch["action"][idx])
            mb_advantages = jnp.asarray(actor_batch["advantages"][idx])

            (_loss, last_policy_info), grads = policy_grad_fn(
                policy_state.params,
                policy_state.apply_fn,
                mb_obs,
                mb_old_log_prob,
                mb_actions,
                mb_advantages,
                hp.clip_epsilon,
                hp.entropy_coef,
            )
            policy_grad_norms.append(float(optax.global_norm(grads)))
            policy_state = policy_state.apply_gradients(grads=grads)

        # --- Critic update: FULL trajectory, minibatched.
        for idx in _minibatch_indices(num_critic_rows, hp.num_minibatches, numpy_rng):
            mb_obs = jnp.asarray(batch["observation"][idx])
            mb_returns = jnp.asarray(batch["returns"][idx])

            (_loss, last_value_info), grads = value_grad_fn(
                value_state.params,
                value_state.apply_fn,
                mb_obs,
                mb_returns,
            )
            value_grad_norms.append(float(optax.global_norm(grads)))
            value_state = value_state.apply_gradients(grads=grads)

    new_training_state = dataclasses.replace(
        training_state, policy_state=policy_state, value_state=value_state
    )

    # Action-distribution stats: policy_mask==1 rows only (SS7.2/SS8).
    actor_actions = actor_batch["action"]
    action_counts = {i: int(np.sum(actor_actions == i)) for i in range(4)}
    total_actions = max(1, int(actor_actions.shape[0]))

    metrics = {
        "ppo/policy_loss": float(last_policy_info["policy_loss"]),
        "ppo/value_loss": float(last_value_info["value_loss"]),
        "ppo/entropy": float(last_policy_info["entropy"]),
        "ppo/approx_kl": float(last_policy_info["approx_kl"]),
        "ppo/clip_fraction": float(last_policy_info["clip_fraction"]),
        "ppo/grad_norm": float(np.mean(policy_grad_norms + value_grad_norms)),
        "action/keep_ratio": action_counts[0] / total_actions,
        "action/follow_ratio": action_counts[1] / total_actions,
        "action/merge_ratio": action_counts[2] / total_actions,
        "action/stop_ratio": action_counts[3] / total_actions,
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

        # reward/terminal: the terminal-outcome component only (the
        # final transition's reward on a truly-terminated episode --
        # SUCCESS/COLLISION/OFFROAD, per Reward V0's fixed table).
        # reward/decision_cost: every step's reward MINUS that terminal
        # component (i.e. every step's -0.01/0.0 decision-cost
        # component, summed over the whole rollout).
        reward_terminal = float(sum(t.reward for t in transitions if t.terminated))
        reward_total = float(np.sum(batch["reward"]))
        reward_decision_cost = reward_total - reward_terminal

        metrics = {
            "train/episode_return": mean_episode_return,
            "train/episode_length": float(len(transitions) / max(1, len(episode_returns))),
            "reward/terminal": reward_terminal,
            "reward/decision_cost": reward_decision_cost,
            "reward/total": reward_total,
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
