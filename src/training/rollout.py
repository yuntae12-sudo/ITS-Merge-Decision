"""Rollout collection against the real ``MergeEnvironment`` (docs/ppo/
PPO_PLAN.md SS0.1 P4, SS7.1).

P4 implementation: connects the real P3 PPO core
(``src.policies.ppo.policy.PPOPolicy`` + a value network/params) to the
real ``MergeEnvironment`` (``downstream_mode="frenet_mpc"`` per SS2),
producing a list of ``Transition``s with the fixed field set:

    observation, action, reward, next_observation, terminated,
    truncated, value, next_value, log_prob, policy_mask

Optional diagnostics: ``episode_id``, ``maneuver_id``, ``step_index``.

``policy_mask`` / ``is_policy_step`` decision timing (SS7.1, the most
safety-critical part of this module): decided BEFORE ``env.step()`` is
called, from the info dict returned by the PREVIOUS call to
``reset()``/``step()`` (never from the info dict this same step's
``env.step()`` call is about to return):

    is_policy_step = not info_before["merge_committed"]
    policy_mask = 1 if is_policy_step else 0

    if is_policy_step:
        action_index, log_prob = ppo_policy.act(observation, rng_key)
        action = ACTION_INDEX_TO_BEHAVIOR[int(action_index)]
    else:
        action = BehaviorAction.MERGE   # auto-execute; policy NOT called
        log_prob = <log_prob PPO would have assigned MERGE, for bookkeeping>

    next_obs, _, terminated, truncated, info_after = env.step(action)
    reward = reward_wrapper.compute(info_after["termination_reason"],
                                     is_policy_step)

This mirrors the exact pattern already used by
``src.environment.full_split_evaluator.run_episode``
(``was_committed_before = info.get("merge_committed", False)`` read
BEFORE calling ``policy.decide``/``env.step``), confirmed during P0 to
be directly supported by ``MergeEnvironment``'s real
``reset()``/``step()`` API with no code change.

On an auto-execution step (``is_policy_step is False``), the PPO
policy network is NOT called at all (SS7.1: "the environment doesn't
actually consult the policy once committed") -- ``BehaviorAction.MERGE``
is submitted directly, matching ``full_split_evaluator.py``'s
precedent. The ``log_prob`` recorded for such a step is still a
well-defined number (the log-probability the CURRENT policy assigns to
MERGE at that observation) purely so ``Transition.log_prob`` is never
``None``/NaN -- SS7.2 already excludes ``policy_mask == 0`` frames from
every Actor-side computation (policy loss, entropy, KL, clip-fraction,
advantage-norm), so this value is never actually used by any Actor
computation; it exists only to satisfy the fixed non-Optional field
contract and to keep this module free of special-cased None-handling
downstream.
"""

import dataclasses
from typing import Any, List, Optional

import jax
import numpy as np

from src.environment.behavior_action import BehaviorAction
from src.environment.merge_environment import ManeuverSpec, MergeEnvironment
from src.policies.ppo import distribution
from src.policies.ppo.networks import NUM_ACTIONS
from src.policies.ppo.policy import PPOPolicy
from src.rewards.reward_wrapper import MergeRewardWrapper
from src.training.config import RewardConfig

# Minimum required transition fields (PPO_PLAN.md SS0.1/P4).
REQUIRED_TRANSITION_FIELDS = (
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
)


@dataclasses.dataclass(frozen=True)
class Transition:
    """One rollout transition. Optional diagnostic fields default to
    ``None`` so P4 can populate them without breaking this contract.

    Pre-P6 hardening (Fix 1/Fix 2) additive diagnostic fields, purely
    for GAE-boundary correctness and the exact-KL diagnostic -- neither
    changes any existing field's meaning:

    - ``rollout_cutoff``: ``True`` iff this transition is the LAST
      transition of its episode AND the episode ended because
      ``collect_episode_rollout``'s ``for step_index in range(max_steps)``
      loop ran out of steps WITHOUT the environment itself returning
      ``terminated``/``truncated`` (an artificial trainer-side cutoff,
      never a reason to synthesize a terminal reward -- SS5.1 stays the
      sole source of truth). ``False`` for every other transition,
      including a real environment ``terminated``/``truncated`` step.
      GAE must treat this exactly like ``truncated`` for bootstrapping
      (bootstrap from ``next_value``) and exactly like any episode
      boundary for trace-continuation (never let advantage leak into
      the next episode).
    - ``old_logits``: the full 4-logit vector the CURRENT policy
      assigned at this observation, at rollout-collection time, for
      ``policy_mask == 1`` decision frames (a sentinel all-zero vector
      for ``policy_mask == 0`` frames, which the exact-KL diagnostic
      excludes from its aggregate -- mirroring the existing
      ``policy_mask == 0`` Actor-side exclusion pattern from P4). Never
      resampled/re-forward-passed beyond what already computes
      ``log_prob`` -- same logits, just kept in full rather than
      reduced to one scalar.
    - ``info``: the raw POST-step ``info_after`` dict returned by this
      step's ``env.step()`` call (Fix 6, pre-P6 hardening), carried
      through verbatim/unmodified so a trainer can read the frozen
      environment's own downstream-diagnostic fields
      (``intervention_rate``, ``planner_infeasible_count``,
      ``collision_blocked_count``, ``controller_failure_count``,
      ``invalid_reference_count``, ``downstream_status``) for W&B
      logging WITHOUT this module (or any caller) re-deriving any of
      those values itself -- purely a pass-through snapshot, never
      recomputed. ``None`` for any ``Transition`` constructed without
      it (e.g. an older/synthetic test batch), so downstream-diagnostic
      aggregation must treat a ``None`` info as "not available" rather
      than assuming zero.
    """

    observation: Any
    action: Any
    reward: float
    next_observation: Any
    terminated: bool
    truncated: bool
    value: float
    next_value: float
    log_prob: float
    policy_mask: int
    episode_id: Optional[str] = None
    maneuver_id: Optional[str] = None
    step_index: Optional[int] = None
    rollout_cutoff: bool = False
    old_logits: Optional[np.ndarray] = None
    info: Optional[dict] = None


def _value_of(value_network, value_params, observation: np.ndarray) -> float:
    """Evaluates the Critic network for one 14D observation, returning
    a plain Python float (never a jax array), so ``Transition.value``/
    ``next_value`` are plain numbers usable by ``gae.compute_gae``
    (which itself works in plain NumPy)."""

    value = value_network.apply(value_params, observation)
    return float(np.asarray(value))


def collect_episode_rollout(
    env: MergeEnvironment,
    maneuver: ManeuverSpec,
    ppo_policy: PPOPolicy,
    value_network,
    value_params,
    reward_config: RewardConfig,
    rng_key: jax.Array,
    max_steps: int,
    episode_id: Optional[str] = None,
) -> List[Transition]:
    """Runs ONE episode of ``maneuver`` against ``env`` under
    ``ppo_policy``, producing a list of ``Transition``s covering the
    FULL physical trajectory (every frame, regardless of
    ``policy_mask`` -- SS7.2's Critic scope requires this; the Actor-
    side masking happens later, in loss/advantage-normalization code,
    not by dropping frames here).

    ``policy_mask``/reward decision timing follows SS7.1 exactly (see
    module docstring). Stops at ``terminated``, ``truncated``, or
    ``max_steps`` physical steps, whichever comes first.
    """

    observation, info_before = env.reset(maneuver)
    reward_wrapper = MergeRewardWrapper(reward_config)

    transitions: List[Transition] = []

    for step_index in range(max_steps):
        # --- SS7.1: policy_mask/is_policy_step decided PRE-step, from
        # the info dict returned by the PREVIOUS reset()/step() call
        # (info_before) -- never from this step's own env.step() result.
        is_policy_step = not info_before.get("merge_committed", False)
        policy_mask = 1 if is_policy_step else 0

        rng_key, action_key = jax.random.split(rng_key)

        if is_policy_step:
            action_index, log_prob = ppo_policy.act(observation, action_key)
            action_index = int(np.asarray(action_index))
            action = distribution.ACTION_INDEX_TO_BEHAVIOR[action_index]
            log_prob = float(np.asarray(log_prob))
            # Fix 2 (exact categorical KL diagnostic): keep the full
            # logit vector the CURRENT (behavior) policy assigned here,
            # not just the sampled action's log_prob -- no extra
            # forward pass beyond ppo_policy.act's own logits() call.
            old_logits = np.asarray(ppo_policy.logits(observation), dtype=np.float64)
        else:
            # SS7.1: the environment does not actually consult the
            # policy once committed -- submit MERGE directly, mirroring
            # full_split_evaluator.run_episode's precedent. log_prob is
            # still computed (never called into env.step, only used for
            # the Transition's non-Optional field -- see module
            # docstring) so this stays a well-defined finite number that
            # SS7.2's Actor-side masking will exclude from every loss/
            # entropy/KL/advantage-norm computation downstream.
            action = BehaviorAction.MERGE
            action_index = int(BehaviorAction.MERGE)
            logits = ppo_policy.logits(observation)
            log_prob = float(
                np.asarray(distribution.log_prob(logits, action_index))
            )
            # Fix 2: policy_mask==0 frames get a zero-vector sentinel,
            # never a real logits vector -- excluded from the exact-KL
            # aggregate at update time, mirroring the existing
            # policy_mask==0 Actor-side exclusion pattern.
            old_logits = np.zeros((NUM_ACTIONS,), dtype=np.float64)

        value = _value_of(value_network, value_params, observation)

        next_observation, _env_reward, terminated, truncated, info_after = env.step(
            action
        )
        del _env_reward  # MergeEnvironment.step always returns 0.0 (Stage B-1
        # scope note in merge_environment.py) -- Reward V0 is computed
        # below from the SAME termination_reason/is_policy_step source
        # of truth (SS5.1), never taken from this discarded value.

        next_value = _value_of(value_network, value_params, next_observation)

        reward = reward_wrapper.compute(
            termination_reason=info_after.get("termination_reason"),
            is_policy_step=is_policy_step,
            info=info_after,
        )

        # Fix 1 (episode-aware GAE correctness): an ARTIFICIAL rollout
        # cutoff is when this is the LAST step of the ``max_steps`` loop
        # (step_index == max_steps - 1) AND the environment itself did
        # NOT report terminated/truncated for this step -- i.e. the
        # trainer's own step budget ran out, not the environment's own
        # termination logic. This must be treated exactly like
        # ``truncated`` for GAE bootstrapping (bootstrap from
        # next_value) and as an episode/segment boundary for trace
        # continuation (never let advantage leak into the next
        # episode's rewards) -- but it is NEVER a reason to synthesize
        # a SUCCESS/COLLISION/OFFROAD/TRUNCATION_HORIZON reward; the
        # reward above was already computed from the environment's own
        # termination_reason (SS5.1), untouched by this flag.
        is_last_loop_iteration = step_index == max_steps - 1
        rollout_cutoff = bool(
            is_last_loop_iteration and not terminated and not truncated
        )

        transitions.append(
            Transition(
                observation=observation,
                action=action_index,
                reward=reward,
                next_observation=next_observation,
                terminated=bool(terminated),
                truncated=bool(truncated),
                value=value,
                next_value=next_value,
                log_prob=log_prob,
                policy_mask=policy_mask,
                episode_id=episode_id,
                maneuver_id=maneuver.maneuver_id,
                step_index=step_index,
                rollout_cutoff=rollout_cutoff,
                old_logits=old_logits,
                info=info_after,
            )
        )

        observation = next_observation
        info_before = info_after

        if terminated or truncated:
            break

    return transitions


def collect_rollout(
    env: MergeEnvironment,
    maneuvers: List[ManeuverSpec],
    ppo_policy: PPOPolicy,
    value_network,
    value_params,
    reward_config: RewardConfig,
    rng_key: jax.Array,
    max_steps_per_episode: int,
) -> List[Transition]:
    """Runs one episode per maneuver in ``maneuvers`` (in order),
    concatenating every episode's transitions into one flat list. Each
    episode gets its own ``episode_id`` (``f"ep{i}_{maneuver_id}"``) and
    a fresh RNG sub-key so episodes are independent."""

    all_transitions: List[Transition] = []
    for episode_index, maneuver in enumerate(maneuvers):
        rng_key, episode_key = jax.random.split(rng_key)
        episode_transitions = collect_episode_rollout(
            env=env,
            maneuver=maneuver,
            ppo_policy=ppo_policy,
            value_network=value_network,
            value_params=value_params,
            reward_config=reward_config,
            rng_key=episode_key,
            max_steps=max_steps_per_episode,
            episode_id=f"ep{episode_index}_{maneuver.maneuver_id}",
        )
        all_transitions.extend(episode_transitions)

    return all_transitions
