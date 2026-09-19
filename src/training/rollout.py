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
    ``None`` so P4 can populate them without breaking this contract."""

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
