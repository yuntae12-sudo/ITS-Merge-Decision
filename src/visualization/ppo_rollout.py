"""Real, deterministic-by-default PPO checkpoint rollout for
VISUALIZATION/DIAGNOSTIC use only (docs/ppo/PPO_VISUALIZATION_GUIDE.md).

This module is NOT part of the training pipeline. It never modifies
``MergeEnvironment``, ``BehaviorAction`` semantics, Reward V0, PPO
hyperparameters/architecture, the Frenet planner, or LTV-MPC -- it only
calls the real, frozen production API
(``MergeEnvironment(downstream_mode="frenet_mpc")``,
``src.training.rollout``'s own commitment contract) and records
whatever the environment/policy already expose.

MERGE-commitment / policy-consultation semantics are copied VERBATIM
from ``src.training.rollout.collect_episode_rollout`` (the single
source of truth for "when is the PPO policy actually consulted"):

    is_policy_step = not info_before["merge_committed"]

    if is_policy_step:
        <policy decides KEEP/FOLLOW/MERGE/STOP>
    else:
        action = BehaviorAction.MERGE   # auto-executed, NOT a fresh
                                         # policy decision -- the
                                         # policy is not even called
                                         # for probabilities in this
                                         # module either (see
                                         # PPOStepRecord.action_probs
                                         # note below).

The only difference from ``collect_episode_rollout`` is what gets
RECORDED per step (full render/trace-ready snapshots, action
probabilities, reward components, 14D observation fields, vehicle
pose) -- never a difference in which action is executed when.
"""

import dataclasses
from typing import Any, List, Optional

import jax
import numpy as np

from src.environment.behavior_action import BehaviorAction
from src.environment.merge_environment import ManeuverSpec, MergeEnvironment
from src.environment.observation_builder import OBSERVATION_FIELD_NAMES
from src.policies.ppo import distribution
from src.policies.ppo.policy import PPOPolicy
from src.rewards.reward_wrapper import MergeRewardWrapper
from src.training.config import RewardConfig

OUTCOME_SUCCESS = "success"
OUTCOME_COLLISION = "collision"
OUTCOME_OFFROAD = "offroad"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_EXCEPTION = "exception"

# Maps MergeEnvironment's own info["termination_reason"] values (the
# sole source of truth -- see src/environment/termination.py) to the
# visualization pipeline's outcome-directory categories. This is a
# renaming/grouping ONLY (never a re-judgment of what counts as a
# failure): "failure_collision" -> "collision", "failure_offroad" ->
# "offroad", "success" -> "success". A `truncated` episode (any
# termination_reason of "truncation_horizon", or no terminal reason at
# all within max_steps) is always "timeout".
_TERMINATION_REASON_TO_OUTCOME = {
    "success": OUTCOME_SUCCESS,
    "failure_collision": OUTCOME_COLLISION,
    "failure_offroad": OUTCOME_OFFROAD,
}


@dataclasses.dataclass
class AgentSnapshot:
    agent_id: int
    x: float
    y: float
    yaw: float
    length: float
    width: float
    velocity_x_mps: float
    velocity_y_mps: float


@dataclasses.dataclass
class PPOStepRecord:
    """One physical step's worth of real rollout data -- everything
    sourced directly from MergeEnvironment's own public info dict / the
    14D observation / the restored PPO policy's own outputs, plus the
    read-only ``current_sim_trajectory`` Waymax already carries for
    this frame (mirrors ``scripts/figures/_rollout.py``'s
    ``StepSnapshot`` precedent). No synthetic/hand-drawn/interpolated
    value anywhere.
    """

    step_index: int  # 0 = the reset() frame (pre-first-step)
    maneuver_id: str

    is_policy_step: bool
    merge_committed_before: bool
    selected_action: str  # BehaviorAction.name
    action_index: int

    # Action probabilities: only well-defined as a POLICY DECISION on a
    # real decision step (is_policy_step=True). On an auto-executed
    # post-commitment step, the policy is not consulted for the
    # decision (matches src.training.rollout exactly) -- these fields
    # are populated with NaN and must never be read as if they were a
    # policy decision for that frame (see module + field docs).
    keep_prob: float
    follow_prob: float
    merge_prob: float
    stop_prob: float
    value_estimate: Optional[float]

    # Reward (Reward V0, computed the same way src.training.rollout
    # computes it: MergeRewardWrapper driven by this step's real
    # termination_reason + this step's real is_policy_step).
    reward_total: float
    reward_terminal_component: float
    reward_decision_cost_component: float

    # 14D observation (OBSERVATION_FIELD_NAMES order), the PRE-step
    # observation the decision above was made from.
    observation: np.ndarray

    # Environment / downstream diagnostics (post-step info, verbatim).
    downstream_status: Optional[str]
    intervention_rate: float
    fallback_applied: Optional[str]
    terminated: bool
    truncated: bool
    termination_reason: Optional[str]

    # Vehicle pose (post-step; step_index==0 is the reset()-frame pose).
    ego_x: float
    ego_y: float
    ego_yaw: float
    ego_speed_mps: float

    agents: List[AgentSnapshot]
    active_source_lane_id: Optional[int]
    active_target_lane_id: Optional[int]

    info: dict
    dataset_schema_version: str = "merge_geometry_v1_legacy"
    maneuver_type: Optional[str] = None
    current_stable_lane_id: Optional[int] = None
    v2_saw_relevant_interaction: bool = False


@dataclasses.dataclass
class PPOEpisodeResult:
    maneuver_id: str
    outcome: str  # OUTCOME_SUCCESS | OUTCOME_COLLISION | OUTCOME_OFFROAD | OUTCOME_TIMEOUT | OUTCOME_EXCEPTION
    termination_reason: Optional[str]
    steps: List[PPOStepRecord]
    physical_step_count: int
    policy_decision_count: int
    merge_commit_step: Optional[int]  # first step_index where merge_committed_before became True on the NEXT record
    final_intervention_rate: float
    exception_repr: Optional[str] = None


def _snapshot_agents(env: MergeEnvironment) -> List[AgentSnapshot]:
    """Reads CURRENT-frame surrounding-agent state directly from the
    same Waymax simulator-state arrays MergeEnvironment's own
    observation/reward machinery reads -- duplicated here read-only for
    rendering, byte-identical to ``scripts/figures/_rollout.py``'s
    ``_snapshot_agents`` (kept as a separate copy per that module's own
    stated convention of not sharing fixtures/helpers across
    visualization call sites)."""

    traj = env._state.current_sim_trajectory
    x = np.asarray(traj.x)[:, 0]
    y = np.asarray(traj.y)[:, 0]
    yaw = np.asarray(traj.yaw)[:, 0]
    length = np.asarray(traj.length)[:, 0]
    width = np.asarray(traj.width)[:, 0]
    vel_x = np.asarray(traj.vel_x)[:, 0]
    vel_y = np.asarray(traj.vel_y)[:, 0]
    valid = np.asarray(traj.valid)[:, 0].astype(bool)
    object_ids = np.asarray(env._state.object_metadata.ids)

    agents = []
    for i in range(object_ids.shape[0]):
        if not valid[i] or int(object_ids[i]) == env._sdc_id:
            continue
        agents.append(
            AgentSnapshot(
                agent_id=int(object_ids[i]),
                x=float(x[i]),
                y=float(y[i]),
                yaw=float(yaw[i]),
                length=float(length[i]),
                width=float(width[i]),
                velocity_x_mps=float(vel_x[i]),
                velocity_y_mps=float(vel_y[i]),
            )
        )
    return agents


def _snapshot_ego(env: MergeEnvironment):
    traj = env._state.current_sim_trajectory
    i = env._sdc_index
    ego_x = float(np.asarray(traj.x)[i, 0])
    ego_y = float(np.asarray(traj.y)[i, 0])
    ego_yaw = float(np.asarray(traj.yaw)[i, 0])
    vel_x = float(np.asarray(traj.vel_x)[i, 0])
    vel_y = float(np.asarray(traj.vel_y)[i, 0])
    ego_speed = float(np.hypot(vel_x, vel_y))
    return ego_x, ego_y, ego_yaw, ego_speed


def _select_action(
    policy: PPOPolicy,
    observation: np.ndarray,
    policy_mode: str,
    rng_key: Optional[jax.Array],
):
    """Chooses an action index the same way ``PPOPolicy`` itself
    supports (``act_deterministic`` / ``act``) -- never a new inference
    path. ``policy_mode`` is ``"deterministic"`` (default, argmax via
    ``act_deterministic``) or ``"stochastic"`` (categorical sample via
    ``act``, requires ``rng_key``)."""

    if policy_mode == "deterministic":
        action_index = int(np.asarray(policy.act_deterministic(observation)))
    elif policy_mode == "stochastic":
        if rng_key is None:
            raise ValueError("stochastic policy_mode requires an rng_key")
        action_index, _log_prob = policy.act(observation, rng_key)
        action_index = int(np.asarray(action_index))
    else:
        raise ValueError(f"Unknown policy_mode: {policy_mode!r}")
    return action_index


def run_ppo_episode(
    env: MergeEnvironment,
    maneuver: ManeuverSpec,
    policy: PPOPolicy,
    value_network: Any,
    value_params: Any,
    reward_config: RewardConfig,
    max_steps: int,
    policy_mode: str = "deterministic",
    rng_key: Optional[jax.Array] = None,
) -> PPOEpisodeResult:
    """Runs ONE real, closed-loop ``MergeEnvironment`` episode under a
    restored PPO checkpoint's policy, recording a full render/trace-
    ready ``PPOStepRecord`` per physical step.

    Decision/commitment semantics are byte-identical to
    ``src.training.rollout.collect_episode_rollout`` (see module
    docstring) -- this function differs ONLY in what it records, never
    in which action gets executed on which step.
    """

    try:
        observation, info_before = env.reset(maneuver)
    except Exception as exc:  # noqa: BLE001 -- one bad maneuver must not
        # crash a whole outcome scan; recorded as its own category,
        # mirroring full_split_evaluator.run_episode's precedent.
        return PPOEpisodeResult(
            maneuver_id=maneuver.maneuver_id,
            outcome=OUTCOME_EXCEPTION,
            termination_reason=None,
            steps=[],
            physical_step_count=0,
            policy_decision_count=0,
            merge_commit_step=None,
            final_intervention_rate=0.0,
            exception_repr=f"{type(exc).__name__}: {exc}",
        )

    reward_wrapper = MergeRewardWrapper(reward_config)

    records: List[PPOStepRecord] = []
    policy_decision_count = 0
    merge_commit_step: Optional[int] = None
    outcome = OUTCOME_TIMEOUT
    termination_reason: Optional[str] = None
    final_intervention_rate = 0.0

    try:
        for step_index in range(max_steps):
            is_policy_step = not info_before.get("merge_committed", False)
            merge_committed_before = bool(info_before.get("merge_committed", False))

            keep_prob = follow_prob = merge_prob = stop_prob = float("nan")
            value_estimate: Optional[float] = None

            if is_policy_step:
                if rng_key is not None:
                    rng_key, action_key = jax.random.split(rng_key)
                else:
                    action_key = None
                action_index = _select_action(policy, observation, policy_mode, action_key)
                action = distribution.ACTION_INDEX_TO_BEHAVIOR[action_index]

                logits = np.asarray(policy.logits(observation))
                action_probs = np.asarray(distribution.probs(logits))
                keep_prob = float(action_probs[0])
                follow_prob = float(action_probs[1])
                merge_prob = float(action_probs[2])
                stop_prob = float(action_probs[3])

                policy_decision_count += 1
                if merge_commit_step is None and action == BehaviorAction.MERGE:
                    merge_commit_step = step_index
            else:
                # Post-commitment auto-execution -- NOT a fresh PPO
                # decision (src.training.rollout's own contract). The
                # policy is deliberately NOT consulted for
                # probabilities here either, so a diagnostic-only
                # frame can never be mistaken for an actual decision.
                action = BehaviorAction.MERGE
                action_index = int(BehaviorAction.MERGE)

            if value_network is not None and value_params is not None:
                value_estimate = float(
                    np.asarray(value_network.apply(value_params, observation))
                )

            next_observation, _env_reward, terminated, truncated, info_after = env.step(action)
            del _env_reward  # MergeEnvironment.step always returns 0.0; Reward V0
            # is computed below from the same termination_reason/is_policy_step
            # source of truth src.training.rollout uses (never re-derived
            # differently for visualization).

            reward = reward_wrapper.compute(
                termination_reason=info_after.get("termination_reason"),
                is_policy_step=is_policy_step,
                info=info_after,
            )
            reward_terminal_component = reward_wrapper.last_terminal_component
            reward_decision_cost_component = reward_wrapper.last_decision_cost_component

            ego_x, ego_y, ego_yaw, ego_speed = _snapshot_ego(env)

            records.append(
                PPOStepRecord(
                    step_index=step_index,
                    maneuver_id=maneuver.maneuver_id,
                    is_policy_step=is_policy_step,
                    merge_committed_before=merge_committed_before,
                    selected_action=action.name,
                    action_index=action_index,
                    keep_prob=keep_prob,
                    follow_prob=follow_prob,
                    merge_prob=merge_prob,
                    stop_prob=stop_prob,
                    value_estimate=value_estimate,
                    reward_total=reward,
                    reward_terminal_component=reward_terminal_component,
                    reward_decision_cost_component=reward_decision_cost_component,
                    observation=np.asarray(observation, dtype=np.float64),
                    downstream_status=info_after.get("downstream_status"),
                    intervention_rate=float(info_after.get("intervention_rate", 0.0)),
                    fallback_applied=info_after.get("fallback_applied"),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    termination_reason=info_after.get("termination_reason"),
                    ego_x=ego_x,
                    ego_y=ego_y,
                    ego_yaw=ego_yaw,
                    ego_speed_mps=ego_speed,
                    agents=_snapshot_agents(env),
                    active_source_lane_id=info_after.get("active_source_lane_id"),
                    active_target_lane_id=info_after.get("active_target_lane_id"),
                    info=info_after,
                    dataset_schema_version=info_after.get(
                        "dataset_schema_version", "merge_geometry_v1_legacy"
                    ),
                    maneuver_type=info_after.get("maneuver_type"),
                    current_stable_lane_id=info_after.get("current_stable_lane_id"),
                    v2_saw_relevant_interaction=bool(
                        info_after.get("v2_saw_relevant_interaction", False)
                    ),
                )
            )

            final_intervention_rate = float(info_after.get("intervention_rate", 0.0))
            observation = next_observation
            info_before = info_after

            if terminated:
                termination_reason = info_after.get("termination_reason")
                outcome = _TERMINATION_REASON_TO_OUTCOME.get(termination_reason, OUTCOME_TIMEOUT)
                break
            if truncated:
                termination_reason = info_after.get("termination_reason")
                outcome = OUTCOME_TIMEOUT
                break
        else:
            outcome = OUTCOME_TIMEOUT
    except Exception as exc:  # noqa: BLE001 -- see reset()'s try/except above.
        return PPOEpisodeResult(
            maneuver_id=maneuver.maneuver_id,
            outcome=OUTCOME_EXCEPTION,
            termination_reason=termination_reason,
            steps=records,
            physical_step_count=len(records),
            policy_decision_count=policy_decision_count,
            merge_commit_step=merge_commit_step,
            final_intervention_rate=final_intervention_rate,
            exception_repr=f"{type(exc).__name__}: {exc}",
        )

    return PPOEpisodeResult(
        maneuver_id=maneuver.maneuver_id,
        outcome=outcome,
        termination_reason=termination_reason,
        steps=records,
        physical_step_count=len(records),
        policy_decision_count=policy_decision_count,
        merge_commit_step=merge_commit_step,
        final_intervention_rate=final_intervention_rate,
    )


__all__ = [
    "AgentSnapshot",
    "PPOStepRecord",
    "PPOEpisodeResult",
    "run_ppo_episode",
    "OUTCOME_SUCCESS",
    "OUTCOME_COLLISION",
    "OUTCOME_OFFROAD",
    "OUTCOME_TIMEOUT",
    "OUTCOME_EXCEPTION",
    "OBSERVATION_FIELD_NAMES",
]
