"""Shared rollout-capture helper for Phase 3 paper/PPT figures.

VISUALIZATION-ONLY. This module NEVER modifies MergeEnvironment,
BehaviorAction, the Frenet planner, LTV-MPC, or any frozen config --
it only CALLS the real, frozen production API
(``src.environment.merge_environment.MergeEnvironment``,
``src.environment.full_split_evaluator.load_maneuver_specs``) and
records the real per-step state it already exposes (via the public
``info`` dict) plus a few read-only internal attributes
(``env._state``, ``env._sdc_index``, ``env._polylines_by_id``,
``env._episode_context``) that MergeEnvironment does not otherwise
expose through its public step()/reset() return values, purely for
rendering (real roadgraph points, real oriented-box length/width,
real surrounding-agent positions). No internal attribute is ever
assigned to or mutated by this module.

Every figure script imports ``run_rollout`` (or the lower-level
``build_env``/``load_spec_by_id``) rather than re-implementing the
env-construction/stepping pattern, so every figure uses byte-identical
environment construction. This exactly mirrors the precedent already
established by ``scripts/audit_phase3_robustness.py`` (dataset config
path, MergeEnvironment construction, load_maneuver_specs usage).
"""

import dataclasses
from typing import List, Optional

import numpy as np

from src.environment.behavior_action import BehaviorAction
from src.environment.full_split_evaluator import load_maneuver_specs
from src.environment.merge_environment import MergeEnvironment

DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"


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
class StepSnapshot:
    """One real rollout frame's worth of state, everything sourced
    directly from MergeEnvironment's own public info dict plus the
    read-only current_sim_trajectory Waymax already carries for this
    frame (no synthetic/hand-drawn value anywhere)."""

    step_index: int  # 0 = the reset() frame (pre-first-step), 1.. = post-step
    ego_x: float
    ego_y: float
    ego_yaw: float
    ego_length: float
    ego_width: float
    ego_speed_mps: float
    agents: List[AgentSnapshot]
    active_source_lane_id: int
    active_target_lane_id: int
    downstream_status: Optional[str]
    intervention_rate: float
    frenet_d: Optional[float]
    terminated: bool
    truncated: bool
    termination_reason: Optional[str]
    info: dict


def build_env(downstream_mode: str = "frenet_mpc") -> MergeEnvironment:
    return MergeEnvironment(
        dataset_config_path=DATASET_CONFIG_PATH, downstream_mode=downstream_mode
    )


def load_spec_by_id(maneuver_id: str):
    """Finds one real ManeuverSpec by id across both TRAIN/VALIDATION
    canonical splits (searches TRAIN first, matching this repo's own
    convention that most named example maneuvers, e.g. MAN_0041/
    MAN_0071/MAN_0107, are TRAIN-split)."""

    for split in ("train", "validation"):
        for spec in load_maneuver_specs(split):
            if spec.maneuver_id == maneuver_id:
                return spec, split
    raise KeyError(f"maneuver_id {maneuver_id!r} not found in TRAIN or VALIDATION split")


def _snapshot_agents(env: MergeEnvironment) -> List[AgentSnapshot]:
    """Reads CURRENT-frame surrounding-agent state directly from the
    same Waymax simulator-state arrays MergeEnvironment's own
    ``_build_surrounding_agents`` reads (see merge_environment.py) --
    duplicated here read-only for rendering purposes (positions/
    length/width/yaw), never re-derived differently."""

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
    ego_length = float(np.asarray(traj.length)[i, 0])
    ego_width = float(np.asarray(traj.width)[i, 0])
    vel_x = float(np.asarray(traj.vel_x)[i, 0])
    vel_y = float(np.asarray(traj.vel_y)[i, 0])
    ego_speed = float(np.hypot(vel_x, vel_y))
    return ego_x, ego_y, ego_yaw, ego_length, ego_width, ego_speed


def run_rollout(
    maneuver_id: str,
    max_steps: int = 120,
    downstream_mode: str = "frenet_mpc",
    action=BehaviorAction.MERGE,
    env: Optional[MergeEnvironment] = None,
):
    """Runs one real, deterministic MergeEnvironment rollout on
    ``maneuver_id``, repeating ``action`` every step (default MERGE,
    matching Stage 3-G/3-H's own precedent script
    ``scripts/audit_phase3_robustness.py`` and
    ``tests/environment/test_merge_environment*.py``'s own
    success-test convention). This is a downstream QUALITATIVE
    demonstration rollout, NOT an FSM/PPO decision-quality result --
    every figure caption built from this must say so explicitly.

    Returns (spec, split, snapshots: List[StepSnapshot], polylines_by_id,
    env) so callers can also draw real roadgraph geometry via
    ``env._polylines_by_id`` / ``src.scenarios.lane_geometry``.
    """

    spec, split = load_spec_by_id(maneuver_id)
    if env is None:
        env = build_env(downstream_mode=downstream_mode)

    observation, info = env.reset(spec)
    snapshots = []

    ego_x, ego_y, ego_yaw, ego_length, ego_width, ego_speed = _snapshot_ego(env)
    snapshots.append(
        StepSnapshot(
            step_index=0,
            ego_x=ego_x, ego_y=ego_y, ego_yaw=ego_yaw,
            ego_length=ego_length, ego_width=ego_width, ego_speed_mps=ego_speed,
            agents=_snapshot_agents(env),
            active_source_lane_id=info["active_source_lane_id"],
            active_target_lane_id=info["active_target_lane_id"],
            downstream_status=info["downstream_status"],
            intervention_rate=info["intervention_rate"],
            frenet_d=float(observation[1]),
            terminated=False, truncated=False,
            termination_reason=None,
            info=info,
        )
    )

    terminated = truncated = False
    for step_idx in range(max_steps):
        observation, reward, terminated, truncated, info = env.step(action)
        ego_x, ego_y, ego_yaw, ego_length, ego_width, ego_speed = _snapshot_ego(env)
        snapshots.append(
            StepSnapshot(
                step_index=step_idx + 1,
                ego_x=ego_x, ego_y=ego_y, ego_yaw=ego_yaw,
                ego_length=ego_length, ego_width=ego_width, ego_speed_mps=ego_speed,
                agents=_snapshot_agents(env),
                active_source_lane_id=info["active_source_lane_id"],
                active_target_lane_id=info["active_target_lane_id"],
                downstream_status=info["downstream_status"],
                intervention_rate=info["intervention_rate"],
                frenet_d=float(observation[1]),
                terminated=bool(terminated), truncated=bool(truncated),
                termination_reason=info.get("termination_reason"),
                info=info,
            )
        )
        if terminated or truncated:
            break

    return spec, split, snapshots, env
