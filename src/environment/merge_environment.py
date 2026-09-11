"""Phase 2 Stage B-1: the first complete, working Merge Environment.

Wires the Stage B-0 foundation modules (observation_builder,
behavior_action, decision_state, termination) plus Stage B-1's new
episode_context (chained-maneuver runtime model), low_level_controller,
and merge_reference into Waymax's real
``waymax.env.planning_agent_environment.PlanningAgentEnvironment`` +
``waymax.dynamics.bicycle_model.InvertibleBicycleModel`` (Stage A/B-0
confirmed architecture; pinned Waymax revision
a64dfec9be8576b60d9cecc94f406d9812d4a7d0).

reset(maneuver_id) -> observation, info
step(action) -> observation, reward, terminated, truncated, info

This is a plain Python class, not a gymnasium.Env subclass: gymnasium
is not an existing dependency of this repository (confirmed via
requirements.txt), and Stage B-1's instructions explicitly say not to
add a new dependency unnecessarily. The (obs, reward, terminated,
truncated, info) return shape from ``step`` already matches the
modern Gymnasium API convention, so a thin Gymnasium wrapper could be
added later at near-zero cost if a training library specifically
requires it -- not needed for Stage B-1's own tests.

No reward is computed yet (Stage B-1 explicit scope: environment core
only, reward design/tuning is later). ``step`` returns
``reward=0.0`` unconditionally -- callers must not read anything into
this value yet.
"""

import dataclasses
from typing import Optional

import numpy as np
from waymax import config as waymax_config
from waymax import dynamics as waymax_dynamics
from waymax.datatypes import Action as WaymaxAction
from waymax.env.planning_agent_environment import PlanningAgentEnvironment

from src.environment.behavior_action import BehaviorAction, BehaviorExecutor
from src.environment.decision_state import DecisionState
from src.environment.episode_context import (
    EpisodeContext,
    parse_candidate_ids,
    parse_lane_chain,
)
from src.environment.low_level_controller import LowLevelController
from src.environment.merge_reference import build_merge_reference
from src.environment.observation_builder import ObservationInputs, build_observation
from src.environment.termination import (
    MAX_EPISODE_HORIZON_FRAMES,
    TerminationInputs,
    TerminationReason,
    check_online_causal_merge_success,
    check_termination,
)
from src.scenarios.lane_assignment import (
    LaneAssignmentConfig,
    load_lane_assignment_config,
)
from src.scenarios.lane_geometry import (
    extract_lane_polylines,
    project_point_to_polyline_signed,
)
from src.scenarios.merge_detector import (
    compute_merge_start_end_s,
    load_merge_topology_config,
)
from src.scenarios.scenario_features import (
    AgentSelectionConfig,
    load_agent_selection_config,
)
from src.scenarios.scenario_loader import (
    build_waymax_config,
    iter_scenarios,
    load_dataset_config,
    select_single_shard_for_inspection,
)

DEFAULT_MERGE_CONFIG_PATH = "configs/phase1_merge.yaml"


@dataclasses.dataclass(frozen=True)
class ManeuverSpec:
    """One row's worth of information from
    ``training_visual_merge_maneuvers.csv`` (or the equivalent
    validation table) -- everything ``MergeEnvironment.reset`` needs
    to locate and load the right scene, independent of how the caller
    obtained it (CSV row, in-memory list, a single hardcoded test
    fixture, etc.)."""

    maneuver_id: str
    source_shard: str
    record_index: int
    lane_chain: list
    candidate_ids: list
    merge_start_frame: int

    @staticmethod
    def from_csv_row(row: dict, manifest_by_candidate_id: dict) -> "ManeuverSpec":
        """Builds a ManeuverSpec from one
        training_visual_merge_maneuvers.csv row plus the
        merge_manifest*.csv rows needed to look up merge_start_frame
        for the FIRST candidate in the chain (Stage A: episode start
        = first candidate's merge_start_frame, unchanged)."""

        candidate_ids = parse_candidate_ids(row["candidate_ids"])
        lane_chain = parse_lane_chain(row["lane_chain"])
        first_candidate = manifest_by_candidate_id[candidate_ids[0]]
        return ManeuverSpec(
            maneuver_id=row["maneuver_id"],
            source_shard=row["source_shard"],
            record_index=int(row["record_index"]),
            lane_chain=lane_chain,
            candidate_ids=candidate_ids,
            merge_start_frame=int(first_candidate["merge_start_frame"]),
        )


class MergeEnvironment:
    """The Phase 2 Merge behavior-decision environment.

    One instance is reusable across many ``reset()`` calls (each
    ``reset`` loads whichever scene the requested maneuver belongs
    to); it holds no policy-specific state, only Waymax
    config/dynamics objects and Phase 1 config thresholds, so FSM and
    PPO share one instance safely.
    """

    def __init__(
        self,
        dataset_config_path: str,
        merge_config_path: str = DEFAULT_MERGE_CONFIG_PATH,
        max_num_objects: int = 64,
    ):
        self._expansion_config = load_dataset_config(dataset_config_path)
        self._merge_config_path = merge_config_path
        self._lane_assignment_config: LaneAssignmentConfig = (
            load_lane_assignment_config(merge_config_path)
        )
        self._merge_topology_config = load_merge_topology_config(merge_config_path)
        self._agent_selection_config: AgentSelectionConfig = (
            load_agent_selection_config(merge_config_path)
        )

        self._dynamics = waymax_dynamics.InvertibleBicycleModel(
            dt=0.1, max_accel=6.0, max_steering=0.3, normalize_actions=False
        )
        self._max_num_objects = max_num_objects
        self._executor = BehaviorExecutor()
        self._controller = LowLevelController()

        # Rebuilt fresh every reset() (init_steps depends on the
        # maneuver's own merge_start_frame -- see reset()), so no
        # single shared PlanningAgentEnvironment instance is kept.
        self._waymax_env: Optional[PlanningAgentEnvironment] = None

        # Mutable per-episode state, all set by reset().
        self._state = None
        self._episode_context: Optional[EpisodeContext] = None
        self._decision_state: Optional[DecisionState] = None
        self._sdc_id: Optional[int] = None
        self._sdc_index: Optional[int] = None
        self._polylines_by_id = None
        self._merge_end_s: Optional[float] = None
        self._steps_elapsed: int = 0

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, maneuver: ManeuverSpec, seed: Optional[int] = None):
        """Loads the maneuver's scene, initializes Waymax simulation
        state at the maneuver's merge_start_frame, and returns the
        initial (observation, info) pair.

        ``seed`` is accepted for interface compatibility but this
        environment has no stochastic component (deterministic scene
        load + deterministic dynamics) -- see the Stage B-1
        determinism tests.
        """

        del seed  # no stochastic component; accepted for API symmetry

        shard_path = select_single_shard_for_inspection(
            self._expansion_config,
            source_shard=maneuver.source_shard,
            record_index=maneuver.record_index,
        )
        dataset_config = build_waymax_config(self._expansion_config, shard_path)

        record = None
        for candidate in iter_scenarios(
            dataset_config,
            limit=maneuver.record_index + 1,
            source_dataset=self._expansion_config.dataset_name,
            source_split=self._expansion_config.split,
        ):
            record = candidate
        if record is None:
            raise RuntimeError(
                f"Could not load scenario for maneuver {maneuver.maneuver_id} "
                f"(shard={maneuver.source_shard}, record_index={maneuver.record_index})"
            )

        self._sdc_id = record.sdc_id
        self._sdc_index = record.sdc_index
        self._polylines_by_id = {
            polyline.lane_id: polyline
            for polyline in extract_lane_polylines(record.state.roadgraph_points)
        }

        self._episode_context = EpisodeContext(
            maneuver_id=maneuver.maneuver_id,
            scene_key=record.scene_key,
            candidate_ids=maneuver.candidate_ids,
            lane_chain=maneuver.lane_chain,
            merge_start_frame=maneuver.merge_start_frame,
        )
        self._decision_state = DecisionState()
        self._merge_end_s = self._compute_active_merge_end_s()

        # Waymax's PlanningAgentEnvironment.reset always starts the
        # controllable simulation at timestep = init_steps - 1 (Stage
        # B-1 investigation of the installed source: init_steps
        # defaults to 11, WOMD's fixed 10-warmup + 1-current
        # convention). To start controllable simulation exactly AT
        # the maneuver's own merge_start_frame (Stage A, unchanged
        # episode-start decision) rather than a fixed frame 10, this
        # environment passes a PER-EPISODE init_steps =
        # merge_start_frame + 1, which Waymax's own config explicitly
        # supports as a plain parameter (not hardcoded in the
        # library) -- confirmed by reading config.py before writing
        # this.
        init_steps = maneuver.merge_start_frame + 1
        env_config = waymax_config.EnvironmentConfig(
            max_num_objects=self._max_num_objects,
            init_steps=init_steps,
            controlled_object=waymax_config.ObjectType.SDC,
            compute_reward=False,
        )
        self._waymax_env = PlanningAgentEnvironment(
            dynamics_model=self._dynamics,
            config=env_config,
        )

        self._state = self._waymax_env.reset(record.state)
        self._steps_elapsed = 0

        observation = self._build_observation()
        info = self._build_info(
            requested_action=None,
            executed_action=None,
            fallback_applied=None,
        )
        return observation, info

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: BehaviorAction):
        """Advances the simulation by one Waymax timestep (dt=0.1s)
        under the given high-level behavior action, resolving MERGE
        commitment, chained-maneuver transition advancement, the
        shared behavior executor + low-level controller, Waymax's
        real bicycle-model ego dynamics + logged non-ego playback, and
        online-causal termination -- in that order.
        """

        if self._state is None:
            raise RuntimeError("step() called before reset()")

        # MERGE commitment (Stage B-0/B-1): once committed, the
        # caller's requested action is ignored for EXECUTION, but
        # still recorded in info for debugging (Section 7).
        requested_action = action
        executed_action = self._decision_state.advance(action)

        observation_before = self._build_observation()
        objective = self._executor.compute_objective(
            executed_action, observation_before
        )

        reference_polyline = self._resolve_reference_polyline(objective)

        ego_x, ego_y, ego_yaw, ego_speed = self._current_ego_pose_and_speed()
        command = self._controller.compute_command(
            reference_speed_mps=objective.reference_speed_mps,
            reference_polyline=reference_polyline,
            ego_x=ego_x,
            ego_y=ego_y,
            ego_yaw=ego_yaw,
            ego_speed_mps=ego_speed,
        )

        waymax_action = WaymaxAction(
            data=np.array(
                [command.acceleration_mps2, command.steering_curvature],
                dtype=np.float32,
            ),
            valid=np.array([True], dtype=bool),
        )
        self._state = self._waymax_env.step(self._state, waymax_action)
        self._steps_elapsed += 1

        chain_advanced = self._maybe_advance_chain()

        success = self._check_final_success()
        collision, offroad = self._compute_safety_metrics()

        termination_result = check_termination(
            TerminationInputs(
                success=success,
                collision=bool(collision),
                offroad=bool(offroad),
                steps_elapsed=self._steps_elapsed,
                max_episode_horizon=MAX_EPISODE_HORIZON_FRAMES,
            )
        )

        terminated = termination_result.reason in (
            TerminationReason.SUCCESS,
            TerminationReason.FAILURE_COLLISION,
            TerminationReason.FAILURE_OFFROAD,
        )
        truncated = (
            termination_result.reason == TerminationReason.TRUNCATION_HORIZON
        )

        observation = self._build_observation()
        info = self._build_info(
            requested_action=requested_action,
            executed_action=executed_action,
            fallback_applied=objective.fallback_applied,
            command=command,
            objective=objective,
            chain_advanced=chain_advanced,
            termination_reason=termination_result.reason,
        )

        reward = 0.0  # Stage B-1 scope: no reward yet.
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_ego_pose_and_speed(self):
        traj = self._state.current_sim_trajectory
        ego_x = float(np.asarray(traj.x)[self._sdc_index, 0])
        ego_y = float(np.asarray(traj.y)[self._sdc_index, 0])
        ego_yaw = float(np.asarray(traj.yaw)[self._sdc_index, 0])
        vel_x = float(np.asarray(traj.vel_x)[self._sdc_index, 0])
        vel_y = float(np.asarray(traj.vel_y)[self._sdc_index, 0])
        return ego_x, ego_y, ego_yaw, float(np.hypot(vel_x, vel_y))

    def _compute_active_merge_end_s(self) -> float:
        source_polyline = self._polylines_by_id[
            self._episode_context.active_source_lane_id
        ]
        target_polyline = self._polylines_by_id[
            self._episode_context.active_target_lane_id
        ]
        _, merge_end_s = compute_merge_start_end_s(
            source_polyline, target_polyline, self._merge_topology_config
        )
        return merge_end_s

    def _resolve_reference_polyline(self, objective):
        source_polyline = self._polylines_by_id[
            self._episode_context.active_source_lane_id
        ]
        target_polyline = self._polylines_by_id[
            self._episode_context.active_target_lane_id
        ]

        if objective.reference_lane == "source":
            return source_polyline

        # MERGE: blend from source toward target (Stage B-1 Section 5)
        # rather than an instant switch.
        ego_x, ego_y, _, _ = self._current_ego_pose_and_speed()
        return build_merge_reference(
            source_polyline, target_polyline, ego_x, ego_y
        ).polyline

    def _build_observation(self) -> np.ndarray:
        traj = self._state.current_sim_trajectory
        ego_x, ego_y, _, _ = self._current_ego_pose_and_speed()

        source_polyline = self._polylines_by_id[
            self._episode_context.active_source_lane_id
        ]
        target_polyline = self._polylines_by_id[
            self._episode_context.active_target_lane_id
        ]

        ego_source_projection = project_point_to_polyline_signed(
            source_polyline, ego_x, ego_y
        )["arc_length_m"]

        vel_x = np.asarray(traj.vel_x)[:, 0]
        vel_y = np.asarray(traj.vel_y)[:, 0]
        x = np.asarray(traj.x)[:, 0]
        y = np.asarray(traj.y)[:, 0]
        yaw = np.asarray(traj.yaw)[:, 0]
        length = np.asarray(traj.length)[:, 0]
        valid = np.asarray(traj.valid)[:, 0].astype(bool)

        object_ids = np.asarray(self._state.object_metadata.ids)
        object_types = np.asarray(self._state.object_metadata.object_types)

        inputs = ObservationInputs(
            ego_id=self._sdc_id,
            ego_x=float(x[self._sdc_index]),
            ego_y=float(y[self._sdc_index]),
            ego_vel_x=float(vel_x[self._sdc_index]),
            ego_vel_y=float(vel_y[self._sdc_index]),
            ego_length_m=float(length[self._sdc_index]),
            source_polyline=source_polyline,
            target_polyline=target_polyline,
            merge_end_s=self._merge_end_s,
            ego_source_arc_length_m=ego_source_projection,
            object_ids=object_ids,
            object_types=object_types,
            valid=valid,
            x=x,
            y=y,
            yaw=yaw,
            vel_x=vel_x,
            vel_y=vel_y,
            length=length,
            agent_selection_config=self._agent_selection_config,
        )
        return build_observation(inputs)

    def _maybe_advance_chain(self) -> bool:
        """Checks the ACTIVE target lane (not the final one) for
        stable entry and advances episode_context if it's an
        intermediate target (Stage B-1 Section 1's chained-maneuver
        model).

        Stage B-2.5 fix: gated on ``self._decision_state.is_committed``
        -- see ``_check_final_success`` below for why. Without this
        gate, a policy that never selects MERGE could still have its
        chain silently "advance" purely from incidental lane-keeping
        geometry, which is exactly the same causality violation as the
        final-success check.
        """

        if not self._decision_state.is_committed:
            return False

        traj = self._state.current_sim_trajectory
        current_lane_id = self._current_stable_target_check(
            self._episode_context.active_target_lane_id
        )
        if current_lane_id is None:
            return False
        advanced = self._episode_context.advance_if_intermediate_reached(
            self._episode_context.active_target_lane_id
        )
        if advanced:
            self._merge_end_s = self._compute_active_merge_end_s()
        return advanced

    def _current_stable_target_check(self, target_lane_id: int):
        """Returns target_lane_id if ego is ONLINE-CAUSALLY confirmed
        stably on it right now (reusing termination.py's success
        check against the ACTIVE target, not the final one), else
        None."""

        history = self._sim_trajectory_history()
        success = check_online_causal_merge_success(
            history["x"],
            history["y"],
            history["yaw"],
            history["valid"],
            target_lane_id=target_lane_id,
            polylines=list(self._polylines_by_id.values()),
            lane_assignment_config=self._lane_assignment_config,
            episode_start_frame=self._episode_context.merge_start_frame,
        )
        return target_lane_id if success else None

    def _check_final_success(self) -> bool:
        """Stage B-2.5 fix (Section 9): ``check_online_causal_merge_
        success`` is a purely GEOMETRIC lane-assignment check -- its
        own docstring (termination.py) already documents that it is
        "only meaningful once the episode is in MERGE_COMMITTED
        phase", but this precondition was never actually enforced
        here. A real bug was found via direct trace: a policy that
        NEVER selects MERGE (e.g. AlwaysKeep) could still have this
        check spuriously return True purely from incidental
        lane-keeping drift, whenever the source and target lane
        centerlines are close/converging near the merge point (as they
        necessarily are, by definition of a merge scenario) -- ego's
        stable lane assignment can cross into the target lane's
        geometric envelope without the policy ever having chosen to
        merge. Confirmed directly on MAN_0001: pure KEEP reached
        ``termination_reason=success`` at frame 13 while
        ``decision_phase`` stayed "decision" and ``merge_committed``
        stayed False the entire episode. Guarding on
        ``is_committed`` restores the documented precondition: success
        can only be evaluated -- let alone reported -- once the policy
        has actually committed to MERGE.
        """

        if not self._decision_state.is_committed:
            return False
        if not self._episode_context.is_on_final_transition:
            return False
        history = self._sim_trajectory_history()
        return check_online_causal_merge_success(
            history["x"],
            history["y"],
            history["yaw"],
            history["valid"],
            target_lane_id=self._episode_context.final_target_lane_id,
            polylines=list(self._polylines_by_id.values()),
            lane_assignment_config=self._lane_assignment_config,
            episode_start_frame=self._episode_context.merge_start_frame,
        )

    def _sim_trajectory_history(self):
        """The SIMULATED ego trajectory from episode start through
        the current timestep only -- never a future frame (causality
        requirement, Stage B-0 Section 6/8)."""

        current_timestep = int(self._state.timestep)
        start = self._episode_context.merge_start_frame
        traj = self._state.sim_trajectory

        x = np.asarray(traj.x)[self._sdc_index, start : current_timestep + 1]
        y = np.asarray(traj.y)[self._sdc_index, start : current_timestep + 1]
        yaw = np.asarray(traj.yaw)[self._sdc_index, start : current_timestep + 1]
        valid = np.asarray(traj.valid)[
            self._sdc_index, start : current_timestep + 1
        ].astype(bool)
        return {"x": x, "y": y, "yaw": yaw, "valid": valid}

    def _compute_safety_metrics(self):
        metrics = self._waymax_env.metrics(self._state)
        collision = bool(np.asarray(metrics["overlap"].value)) if "overlap" in metrics else False
        offroad = bool(np.asarray(metrics["offroad"].value)) if "offroad" in metrics else False
        return collision, offroad

    def _build_info(
        self,
        requested_action,
        executed_action,
        fallback_applied,
        command=None,
        objective=None,
        chain_advanced=None,
        termination_reason=None,
    ) -> dict:
        return {
            "requested_action": requested_action,
            "executed_action": executed_action,
            "decision_phase": self._decision_state.phase.value,
            "merge_committed": self._decision_state.is_committed,
            "fallback_applied": fallback_applied,
            "active_transition_index": self._episode_context.active_transition_index,
            "active_source_lane_id": self._episode_context.active_source_lane_id,
            "active_target_lane_id": self._episode_context.active_target_lane_id,
            "final_target_lane_id": self._episode_context.final_target_lane_id,
            "chain_advanced": chain_advanced,
            "reference_speed_mps": (
                objective.reference_speed_mps if objective else None
            ),
            "reference_lane": objective.reference_lane if objective else None,
            "controller_acceleration_mps2": (
                command.acceleration_mps2 if command else None
            ),
            "controller_steering_curvature": (
                command.steering_curvature if command else None
            ),
            "termination_reason": (
                termination_reason.value if termination_reason else None
            ),
            "steps_elapsed": self._steps_elapsed,
        }
