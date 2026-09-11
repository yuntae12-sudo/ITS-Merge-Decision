"""Tests for src/environment/merge_environment.py -- the Stage B-1
Environment Core.

Unlike the rest of tests/scenarios/ and tests/environment/, these
tests load REAL WOMD scenes (via the already-downloaded local
training shards under data/womd/training/, the same 10-shard pool
Phase 1's training-pool construction used) rather than synthetic
geometry: the whole point of Stage B-1 is verifying real Waymax
dynamics wiring (PlanningAgentEnvironment + InvertibleBicycleModel),
which cannot be meaningfully exercised without a real
SimulatorState/roadgraph. This matches how
scripts/inspect_merge_candidate.py and similar Phase 1 CLIs already
depend on local WOMD data being present -- these tests will fail (not
skip) if outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml
and the shards it points to are not present locally.
"""

import numpy as np
import pytest

from src.environment.behavior_action import BehaviorAction
from src.environment.merge_environment import MergeEnvironment, ManeuverSpec

DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"

# A real single-candidate (non-chained) HIGH-confidence maneuver from
# the training pool: training_tfexample.tfrecord-00000-of-01000#12,
# lane_chain 637->628, merge_start_frame=36 (from
# outputs/phase1/training_10shard_pilot/merge_manifest_training_scratch.csv).
SINGLE_MANEUVER = ManeuverSpec(
    maneuver_id="MAN_0001",
    source_shard="training_tfexample.tfrecord-00000-of-01000",
    record_index=12,
    lane_chain=[637, 628],
    candidate_ids=[
        "training_tfexample.tfrecord-00000-of-01000#12__t49__637_628"
    ],
    merge_start_frame=36,
)

# A real single-candidate maneuver with clearer lateral separation
# between source/target lanes (endpoint_target_distance_m ~5.0), used
# for the action-causality tests where a visible lateral divergence
# matters: training_tfexample.tfrecord-00001-of-01000#214,
# lane_chain 527->544, merge_start_frame=62.
CAUSALITY_MANEUVER = ManeuverSpec(
    maneuver_id="MAN_CAUSALITY",
    source_shard="training_tfexample.tfrecord-00001-of-01000",
    record_index=214,
    lane_chain=[527, 544],
    candidate_ids=[
        "training_tfexample.tfrecord-00001-of-01000#214__t75__527_544"
    ],
    merge_start_frame=62,
)

# A real chained (2-transition) maneuver:
# training_tfexample.tfrecord-00003-of-01000#216, lane_chain
# 618->623->608, from training_visual_merge_maneuvers.csv (MAN_0041).
#
# NOTE on controller robustness (Stage B-1 disclosed limitation): the
# deliberately minimal shared low-level controller (a plain P
# heading/lateral tracker, no Frenet planner/MPC -- out of Stage B-1
# scope by explicit instruction) does not reliably complete MERGE on
# every real maneuver's geometry -- a spot check across 8 real
# single-candidate maneuvers found 5/8 reached SUCCESS under MERGE and
# 3/8 ended in a real (non-spurious) failure_offroad/failure_collision
# on sharply curving lane geometry the controller cannot track well.
# This maneuver was specifically selected because it is CONFIRMED (by
# direct rollout) to reach success reliably, so the chained-maneuver
# structural tests below verify the chain-advancement/final-success
# LOGIC is correct, without being confounded by the controller's known
# tracking limits on other, harder maneuvers.
CHAINED_MANEUVER = ManeuverSpec(
    maneuver_id="MAN_0041",
    source_shard="training_tfexample.tfrecord-00003-of-01000",
    record_index=216,
    lane_chain=[618, 623, 608],
    candidate_ids=[
        "training_tfexample.tfrecord-00003-of-01000#216__t38__618_623",
        "training_tfexample.tfrecord-00003-of-01000#216__t54__623_608",
    ],
    merge_start_frame=11,
)


@pytest.fixture(scope="module")
def env():
    return MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)


# ======================================================================
# reset()
# ======================================================================


def test_reset_returns_finite_14d_observation(env):

    observation, info = env.reset(SINGLE_MANEUVER)

    assert observation.shape == (14,)
    assert np.all(np.isfinite(observation))


def test_reset_info_reflects_initial_decision_state(env):

    _, info = env.reset(SINGLE_MANEUVER)

    assert info["decision_phase"] == "decision"
    assert info["merge_committed"] is False
    assert info["active_transition_index"] == 0
    assert info["active_source_lane_id"] == 637
    assert info["active_target_lane_id"] == 628
    assert info["final_target_lane_id"] == 628


def test_reset_chained_maneuver_starts_at_first_transition(env):

    _, info = env.reset(CHAINED_MANEUVER)

    assert info["active_transition_index"] == 0
    assert info["active_source_lane_id"] == 618
    assert info["active_target_lane_id"] == 623
    assert info["final_target_lane_id"] == 608  # NOT the intermediate lane


# ======================================================================
# step() basic shape / info
# ======================================================================


def test_step_returns_five_tuple_with_finite_observation(env):

    env.reset(SINGLE_MANEUVER)
    observation, reward, terminated, truncated, info = env.step(
        BehaviorAction.KEEP
    )

    assert observation.shape == (14,)
    assert np.all(np.isfinite(observation))
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_info_records_requested_and_executed_action(env):

    env.reset(SINGLE_MANEUVER)
    _, _, _, _, info = env.step(BehaviorAction.FOLLOW)

    assert info["requested_action"] == BehaviorAction.FOLLOW
    assert info["executed_action"] == BehaviorAction.FOLLOW


# ======================================================================
# Action causality (Section 9 -- the single most important test)
# ======================================================================


def test_keep_and_stop_produce_different_controller_commands(env):
    """Same reset state, different actions -> genuinely different
    low-level commands (the minimum bar for action causality)."""

    env.reset(SINGLE_MANEUVER)
    _, _, _, _, keep_info = env.step(BehaviorAction.KEEP)

    env.reset(SINGLE_MANEUVER)
    _, _, _, _, stop_info = env.step(BehaviorAction.STOP)

    assert keep_info["controller_acceleration_mps2"] != pytest.approx(
        stop_info["controller_acceleration_mps2"]
    )
    assert keep_info["reference_speed_mps"] != stop_info["reference_speed_mps"]


def test_keep_vs_stop_ego_state_diverges_over_steps(env):
    """reset same maneuver twice, apply KEEP vs STOP for several
    steps, ego state A != ego state B (Stage B-1 Section 9's explicit
    required test)."""

    env.reset(SINGLE_MANEUVER)
    for _ in range(5):
        obs_keep, _, terminated, truncated, _ = env.step(BehaviorAction.KEEP)
        if terminated or truncated:
            break

    env.reset(SINGLE_MANEUVER)
    for _ in range(5):
        obs_stop, _, terminated, truncated, _ = env.step(BehaviorAction.STOP)
        if terminated or truncated:
            break

    # v_e (index 0) must differ: STOP decelerates toward 0, KEEP
    # toward the nominal cruise speed -- these are different targets.
    assert obs_keep[0] != pytest.approx(obs_stop[0], abs=1e-6)


def test_merge_produces_different_lateral_trajectory_than_keep(env):
    """MERGE vs KEEP from the same reset state must produce physically
    DIFFERENT ego lateral behavior -- not just a different internal
    flag. Uses CAUSALITY_MANEUVER (clearer lateral separation) to keep
    the test robust to shallow real-world merge geometry."""

    def rollout_final_position(action, n_steps=25):
        env.reset(CAUSALITY_MANEUVER)
        for _ in range(n_steps):
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        return env._current_ego_pose_and_speed()[:2]

    keep_final = rollout_final_position(BehaviorAction.KEEP)
    merge_final = rollout_final_position(BehaviorAction.MERGE)

    lateral_divergence = np.hypot(
        keep_final[0] - merge_final[0], keep_final[1] - merge_final[1]
    )
    assert lateral_divergence > 0.5  # meaningfully different, not noise-level


def test_merge_commitment_locks_in_across_steps(env):
    """Once MERGE is selected, subsequent step() calls with a
    DIFFERENT requested action must still execute MERGE (Stage B-0/
    B-1 commitment semantics) -- verified through the real
    environment, not just the standalone DecisionState unit test."""

    env.reset(SINGLE_MANEUVER)
    _, _, terminated, truncated, info = env.step(BehaviorAction.MERGE)
    assert info["executed_action"] == BehaviorAction.MERGE
    assert info["merge_committed"] is True

    if not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(BehaviorAction.STOP)
        assert info["requested_action"] == BehaviorAction.STOP
        assert info["executed_action"] == BehaviorAction.MERGE  # ignored request
        assert info["merge_committed"] is True


# ======================================================================
# Logged surrounding-agent verification
# ======================================================================


def test_non_ego_agents_follow_logged_trajectory(env):
    """Controlled ego may diverge from the log; non-ego agents must
    match the WOMD logged trajectory exactly (within floating-point
    tolerance) at every simulated step, since no sim_agent_actors are
    configured (Stage A/B-0: default empty tuple -> log playback for
    all non-SDC objects)."""

    observation, info = env.reset(SINGLE_MANEUVER)

    for _ in range(5):
        env.step(BehaviorAction.STOP)  # a strong action, to maximize any
        # accidental non-ego divergence if the wiring were wrong.

        sim_traj = env._state.current_sim_trajectory
        log_traj = env._state.log_trajectory
        timestep = int(env._state.timestep)

        non_ego_mask = ~np.asarray(env._state.object_metadata.is_sdc)
        sim_x = np.asarray(sim_traj.x)[:, 0]
        log_x = np.asarray(log_traj.x)[:, timestep]
        sim_y = np.asarray(sim_traj.y)[:, 0]
        log_y = np.asarray(log_traj.y)[:, timestep]

        np.testing.assert_allclose(
            sim_x[non_ego_mask], log_x[non_ego_mask], atol=1e-3
        )
        np.testing.assert_allclose(
            sim_y[non_ego_mask], log_y[non_ego_mask], atol=1e-3
        )


# ======================================================================
# Single-maneuver success
# ======================================================================


def test_single_maneuver_merge_reaches_success(env):
    """A real single-candidate maneuver, driven by repeated MERGE,
    must reach SUCCESS termination within the episode horizon (not
    truncation, not failure) -- proving the whole
    action->controller->dynamics->online-causal-success pipeline
    works end to end."""

    env.reset(CAUSALITY_MANEUVER)
    reasons = []
    for _ in range(60):
        _, _, terminated, truncated, info = env.step(BehaviorAction.MERGE)
        reasons.append(info["termination_reason"])
        if terminated or truncated:
            break

    assert reasons[-1] == "success"


# ======================================================================
# Chained maneuver active-transition / final-success
# ======================================================================


def test_chained_maneuver_advances_active_transition():
    """A real chained maneuver must advance active_transition_index
    exactly once (from 0 to 1) upon stably entering the INTERMEDIATE
    target lane. At the EXACT step of that advancement, it must not
    also report success (reaching the intermediate lane is not itself
    success) -- success is legitimately allowed on a LATER step, once
    idx==1 is the final transition and ego has stably progressed onto
    it (see test_chained_maneuver_final_success_only_on_final_target
    for that case)."""

    env = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)
    env.reset(CHAINED_MANEUVER)

    previous_index = 0
    saw_intermediate_advance = False
    for _ in range(80):
        _, _, terminated, truncated, info = env.step(BehaviorAction.MERGE)
        if info["active_transition_index"] != previous_index:
            saw_intermediate_advance = True
            # The step where the index CHANGES must not simultaneously
            # report success -- advancing past the intermediate target
            # is not itself the final success.
            assert info["termination_reason"] != "success"
            previous_index = info["active_transition_index"]
        if terminated or truncated:
            break

    assert saw_intermediate_advance, (
        "chained maneuver never advanced past the intermediate transition "
        "within the step budget"
    )


def test_chained_maneuver_final_success_only_on_final_target():
    """A real chained maneuver, driven to completion, must report
    success only once active_transition_index has reached the FINAL
    transition (not the intermediate one)."""

    env = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)
    env.reset(CHAINED_MANEUVER)

    final_transition_index = len(CHAINED_MANEUVER.lane_chain) - 2
    reasons = []
    transition_indices_at_success = []
    for _ in range(100):
        _, _, terminated, truncated, info = env.step(BehaviorAction.MERGE)
        reasons.append(info["termination_reason"])
        if info["termination_reason"] == "success":
            transition_indices_at_success.append(info["active_transition_index"])
        if terminated or truncated:
            break

    assert "success" in reasons
    assert all(
        idx == final_transition_index for idx in transition_indices_at_success
    )


# ======================================================================
# Horizon truncation
# ======================================================================


def test_horizon_truncation_fires_when_nothing_else_terminates(monkeypatch):
    """Integration-level check that TRUNCATION_HORIZON actually fires
    through the real environment's step() wiring once steps_elapsed
    reaches the configured horizon, if no other termination condition
    (success/collision/offroad) fires first.

    Uses a monkeypatched, deliberately short base horizon
    (``MAX_EPISODE_HORIZON_FRAMES=3``) rather than the real 100: real
    ACCEPT candidates' merge_start_frame is, by construction, already
    close to their eventual transition_frame (Phase 1 Stage B: median
    gap 16 frames, max 52) -- direct rollout testing found EVERY real
    maneuver tried reaches success, collision, or offroad well before
    100 steps under simple KEEP/STOP actions, since even a decelerating
    or holding vehicle's forward momentum is often enough to complete a
    already-close, already-converging real merge geometry. Reaching
    the horizon specifically requires a scenario where nothing else
    happens for the full horizon, which real ACCEPT candidates are not
    selected to have -- so this test verifies the wiring/mechanism
    directly (already covered in isolation by
    test_termination.py::test_termination_truncation_at_horizon)
    rather than depending on finding a real scene that survives the
    full horizon completely uneventfully.

    Stage B-2.8: the environment's EFFECTIVE horizon is no longer the
    bare ``MAX_EPISODE_HORIZON_FRAMES`` constant -- ``_resolve_episode_
    horizon`` extends it by ``merge_start_frame - decision_start_frame``
    so starting the episode earlier (at the causal decision_start_frame)
    doesn't shrink the old absolute post-merge-start opportunity (see
    ``_resolve_episode_horizon`` docstring). This test reads the actual
    effective horizon back from ``info["episode_horizon"]`` rather than
    assuming it equals the bare monkeypatched constant.
    """

    import src.environment.merge_environment as merge_environment_module

    monkeypatch.setattr(
        merge_environment_module, "MAX_EPISODE_HORIZON_FRAMES", 3
    )

    env = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)
    _, reset_info = env.reset(SINGLE_MANEUVER)
    effective_horizon = reset_info["episode_horizon"]

    steps_taken = 0
    truncated = False
    info = None
    for _ in range(effective_horizon + 5):
        _, _, terminated, truncated, info = env.step(BehaviorAction.KEEP)
        steps_taken += 1
        if terminated or truncated:
            break

    assert steps_taken <= effective_horizon
    if info["termination_reason"] not in ("success", "failure_collision", "failure_offroad"):
        assert truncated is True
        assert info["termination_reason"] == "truncation_horizon"


# ======================================================================
# Determinism / reproducibility
# ======================================================================


def test_reset_is_deterministic():

    env_a = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)
    obs_a, _ = env_a.reset(SINGLE_MANEUVER)

    env_b = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)
    obs_b, _ = env_b.reset(SINGLE_MANEUVER)

    np.testing.assert_array_equal(obs_a, obs_b)


def test_same_action_sequence_produces_same_rollout():

    action_sequence = [
        BehaviorAction.KEEP,
        BehaviorAction.FOLLOW,
        BehaviorAction.KEEP,
        BehaviorAction.STOP,
        BehaviorAction.MERGE,
    ]

    def run(env_instance):
        env_instance.reset(SINGLE_MANEUVER)
        observations = []
        for action in action_sequence:
            obs, _, terminated, truncated, _ = env_instance.step(action)
            observations.append(obs)
            if terminated or truncated:
                break
        return observations

    env_a = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)
    env_b = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)

    observations_a = run(env_a)
    observations_b = run(env_b)

    assert len(observations_a) == len(observations_b)
    for obs_a, obs_b in zip(observations_a, observations_b):
        np.testing.assert_array_equal(obs_a, obs_b)


def test_different_maneuver_resets_context_correctly():

    env = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)

    env.reset(SINGLE_MANEUVER)
    _, first_info = env.reset(CHAINED_MANEUVER)

    assert first_info["active_source_lane_id"] == 618
    assert first_info["active_target_lane_id"] == 623
    assert first_info["final_target_lane_id"] == 608
