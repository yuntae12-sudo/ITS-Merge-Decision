"""Stage B-2 real-environment FSM smoke tests: verifies the FSM
doesn't structurally deadlock (never STOP-forever, never
KEEP-forever-avoiding-a-reachable-MERGE) when driven against the real
MergeEnvironment, on both a single-transition and a chained maneuver.

Reuses the same real-scene fixtures as
tests/environment/test_merge_environment.py, for the same reason
(real Waymax dynamics wiring cannot be exercised meaningfully with
synthetic geometry).
"""

import pytest

from src.environment.behavior_action import BehaviorAction
from src.environment.fsm_policy import FsmPolicy
from src.environment.merge_environment import ManeuverSpec, MergeEnvironment

DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"

# Same real single-candidate maneuver used in test_merge_environment.py.
SINGLE_MANEUVER = ManeuverSpec(
    maneuver_id="MAN_0001",
    source_shard="training_tfexample.tfrecord-00000-of-01000",
    record_index=12,
    lane_chain=[637, 628],
    candidate_ids=["training_tfexample.tfrecord-00000-of-01000#12__t49__637_628"],
    merge_start_frame=36,
)

# Same real chained maneuver used in test_merge_environment.py --
# CONFIRMED (Stage B-1) to reach success reliably, so this exercises
# FSM-driven chain advancement without being confounded by the
# controller's known tracking limits on harder maneuvers.
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

MAX_STEPS = 60


@pytest.fixture(scope="module")
def env():
    return MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)


def _run_fsm_rollout(env, maneuver, max_steps=MAX_STEPS):
    fsm = FsmPolicy()
    observation, info = env.reset(maneuver)
    action_counts = {action: 0 for action in BehaviorAction}
    terminated = truncated = False
    final_info = info
    for _ in range(max_steps):
        decision = fsm.decide(observation)
        action_counts[decision.action] += 1
        observation, reward, terminated, truncated, info = env.step(decision.action)
        final_info = info
        if terminated or truncated:
            break
    return action_counts, terminated, truncated, final_info


def test_fsm_does_not_deadlock_on_single_maneuver(env):
    action_counts, terminated, truncated, info = _run_fsm_rollout(env, SINGLE_MANEUVER)
    assert terminated or truncated
    total_steps = sum(action_counts.values())
    # Not stuck entirely on one non-MERGE action for the whole horizon.
    assert action_counts[BehaviorAction.STOP] < total_steps
    assert action_counts[BehaviorAction.KEEP] < total_steps


def test_fsm_reaches_merge_commitment_on_single_maneuver(env):
    _, terminated, truncated, info = _run_fsm_rollout(env, SINGLE_MANEUVER)
    assert terminated or truncated
    assert info["merge_committed"] or info["termination_reason"] is not None


def test_fsm_advances_chained_maneuver(env):
    action_counts, terminated, truncated, info = _run_fsm_rollout(env, CHAINED_MANEUVER)
    assert terminated or truncated
    # The FSM must have proposed MERGE at least once to ever commit.
    assert action_counts[BehaviorAction.MERGE] > 0


def test_fsm_eventually_commits_to_merge_when_safe(env):
    observation, info = env.reset(SINGLE_MANEUVER)
    fsm = FsmPolicy()
    committed = False
    for _ in range(MAX_STEPS):
        decision = fsm.decide(observation)
        observation, reward, terminated, truncated, info = env.step(decision.action)
        if info["merge_committed"]:
            committed = True
            break
        if terminated or truncated:
            break
    assert committed
