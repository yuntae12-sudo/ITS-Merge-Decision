"""Stage B-1.5 Section A10: FOLLOW vs KEEP causality on a real
maneuver with a genuine source-lane lead present at reset.

MAN_0002 (training_tfexample.tfrecord-00000-of-01000#18,
lane_chain=374->362, merge_start_frame=0) was found (via a live
env.reset() scan, not the offline candidate manifest -- which
measures front_vehicle_id at a different reference frame) to have
source_front_present==1.0 with gap=4.52m at its own reset frame,
making it a genuine real-environment case for this check.
"""

from src.environment.behavior_action import (
    NOMINAL_CRUISE_SPEED_MPS,
    BehaviorAction,
)
from src.environment.merge_environment import ManeuverSpec, MergeEnvironment

DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"

SOURCE_LEAD_MANEUVER = ManeuverSpec(
    maneuver_id="MAN_0002",
    source_shard="training_tfexample.tfrecord-00000-of-01000",
    record_index=18,
    lane_chain=[374, 362],
    candidate_ids=["training_tfexample.tfrecord-00000-of-01000#18__t14__374_362"],
    merge_start_frame=0,
)


def _run(env, action, n=3):
    observation, info = env.reset(SOURCE_LEAD_MANEUVER)
    ref_speeds = []
    for _ in range(n):
        observation, reward, terminated, truncated, info = env.step(action)
        ref_speeds.append(info["reference_speed_mps"])
        if terminated or truncated:
            break
    return ref_speeds


def test_reset_has_a_genuine_source_front_lead():
    env = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)
    observation, info = env.reset(SOURCE_LEAD_MANEUVER)
    assert observation[10] == 1.0  # source_front_present
    assert 0.0 < observation[11] < 10.0  # a real, close-ish gap


def test_keep_holds_nominal_cruise_speed_regardless_of_lead():
    env = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)
    ref_speeds = _run(env, BehaviorAction.KEEP)
    assert all(speed == NOMINAL_CRUISE_SPEED_MPS for speed in ref_speeds)


def test_follow_diverges_from_keep_when_lead_is_present():
    env = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)
    keep_ref_speeds = _run(env, BehaviorAction.KEEP)
    follow_ref_speeds = _run(env, BehaviorAction.FOLLOW)
    assert keep_ref_speeds != follow_ref_speeds
    assert any(
        follow_speed < NOMINAL_CRUISE_SPEED_MPS for follow_speed in follow_ref_speeds
    )
