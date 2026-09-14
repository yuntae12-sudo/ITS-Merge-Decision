"""Phase 3 Stage 3-G.4: action-discriminability sanity check across a
small representative sample of REAL canonical maneuvers, under
downstream_mode="frenet_mpc".

Confirms KEEP/FOLLOW/MERGE/STOP produce meaningfully DIFFERENT physical
motion where expected -- mirrors
tests/environment/test_merge_environment_frenet_mpc.py's own
test_keep_produces_causal_forward_motion_frenet_mpc /
test_merge_produces_lateral_motion_toward_target_frenet_mpc /
test_stop_produces_deceleration_frenet_mpc (which already cover this on
2 fixed maneuvers, SINGLE_MANEUVER and CAUSALITY_MANEUVER) but broadens
the check to a representative sample of REAL maneuvers from the
canonical 168-maneuver pool, per Stage 3-G.4's own brief. This script
is diagnostic only -- it prints per-maneuver measurements, it does not
assert/fail (the existing pytest suite already provides the pass/fail
gate for the two fixed maneuvers it covers).
"""

import numpy as np

from src.environment.behavior_action import BehaviorAction
from src.environment.full_split_evaluator import load_maneuver_specs
from src.environment.merge_environment import MergeEnvironment

DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"
N_STEPS = 20
SAMPLE_SIZE = 8


def rollout_final_pose_and_speed(env, spec, action):
    env.reset(spec)
    initial_pose = env._current_ego_pose_and_speed()
    for _ in range(N_STEPS):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    final_pose = env._current_ego_pose_and_speed()
    return initial_pose, final_pose


def main():
    env = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH, downstream_mode="frenet_mpc")

    train_specs = load_maneuver_specs("train")
    # Evenly-spaced sample across the TRAIN split for representativeness.
    stride = max(len(train_specs) // SAMPLE_SIZE, 1)
    sample = train_specs[::stride][:SAMPLE_SIZE]

    print(f"Action-discriminability sanity check on {len(sample)} real maneuvers "
          f"(frenet_mpc, {N_STEPS}-step rollouts per action):\n")

    for spec in sample:
        try:
            (kx0, ky0, _, ks0), (kx1, ky1, _, ks1) = rollout_final_pose_and_speed(
                env, spec, BehaviorAction.KEEP
            )
            (mx0, my0, _, ms0), (mx1, my1, _, ms1) = rollout_final_pose_and_speed(
                env, spec, BehaviorAction.MERGE
            )
            (sx0, sy0, _, ss0), (sx1, sy1, _, ss1) = rollout_final_pose_and_speed(
                env, spec, BehaviorAction.STOP
            )
        except Exception as exc:  # noqa: BLE001 -- diagnostic script, report and continue
            print(f"  {spec.maneuver_id}: EXCEPTION during rollout: {type(exc).__name__}: {exc}")
            continue

        keep_displacement = np.hypot(kx1 - kx0, ky1 - ky0)
        keep_vs_merge_divergence = np.hypot(kx1 - mx1, ky1 - my1)
        stop_speed_delta = ss1 - ss0

        print(
            f"  {spec.maneuver_id}: "
            f"KEEP displacement={keep_displacement:.2f}m final_speed={ks1:.2f}m/s | "
            f"KEEP-vs-MERGE final-position divergence={keep_vs_merge_divergence:.2f}m | "
            f"STOP speed delta={stop_speed_delta:+.2f}m/s (initial={ss0:.2f} final={ss1:.2f})"
        )


if __name__ == "__main__":
    main()
