"""Tests for src.training.run_manifest -- the additive `.run.json`
sidecar (docs/ppo/PPO_VISUALIZATION_GUIDE.md, task Section 11). Must
never affect CheckpointPayload/save_checkpoint/load_checkpoint's
contract -- these tests only exercise the sidecar itself."""

import json
import os

from src.training.run_manifest import run_manifest_path_for_checkpoint, write_run_manifest


def test_run_manifest_path_replaces_pkl_suffix():
    assert (
        run_manifest_path_for_checkpoint("outputs/ppo_checkpoints/reward_v1_seed0.pkl")
        == "outputs/ppo_checkpoints/reward_v1_seed0.run.json"
    )


def test_write_run_manifest_writes_expected_fields(tmp_path):
    checkpoint_path = str(tmp_path / "some_run.pkl")

    output_path = write_run_manifest(
        checkpoint_path=checkpoint_path,
        ppo_config_path="configs/ppo/ppo_p6_baseline.yaml",
        reward_config_path="configs/reward/merge_reward_v0.yaml",
        dataset_config_path="outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml",
        maneuver_ids=["MAN_0001", "MAN_0002"],
        seed=0,
        num_updates=50,
        max_episode_steps=100,
        reward_version="v0",
        git_sha="abc123",
        global_env_step=1000,
        ppo_update_step=50,
        resumed_from=None,
    )

    assert output_path == str(tmp_path / "some_run.run.json")
    assert os.path.exists(output_path)

    loaded = json.loads(open(output_path).read())
    for key in [
        "checkpoint_path", "ppo_config_path", "reward_config_path",
        "dataset_config_path", "maneuver_ids", "seed", "num_updates",
        "max_episode_steps", "reward_version", "git_sha",
        "global_env_step", "ppo_update_step", "resumed_from",
    ]:
        assert key in loaded
    assert loaded["maneuver_ids"] == ["MAN_0001", "MAN_0002"]
    assert loaded["seed"] == 0
