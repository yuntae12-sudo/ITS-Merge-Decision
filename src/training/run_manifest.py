"""Additive, non-checkpoint run-metadata sidecar
(docs/ppo/PPO_VISUALIZATION_GUIDE.md, task Section 11).

Writes a small ``<checkpoint_stem>.run.json`` file next to a saved
checkpoint, purely for human/tooling convenience (e.g. so
``scripts/visualize_ppo.py`` or a future script can
discover a run's config paths without unpickling the checkpoint).

This is deliberately ADDITIVE: it changes nothing about
``CheckpointPayload``, ``save_checkpoint``/``load_checkpoint``'s
contract, or ``train_ppo.py``'s resume behavior. A checkpoint remains
fully resume-capable with or without its ``.run.json`` sidecar ever
having been written -- this file is metadata for humans/visualization
tooling, never something training itself reads back.
"""

import json
from pathlib import Path
from typing import List, Optional

from src.scenarios.merge_v2 import LEGACY_DATASET_SCHEMA


def run_manifest_path_for_checkpoint(checkpoint_path: str) -> str:
    """``outputs/checkpoints/reward_v1_seed0.pkl`` ->
    ``outputs/checkpoints/reward_v1_seed0.run.json`` (matches the
    task spec's exact naming: replace the checkpoint's own suffix with
    ``.run.json``, not merely append)."""

    path = Path(checkpoint_path)
    return str(path.with_suffix("")) + ".run.json"


def write_run_manifest(
    checkpoint_path: str,
    ppo_config_path: str,
    reward_config_path: str,
    dataset_config_path: str,
    maneuver_ids: List[str],
    seed: int,
    num_updates: int,
    max_episode_steps: int,
    reward_version: str,
    git_sha: str,
    global_env_step: int,
    ppo_update_step: int,
    resumed_from: Optional[str] = None,
    dataset_schema_version: str = LEGACY_DATASET_SCHEMA,
    dataset_contract: Optional[str] = None,
    dataset_version: Optional[str] = None,
    dataset_freeze_sha: Optional[str] = None,
) -> str:
    """Writes the ``.run.json`` sidecar and returns its path. Best-
    effort: a caller (train_ppo.py) should treat a failure here as
    non-fatal to training completion (the checkpoint itself is already
    saved by the time this is called).

    ``dataset_contract``/``dataset_version``/``dataset_freeze_sha`` are
    additive provenance (default None so older callers/manifests are
    unaffected) recording which frozen dataset a
    ``merge_decision_v2``-contract run was trained against."""

    manifest = {
        "checkpoint_path": checkpoint_path,
        "ppo_config_path": ppo_config_path,
        "reward_config_path": reward_config_path,
        "dataset_config_path": dataset_config_path,
        "maneuver_ids": maneuver_ids,
        "seed": seed,
        "num_updates": num_updates,
        "max_episode_steps": max_episode_steps,
        "reward_version": reward_version,
        "git_sha": git_sha,
        "global_env_step": global_env_step,
        "ppo_update_step": ppo_update_step,
        "resumed_from": resumed_from,
        "dataset_schema_version": dataset_schema_version,
        "dataset_contract": dataset_contract,
        "dataset_version": dataset_version,
        "dataset_freeze_sha": dataset_freeze_sha,
    }

    output_path = run_manifest_path_for_checkpoint(checkpoint_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return output_path


__all__ = ["run_manifest_path_for_checkpoint", "write_run_manifest"]
