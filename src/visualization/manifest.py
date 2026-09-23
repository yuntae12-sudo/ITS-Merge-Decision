"""manifest.json writer for one PPO visualization run (docs/ppo/
PPO_VISUALIZATION_GUIDE.md). Pure serialization of already-computed
values -- no new metric/derivation happens here.
"""

import json
from typing import Dict, List, Optional

from src.training.checkpoint import get_git_sha
from src.visualization.ppo_checkpoint_policy import RestoredPPOCheckpoint


def build_manifest(
    restored: RestoredPPOCheckpoint,
    run_id: str,
    evaluated_maneuver_ids: List[str],
    policy_mode: str,
    policy_seed: Optional[int],
    downstream_mode: str,
    max_episode_steps: int,
    selected_episodes: Dict[str, List[str]],
    scope: str,
    render_policy: Dict,
) -> dict:
    return {
        "checkpoint_path": restored.checkpoint_path,
        "run_id": run_id,
        "checkpoint_git_sha": restored.checkpoint_git_sha,
        "current_git_sha": get_git_sha(),
        "reward_version": restored.reward_version,
        "dataset_schema_version": restored.dataset_schema_version,
        "seed": restored.seed,
        "ppo_update_step": restored.ppo_update_step,
        "global_env_step": restored.global_env_step,
        "network_hidden_sizes": restored.network_hidden_sizes,
        "evaluated_maneuver_ids": evaluated_maneuver_ids,
        "scope": scope,
        "policy_mode": policy_mode,
        "policy_seed": policy_seed,
        "downstream_mode": downstream_mode,
        "max_episode_steps": max_episode_steps,
        "render_policy": render_policy,
        "selected_episodes": selected_episodes,
        "evaluation_kind": "POST-TRAINING FINAL-CHECKPOINT EVALUATION",
        "notes": (
            "This is a post-training evaluation of the final checkpoint, "
            "not a replay of any specific training-time rollout. "
            "outcome_index.csv always contains every evaluated episode; "
            "selected_episodes/rendered assets are a qualitative subset "
            "per render_policy, never an aggregate/generalization claim."
        ),
    }


def write_manifest(manifest: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


__all__ = ["build_manifest", "write_manifest"]
