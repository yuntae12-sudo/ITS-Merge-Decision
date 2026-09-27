"""Checkpoint -> restorable PPO policy/value pair, for VISUALIZATION/
DIAGNOSTIC use only (docs/ppo/PPO_VISUALIZATION_GUIDE.md).

This module never trains anything and never mutates a checkpoint. It
only reconstructs the exact ``(PolicyNetwork, params)`` /
``(ValueNetwork, params)`` pairs a checkpoint's own
``config_snapshot["network_hidden_sizes"]`` describes, using the same
``build_policy_network``/``build_value_network`` constructors P3's
training code uses (``src/policies/ppo/networks.py``) -- never a
hardcoded default network shape.
"""

import dataclasses
from typing import Any, List, Optional

from src.policies.ppo.networks import (
    NUM_ACTIONS,
    OBSERVATION_DIM,
    build_policy_network,
    build_value_network,
)
from src.policies.ppo.policy import PPOPolicy
from src.training.checkpoint import (
    CheckpointPayload,
    load_checkpoint,
    require_checkpoint_dataset_schema,
)


@dataclasses.dataclass(frozen=True)
class RestoredPPOCheckpoint:
    """Everything a visualization run needs from one checkpoint file:
    the reconstructed policy/value networks + their trained params,
    plus the checkpoint's own metadata (verbatim, never re-derived)."""

    checkpoint_path: str
    payload: CheckpointPayload
    policy: PPOPolicy
    value_network: Any
    value_params: Any
    network_hidden_sizes: List[int]
    maneuver_ids: List[str]
    seed: int
    reward_version: str
    checkpoint_git_sha: str
    global_env_step: int
    ppo_update_step: int
    reward_config_path: Optional[str]
    ppo_config_path: Optional[str]
    dataset_schema_version: str = "merge_geometry_v1_legacy"


def restore_ppo_checkpoint(
    checkpoint_path: str, expected_dataset_schema_version: Optional[str] = None
) -> RestoredPPOCheckpoint:
    """Loads ``checkpoint_path`` via the real
    ``src.training.checkpoint.load_checkpoint`` contract and rebuilds
    the exact policy/value network architecture the checkpoint's own
    ``config_snapshot`` describes (falling back to this repo's fixed
    P0-P6 baseline hidden sizes ``(256, 64, 32)`` only if an older
    checkpoint's snapshot lacks the key -- never silently assuming a
    different, newer default)."""

    payload = load_checkpoint(checkpoint_path)
    if expected_dataset_schema_version is not None:
        require_checkpoint_dataset_schema(payload, expected_dataset_schema_version)

    hidden_sizes = list(
        payload.config_snapshot.get("network_hidden_sizes", (256, 64, 32))
    )

    policy_network = build_policy_network(
        hidden_sizes=hidden_sizes, num_actions=NUM_ACTIONS
    )
    value_network = build_value_network(hidden_sizes=hidden_sizes)

    policy = PPOPolicy(policy_network, payload.policy_params)

    maneuver_ids = list(payload.config_snapshot.get("maneuver_ids", []))

    return RestoredPPOCheckpoint(
        checkpoint_path=checkpoint_path,
        payload=payload,
        policy=policy,
        value_network=value_network,
        value_params=payload.value_params,
        network_hidden_sizes=hidden_sizes,
        maneuver_ids=maneuver_ids,
        seed=payload.seed,
        reward_version=payload.reward_version,
        checkpoint_git_sha=payload.git_sha,
        global_env_step=payload.global_env_step,
        ppo_update_step=payload.ppo_update_step,
        reward_config_path=payload.config_snapshot.get("reward_config_path"),
        ppo_config_path=payload.config_snapshot.get("ppo_config_path"),
        dataset_schema_version=payload.dataset_schema_version,
    )


__all__ = ["RestoredPPOCheckpoint", "restore_ppo_checkpoint", "OBSERVATION_DIM"]
