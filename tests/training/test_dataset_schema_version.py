import pytest

from src.scenarios.merge_v2 import DATASET_SCHEMA_V2, LEGACY_DATASET_SCHEMA
from src.training.checkpoint import (
    CheckpointPayload,
    require_checkpoint_dataset_schema,
    require_single_dataset_schema,
)


def payload(schema=LEGACY_DATASET_SCHEMA):
    return CheckpointPayload(
        policy_params={}, value_params={}, optimizer_state={}, jax_rng_key=None,
        global_env_step=0, ppo_update_step=0, seed=0, config_snapshot={},
        reward_version="v0", git_sha="test", dataset_schema_version=schema,
    )


def test_legacy_checkpoint_cannot_be_evaluated_as_v2():
    with pytest.raises(ValueError, match="schema mismatch"):
        require_checkpoint_dataset_schema(payload(), DATASET_SCHEMA_V2)


def test_v2_checkpoint_matches_v2_dataset():
    require_checkpoint_dataset_schema(payload(DATASET_SCHEMA_V2), DATASET_SCHEMA_V2)


def test_v1_and_v2_checkpoints_cannot_share_an_aggregate():
    with pytest.raises(ValueError, match="Mixed dataset schemas"):
        require_single_dataset_schema([payload(), payload(DATASET_SCHEMA_V2)])
