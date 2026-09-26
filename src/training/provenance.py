"""Dataset/reward run-provenance metadata (outputs/reward_v1_spec/
REWARD_V1_SPEC_FINAL.md Section 13/22).

Reads the frozen MERGE Decision Dataset v2's own freeze metadata as the
sole source of truth for ``dataset/version``/``dataset/freeze_sha``/
``filter_config_hash``/``canonical_manifest_hash`` -- never duplicates
these values as a second hardcoded copy anywhere else in this repo.
"""

import dataclasses
import json


DEFAULT_FREEZE_METADATA_PATH = "data/manifests/v2/MERGE_DECISION_DATASET_V2_FREEZE.json"


@dataclasses.dataclass(frozen=True)
class DatasetProvenance:
    version: str
    git_sha: str
    filter_config_hash: str
    canonical_manifest_hash: str


def load_dataset_provenance(
    path: str = DEFAULT_FREEZE_METADATA_PATH,
) -> DatasetProvenance:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return DatasetProvenance(
        version=str(raw["version"]),
        git_sha=str(raw["git_sha"]),
        filter_config_hash=str(raw["filter_config_sha256"]),
        canonical_manifest_hash=str(raw["canonical_manifest_sha256"]),
    )


def build_wandb_provenance_config(
    reward_version: str,
    freeze_metadata_path: str = DEFAULT_FREEZE_METADATA_PATH,
) -> dict:
    """Builds the ``OPTIONAL_PROVENANCE_KEYS`` dict
    (``src.tracking.wandb_logger``) a caller can merge into its
    ``log_config`` payload -- e.g.
    ``wandb_logger.log_config({**ppo_config_dict,
    **build_wandb_provenance_config("v1")})``."""

    provenance = load_dataset_provenance(freeze_metadata_path)
    return {
        "dataset/version": provenance.version,
        "dataset/freeze_sha": provenance.git_sha,
        "reward/version": reward_version,
        "filter_config_hash": provenance.filter_config_hash,
        "canonical_manifest_hash": provenance.canonical_manifest_hash,
    }
