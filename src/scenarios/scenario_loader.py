"""Reusable WOMD scenario loading for Phase 1 dataset tooling.

This module wraps Waymax's ``dataloader.simulator_state_generator`` so
Phase 1 scripts can iterate over many scenarios from a config file
instead of the single hardcoded scene used by the Phase 0 validation
scripts (``scripts/run_scene.py``, ``scripts/run_rollout.py``).

Scene identity
--------------
The WOMD tf_example feature set parsed by Waymax does not expose a
parsed scenario id string (only per-object / per-roadgraph-point
fields). To keep scene references reproducible without inventing
identifiers the data does not provide, each scenario is identified by
its 0-based position in the deterministic iteration order of a given
dataset config (``source_shard`` + ``record_index``). Waymax's
``DatasetConfig.deterministic`` defaults to True and this loader does
not override it, so the same config always yields scenarios in the
same order.
"""

import dataclasses
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import yaml
from waymax import config as waymax_config
from waymax import dataloader
from waymax.datatypes.simulator_state import SimulatorState


@dataclasses.dataclass(frozen=True)
class ScenarioRecord:
    """A single loaded WOMD scenario plus lightweight scan metadata."""

    record_index: int
    source_shard: str
    scene_key: str
    state: SimulatorState
    num_objects: int
    sdc_index: int
    sdc_id: int
    valid_trajectory_length: int
    roadgraph_point_count: int


def load_dataset_config(config_path: str) -> waymax_config.DatasetConfig:
    """Builds a Waymax ``DatasetConfig`` from a YAML file.

    The YAML only needs to specify the fields that differ from
    Waymax's ``WOD_1_3_1_VALIDATION`` base config (path, max_num_objects,
    repeat, shuffle_seed), matching the Phase 0 loading pattern.
    """

    with open(config_path, "r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file) or {}

    return dataclasses.replace(
        waymax_config.WOD_1_3_1_VALIDATION,
        **raw,
    )


def _find_sdc_index(state: SimulatorState) -> int:
    is_sdc = np.asarray(state.object_metadata.is_sdc).astype(bool)
    sdc_indices = np.flatnonzero(is_sdc)

    if len(sdc_indices) != 1:
        raise RuntimeError(
            f"Expected exactly one SDC, but found {len(sdc_indices)}"
        )

    return int(sdc_indices[0])


def _valid_trajectory_length(state: SimulatorState, sdc_index: int) -> int:
    valid = np.asarray(state.log_trajectory.valid[sdc_index])
    return int(np.sum(valid))


def _roadgraph_point_count(state: SimulatorState) -> int:
    if state.roadgraph_points is None:
        return 0

    valid = np.asarray(state.roadgraph_points.valid)
    return int(np.sum(valid))


def iter_scenarios(
    dataset_config: waymax_config.DatasetConfig,
    limit: Optional[int] = None,
) -> Iterator[ScenarioRecord]:
    """Iterates WOMD scenarios one at a time without loading the full dataset.

    Args:
        dataset_config: Waymax dataset config (see ``load_dataset_config``).
        limit: if set, stop after yielding this many scenarios.

    Yields:
        ScenarioRecord: one per scenario, in deterministic order.
    """

    scenario_generator = dataloader.simulator_state_generator(
        config=dataset_config
    )

    for record_index, state in enumerate(scenario_generator):

        if limit is not None and record_index >= limit:
            break

        sdc_index = _find_sdc_index(state)
        sdc_id = int(np.asarray(state.object_metadata.ids)[sdc_index])

        source_shard = Path(dataset_config.path).name
        scene_key = f"{source_shard}#{record_index}"

        yield ScenarioRecord(
            record_index=record_index,
            source_shard=source_shard,
            scene_key=scene_key,
            state=state,
            num_objects=int(state.num_objects),
            sdc_index=sdc_index,
            sdc_id=sdc_id,
            valid_trajectory_length=_valid_trajectory_length(
                state, sdc_index
            ),
            roadgraph_point_count=_roadgraph_point_count(state),
        )
