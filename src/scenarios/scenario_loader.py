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
physical shard (``source_shard`` + ``record_index``). Waymax's
``DatasetConfig.deterministic`` defaults to True and this loader does
not override it, so the same physical shard always yields scenarios in
the same order. ``record_index`` is LOCAL to one physical shard file --
it is never made globally unique across shards. Uniqueness across a
multi-shard scan comes from ``scene_key`` (``f"{source_shard}#{record_index}"``)
since two different physical shards may legitimately both have a
``record_index=28``.

Multi-shard dataset expansion
------------------------------
``load_dataset_config`` accepts two YAML schemas:

1. Legacy single-shard schema (unchanged since Phase 1 Commit A): a
   bare ``path:`` field (plus ``max_num_objects``/``repeat``/
   ``shuffle_seed``), no ``split``/``shards``/``shard_glob``. Detected
   by the presence of ``path`` and the absence of ``shards``/
   ``shard_glob``. Defaults to ``split: validation`` (this has always
   been what it meant in practice -- see ``configs/dataset.yaml``).
   Returned as a ``DatasetExpansionConfig`` with exactly one resolved
   shard path.

2. New multi-shard "dataset expansion" schema:

       dataset_name: WOMD
       split: validation   # or "training" -- explicit, never inferred
       shards:              # OR shard_glob (not both required)
         - data/womd/validation/validation_tfexample.tfrecord-00000-of-00150
       # shard_glob: "data/womd/validation/validation_tfexample.tfrecord-*-of-00150"
       max_num_objects: 64
       repeat: 1
       shuffle_seed: null

   ``split: validation`` maps to Waymax's ``WOD_1_3_1_VALIDATION`` base
   config; ``split: training`` maps to ``WOD_1_3_1_TRAINING``. Any
   other value raises loudly. One config = one split; there is no
   mechanism to mix training and validation shards in a single
   invocation (only one ``split`` field exists).

Either schema resolves to a ``DatasetExpansionConfig``: a parsed,
validated, and fully resolved (sorted, existing, physical-file) list of
shard paths -- NOT yet a Waymax ``DatasetConfig``. Use
``build_waymax_config`` to build the per-shard Waymax config, and
``iter_all_shards`` to iterate every scenario across every shard in the
expansion config (reusing the existing per-shard ``iter_scenarios``
once per physical file).
"""

import dataclasses
import glob as glob_module
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import yaml
from waymax import config as waymax_config
from waymax import dataloader
from waymax.datatypes.simulator_state import SimulatorState

_SPLIT_TO_BASE_CONFIG = {
    "validation": waymax_config.WOD_1_3_1_VALIDATION,
    "training": waymax_config.WOD_1_3_1_TRAINING,
}


@dataclasses.dataclass(frozen=True)
class ScenarioRecord:
    """A single loaded WOMD scenario plus lightweight scan metadata."""

    record_index: int
    source_shard: str
    source_dataset: str
    source_split: str
    scene_key: str
    state: SimulatorState
    num_objects: int
    sdc_index: int
    sdc_id: int
    valid_trajectory_length: int
    roadgraph_point_count: int


@dataclasses.dataclass(frozen=True)
class DatasetExpansionConfig:
    """A parsed + resolved multi-shard dataset expansion config.

    ``shard_paths`` is always the fully resolved, sorted list of
    physical file paths that actually exist on disk -- never a glob
    pattern, and never an unresolved/unsorted list.
    """

    dataset_name: str
    split: str
    shard_paths: List[str]
    max_num_objects: Optional[int]
    repeat: Optional[int]
    shuffle_seed: Optional[int]


def _resolve_shard_paths(raw: dict, config_path: str) -> List[str]:
    """Resolves the ``shards``/``shard_glob`` (or legacy ``path``) field
    of a raw YAML dict into a sorted list of existing physical file
    paths. Exactly one of ``shards``/``shard_glob``/``path`` must be
    present.
    """

    has_shards = "shards" in raw
    has_glob = "shard_glob" in raw
    has_path = "path" in raw

    if sum([has_shards, has_glob, has_path]) != 1:
        raise ValueError(
            f"Dataset config {config_path!r} must specify exactly one "
            "of 'shards', 'shard_glob', or the legacy 'path' field -- "
            f"got shards={has_shards}, shard_glob={has_glob}, "
            f"path={has_path}"
        )

    if has_path:
        paths = [raw["path"]]
    elif has_shards:
        shards = raw["shards"]
        if not isinstance(shards, list) or not shards:
            raise ValueError(
                f"Dataset config {config_path!r}: 'shards' must be a "
                "non-empty list of file paths."
            )
        if len(set(shards)) != len(shards):
            duplicates = sorted(
                {s for s in shards if shards.count(s) > 1}
            )
            raise ValueError(
                f"Dataset config {config_path!r}: 'shards' contains "
                f"duplicate path(s): {duplicates}"
            )
        paths = sorted(shards)
    else:
        pattern = raw["shard_glob"]
        matched = sorted(glob_module.glob(pattern))
        if not matched:
            raise ValueError(
                f"Dataset config {config_path!r}: shard_glob pattern "
                f"{pattern!r} matched zero files. Refusing to build an "
                "empty dataset expansion."
            )
        paths = matched

    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise ValueError(
            f"Dataset config {config_path!r}: the following shard "
            f"path(s) do not exist on disk: {missing}"
        )

    return paths


def load_dataset_config(config_path: str) -> DatasetExpansionConfig:
    """Builds a ``DatasetExpansionConfig`` from a YAML file.

    Supports both the legacy single-``path:`` schema (defaults to
    ``split: validation``) and the new multi-shard expansion schema
    (``dataset_name``/``split``/``shards`` or ``shard_glob``). See the
    module docstring for the full schema description.

    Raises:
        ValueError: unsupported ``split`` value, ``repeat`` != 1
            (dataset-expansion configs must not repeat -- this would
            produce duplicate candidate_ids), a zero-match shard_glob,
            a missing physical shard file, or a malformed
            shards/shard_glob/path specification.
    """

    with open(config_path, "r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file) or {}

    is_legacy = "path" in raw and "shards" not in raw and "shard_glob" not in raw

    split = raw.get("split", "validation" if is_legacy else None)
    if split is None:
        raise ValueError(
            f"Dataset config {config_path!r} must specify 'split' "
            "('validation' or 'training')."
        )
    if split not in _SPLIT_TO_BASE_CONFIG:
        raise ValueError(
            f"Dataset config {config_path!r}: unsupported split "
            f"{split!r} -- must be one of "
            f"{sorted(_SPLIT_TO_BASE_CONFIG)}."
        )

    repeat = raw.get("repeat", 1)
    if repeat is not None and repeat != 1:
        raise ValueError(
            f"Dataset config {config_path!r}: 'repeat' must be 1 (or "
            f"null/omitted, treated as 1) for dataset-expansion configs "
            f"-- got {repeat!r}. repeat != 1 could produce duplicate "
            "candidate_ids within a single scan."
        )

    shard_paths = _resolve_shard_paths(raw, config_path)

    dataset_name = raw.get("dataset_name", "WOMD")

    return DatasetExpansionConfig(
        dataset_name=dataset_name,
        split=split,
        shard_paths=shard_paths,
        max_num_objects=raw.get("max_num_objects"),
        repeat=repeat,
        shuffle_seed=raw.get("shuffle_seed"),
    )


def resolve_physical_shard(
    expansion_config: "DatasetExpansionConfig",
    source_shard: str,
    source_split: Optional[str] = None,
) -> str:
    """Resolves a stable shard filename to its exact configured physical path.

    This is the ONLY place shard-filename matching logic should live.
    Every caller that needs to turn a stored ``source_shard`` basename
    (e.g. from a candidates/manifest CSV row) back into a real physical
    file must go through this function -- never re-implement path
    -guessing/matching (e.g. a hardcoded ``data/womd/<split>/<file>``
    convention) elsewhere.

    Args:
        expansion_config: DatasetExpansionConfig -- the authority for
            what physical files are available
            (``expansion_config.shard_paths``).
        source_shard: the stable basename identity (e.g.
            "validation_tfexample.tfrecord-00003-of-00150") to resolve
            -- NEVER an absolute path; this is the machine-independent
            candidate identity stored in CSVs.
        source_split: if given, must match ``expansion_config.split``,
            else raise immediately (before any path resolution) -- this
            protects against materializing/rendering a "training"
            candidate row against a "validation" expansion config or
            vice versa.

    Returns:
        The exact resolved physical path (str) from
        ``expansion_config.shard_paths`` whose basename equals
        ``source_shard``.

    Raises:
        ValueError: if ``source_split`` is given and disagrees with
            ``expansion_config.split`` (checked FIRST, before touching
            paths).
        ValueError: if zero paths in ``expansion_config.shard_paths``
            have that basename (unknown shard -- no fallback to any
            local-layout convention like ``data/womd/<split>/<filename>``
            is ever attempted).
        ValueError: if more than one path shares that basename
            (ambiguous duplicate basename -- does NOT silently pick the
            first).
    """

    if source_split is not None and source_split != expansion_config.split:
        raise ValueError(
            f"source_split mismatch: candidate row has "
            f"source_split={source_split!r} but the supplied dataset "
            f"expansion config has split={expansion_config.split!r}. "
            "Refusing to resolve a shard path across mismatched splits."
        )

    matches = [
        path
        for path in expansion_config.shard_paths
        if Path(path).name == source_shard
    ]

    if not matches:
        available = sorted(
            Path(path).name for path in expansion_config.shard_paths
        )
        raise ValueError(
            f"Unknown shard: source_shard={source_shard!r} does not "
            "match the basename of any physical path in this dataset "
            f"expansion config's shard_paths. Available shard(s): "
            f"{available}. No fallback to any local-layout convention "
            "(e.g. data/womd/<split>/<filename>) is attempted."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous shard: source_shard={source_shard!r} matches "
            f"{len(matches)} distinct physical paths in "
            f"expansion_config.shard_paths: {sorted(matches)}. Refusing "
            "to silently pick one."
        )

    return matches[0]


def select_single_shard_for_inspection(
    expansion_config: DatasetExpansionConfig,
    source_shard: Optional[str] = None,
    record_index: Optional[int] = None,
) -> str:
    """Shared CLI shard-selection policy for single-scene inspection tools
    (``inspect_lane_geometry.py``, ``inspect_ego_lane_sequence.py``,
    ``inspect_merge_candidate.py``).

    Policy (identical across all three scripts -- implemented once here
    so it is not triplicated in each CLI):

    - Config resolves to exactly 1 physical shard AND ``source_shard``
      omitted -> use that one shard automatically (backward compatible
      with the legacy single-shard config/CLI usage).
    - Config resolves to multiple shards AND ``source_shard`` given ->
      resolve via ``resolve_physical_shard``, use it.
    - Config resolves to multiple shards AND ``source_shard`` omitted
      -> FAIL LOUDLY listing all available shard basenames.
    - ``source_shard`` given but doesn't match any configured shard ->
      error via ``resolve_physical_shard``'s unknown-shard ValueError.
    - Multiple resolved paths share the same basename -> ambiguity
      error via ``resolve_physical_shard``.

    Returns:
        The exact resolved physical shard path (str).

    Raises:
        ValueError: ambiguous record_index selection (multiple shards,
            no --source-shard given), or any error raised by
            ``resolve_physical_shard``.
    """

    if source_shard is None:
        if len(expansion_config.shard_paths) == 1:
            return expansion_config.shard_paths[0]

        available = sorted(
            Path(path).name for path in expansion_config.shard_paths
        )
        idx_str = "<idx>" if record_index is None else str(record_index)
        raise ValueError(
            f"Dataset config contains {len(expansion_config.shard_paths)} "
            f"physical shards. --record-index={idx_str} is ambiguous "
            "because record_index is shard-local. Pass --source-shard "
            "with one of:\n  " + "\n  ".join(available)
        )

    return resolve_physical_shard(expansion_config, source_shard=source_shard)


def build_waymax_config(
    expansion_config: DatasetExpansionConfig, physical_shard_path: str
) -> waymax_config.DatasetConfig:
    """Builds a Waymax ``DatasetConfig`` for one physical shard file.

    Picks the base config (``WOD_1_3_1_VALIDATION`` or
    ``WOD_1_3_1_TRAINING``) from ``expansion_config.split``, then
    overrides ``path``/``max_num_objects``/``repeat``/``shuffle_seed``.
    """

    base_config = _SPLIT_TO_BASE_CONFIG[expansion_config.split]

    overrides = {"path": str(physical_shard_path)}
    if expansion_config.max_num_objects is not None:
        overrides["max_num_objects"] = expansion_config.max_num_objects
    if expansion_config.repeat is not None:
        overrides["repeat"] = expansion_config.repeat
    overrides["shuffle_seed"] = expansion_config.shuffle_seed

    return dataclasses.replace(base_config, **overrides)


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
    start_index: int = 0,
    source_dataset: str = "WOMD",
    source_split: str = "validation",
) -> Iterator[ScenarioRecord]:
    """Iterates WOMD scenarios one at a time without loading the full dataset.

    Args:
        dataset_config: Waymax dataset config (see ``build_waymax_config``).
        limit: if set, stop after yielding this many scenarios (counted
            from `start_index`, not from the start of the dataset).
        start_index: number of leading scenarios to skip without
            yielding. Since Waymax's generator has no native seek, this
            still constructs every skipped scenario internally (not
            maximally efficient) but is otherwise the correct place
            for this: it keeps `record_index`/`scene_key` correct and
            avoids re-implementing skip logic in every caller.
        source_dataset: recorded on each yielded ``ScenarioRecord``
            (default "WOMD" -- the only dataset this study uses).
        source_split: recorded on each yielded ``ScenarioRecord``
            (default "validation" -- matches this function's own
            historical single-shard default so existing direct callers
            that don't pass a dataset expansion config are unaffected).

    Yields:
        ScenarioRecord: one per scenario, in deterministic order.
            ``record_index`` is always LOCAL to this one physical
            shard (0-based, per this call) -- it is not made unique
            across multiple calls/shards. See ``iter_all_shards`` for
            multi-shard iteration.
    """

    scenario_generator = dataloader.simulator_state_generator(
        config=dataset_config
    )

    stop_index = None if limit is None else start_index + limit

    for record_index, state in enumerate(scenario_generator):

        if record_index < start_index:
            continue

        if stop_index is not None and record_index >= stop_index:
            break

        sdc_index = _find_sdc_index(state)
        sdc_id = int(np.asarray(state.object_metadata.ids)[sdc_index])

        source_shard = Path(dataset_config.path).name
        scene_key = f"{source_shard}#{record_index}"

        yield ScenarioRecord(
            record_index=record_index,
            source_shard=source_shard,
            source_dataset=source_dataset,
            source_split=source_split,
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


def iter_all_shards(
    expansion_config: DatasetExpansionConfig,
    limit: Optional[int] = None,
    start_index: int = 0,
) -> Iterator[ScenarioRecord]:
    """Iterates every scenario across every physical shard in a
    ``DatasetExpansionConfig``, in sorted shard-path order.

    For each physical shard path (sorted), builds that shard's own
    Waymax config via ``build_waymax_config`` and yields from the
    existing per-shard ``iter_scenarios`` -- this is NOT a parallel
    reimplementation of ``iter_scenarios``; it calls it once per shard.

    ``limit``/``start_index`` apply to the TOTAL scan across all
    shards in this expansion config (not per-shard): e.g. ``limit=10``
    stops after 10 scenes total, scanning shard by shard in sorted
    order and moving to the next shard once the current one is
    exhausted. ``record_index`` on each yielded ``ScenarioRecord``
    stays LOCAL to its own physical shard (0-based within that shard);
    only ``scene_key`` (``source_shard#record_index``) is unique across
    the whole multi-shard scan.

    Yields:
        ScenarioRecord: one per scenario, shard by shard, in sorted
            shard-path order.
    """

    stop_index = None if limit is None else start_index + limit
    scenes_yielded_so_far = 0

    for shard_path in expansion_config.shard_paths:

        if stop_index is not None and scenes_yielded_so_far >= stop_index:
            break

        waymax_cfg = build_waymax_config(expansion_config, shard_path)

        for record in iter_scenarios(
            waymax_cfg,
            source_dataset=expansion_config.dataset_name,
            source_split=expansion_config.split,
        ):
            if scenes_yielded_so_far < start_index:
                scenes_yielded_so_far += 1
                continue

            if stop_index is not None and scenes_yielded_so_far >= stop_index:
                break

            yield record
            scenes_yielded_so_far += 1
