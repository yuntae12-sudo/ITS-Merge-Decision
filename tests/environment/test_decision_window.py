"""Stage B-2.8: production decision_start_frame tests.

Covers Section 15's checklist:
  - decision_start_frame determinism
  - decision_start <= merge_start for every canonical maneuver
  - all 168 canonical maneuvers resolvable
  - actual reset timestep == decision_start
  - 14D finite observation at the new reset
  - horizon preserves old absolute episode opportunity (capped safely
    at the scenario's own logged length)
  - canonical split unchanged
  - Phase 1 canonical labels unchanged

Uses REAL WOMD scenes (see test_merge_environment.py's module
docstring for why) -- these tests fail, not skip, if the local
training shards are not present.
"""

import csv
import hashlib

import numpy as np
import pytest

import src.environment.merge_environment as merge_environment_module
from src.environment.behavior_action import BehaviorAction
from src.environment.decision_window import (
    DecisionStartUnresolvedError,
    compute_decision_start_frame,
)
from src.environment.dataset_split import load_split_manifest
from src.environment.episode_context import parse_candidate_ids, parse_lane_chain
from src.environment.merge_environment import ManeuverSpec, MergeEnvironment
from src.scenarios.lane_assignment import load_lane_assignment_config
from src.scenarios.lane_geometry import extract_lane_polylines
from src.scenarios.scenario_loader import (
    build_waymax_config,
    iter_scenarios,
    load_dataset_config,
    select_single_shard_for_inspection,
)

DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"
MERGE_CONFIG_PATH = "configs/phase1_merge.yaml"
MANEUVER_TABLE = "outputs/phase1/training_10shard_pilot/training_visual_merge_maneuvers.csv"
CANDIDATE_MANIFEST = "outputs/phase1/training_10shard_pilot/merge_manifest_training_scratch.csv"

# Same real single-candidate maneuver test_merge_environment.py uses.
SINGLE_MANEUVER = ManeuverSpec(
    maneuver_id="MAN_0001",
    source_shard="training_tfexample.tfrecord-00000-of-01000",
    record_index=12,
    lane_chain=[637, 628],
    candidate_ids=[
        "training_tfexample.tfrecord-00000-of-01000#12__t49__637_628"
    ],
    merge_start_frame=36,
)


@pytest.fixture
def env():
    return MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)


def _load_all_specs():
    """All 168 canonical maneuvers (both splits), as ManeuverSpecs."""

    with open(CANDIDATE_MANIFEST, newline="") as f:
        manifest_by_candidate_id = {row["candidate_id"]: row for row in csv.DictReader(f)}
    split_by_id = {r.maneuver_id: r.split for r in load_split_manifest()}

    specs = []
    with open(MANEUVER_TABLE, newline="") as f:
        for row in csv.DictReader(f):
            if row["maneuver_id"] not in split_by_id:
                continue
            spec = ManeuverSpec.from_csv_row(row, manifest_by_candidate_id)
            specs.append(spec)
    return specs


# ======================================================================
# Determinism
# ======================================================================


def test_decision_start_frame_deterministic_across_env_instances():
    """Two independent MergeEnvironment instances resetting the same
    maneuver must compute the identical decision_start_frame -- no
    hidden randomness, no cross-episode state leakage."""

    env_a = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)
    _, info_a = env_a.reset(SINGLE_MANEUVER)

    env_b = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)
    _, info_b = env_b.reset(SINGLE_MANEUVER)

    assert info_a["decision_start_frame"] == info_b["decision_start_frame"]


def test_decision_start_frame_deterministic_across_repeated_resets(env):
    """Resetting the SAME maneuver twice on one environment instance
    yields the same decision_start_frame each time."""

    _, info_1 = env.reset(SINGLE_MANEUVER)
    _, info_2 = env.reset(SINGLE_MANEUVER)

    assert info_1["decision_start_frame"] == info_2["decision_start_frame"]


# ======================================================================
# Bounds: decision_start <= merge_start, all resolvable
# ======================================================================


def test_decision_start_frame_le_merge_start_frame_for_all_canonical_maneuvers():
    """Section 5's required invariant, checked EXHAUSTIVELY over all
    168 canonical maneuvers (both splits) using the raw geometric
    helper directly (no Waymax reset needed -- these are static
    per-maneuver checks over the logged trajectory)."""

    expansion_config = load_dataset_config(DATASET_CONFIG_PATH)
    lane_assignment_config = load_lane_assignment_config(MERGE_CONFIG_PATH)
    specs = _load_all_specs()
    assert len(specs) == 168

    # Cache loaded records by (source_shard, record_index) since many
    # maneuvers share a shard/scene -- avoids redundant TFRecord scans.
    _cache_key = None
    _cache_records = []

    def _load_record(source_shard, record_index):
        nonlocal _cache_key, _cache_records
        shard_path = select_single_shard_for_inspection(
            expansion_config, source_shard=source_shard, record_index=record_index
        )
        dataset_config = build_waymax_config(expansion_config, shard_path)
        key = (str(dataset_config), expansion_config.dataset_name, expansion_config.split)
        if key != _cache_key:
            _cache_key = key
            _cache_records = []
        if len(_cache_records) < record_index + 1:
            _cache_records = list(
                iter_scenarios(
                    dataset_config,
                    limit=record_index + 1,
                    source_dataset=expansion_config.dataset_name,
                    source_split=expansion_config.split,
                )
            )
        return _cache_records[record_index]

    failures = []
    for spec in specs:
        record = _load_record(spec.source_shard, spec.record_index)
        log_trajectory = record.state.log_trajectory
        sdc_index = record.sdc_index

        ego_x = np.asarray(log_trajectory.x[sdc_index])
        ego_y = np.asarray(log_trajectory.y[sdc_index])
        ego_yaw = np.asarray(log_trajectory.yaw[sdc_index])
        ego_valid = np.asarray(log_trajectory.valid[sdc_index]).astype(bool)
        polylines = extract_lane_polylines(record.state.roadgraph_points)

        try:
            result = compute_decision_start_frame(
                ego_x, ego_y, ego_yaw, ego_valid, polylines, lane_assignment_config,
                source_lane_id=spec.lane_chain[0],
                merge_start_frame=spec.merge_start_frame,
            )
        except DecisionStartUnresolvedError as exc:
            failures.append(f"{spec.maneuver_id}: UNRESOLVED ({exc})")
            continue

        if result.decision_start_frame > spec.merge_start_frame:
            failures.append(
                f"{spec.maneuver_id}: decision_start_frame="
                f"{result.decision_start_frame} > merge_start_frame="
                f"{spec.merge_start_frame}"
            )

    assert not failures, "\n".join(failures)


def test_decision_start_unresolved_raises_not_silently_falls_back():
    """A maneuver whose source lane never raw-matches must raise
    DecisionStartUnresolvedError -- never silently fall back to
    merge_start_frame (Stage B-2.8 Section 6)."""

    config = load_lane_assignment_config(MERGE_CONFIG_PATH)
    ego_x = np.array([0.0, 1.0, 2.0])
    ego_y = np.array([0.0, 1.0, 2.0])
    ego_yaw = np.array([0.0, 0.0, 0.0])
    ego_valid = np.array([True, True, True])

    with pytest.raises(DecisionStartUnresolvedError):
        compute_decision_start_frame(
            ego_x, ego_y, ego_yaw, ego_valid,
            polylines=[],  # no polylines -> can never match any lane
            lane_assignment_config=config,
            source_lane_id=999999,
            merge_start_frame=10,
        )


# ======================================================================
# Actual reset timestep == decision_start_frame
# ======================================================================


def test_actual_reset_timestep_equals_decision_start_frame(env):
    """The real Waymax state's timestep after reset() must equal the
    resolved decision_start_frame exactly -- not merge_start_frame,
    not init_steps - 1 miscounted."""

    _, info = env.reset(SINGLE_MANEUVER)

    assert int(env._state.timestep) == info["decision_start_frame"]
    assert info["decision_start_frame"] <= info["merge_start_frame"]


def test_reset_timestep_matches_decision_start_for_chained_maneuver(env):
    """Same invariant for a chained (multi-transition) maneuver."""

    chained = ManeuverSpec(
        maneuver_id="TEST_CHAINED",
        source_shard=SINGLE_MANEUVER.source_shard,
        record_index=SINGLE_MANEUVER.record_index,
        lane_chain=[637, 628, 628],
        candidate_ids=[
            "training_tfexample.tfrecord-00000-of-01000#12__t49__637_628",
            "training_tfexample.tfrecord-00000-of-01000#12__t49__637_628",
        ],
        merge_start_frame=36,
    )
    _, info = env.reset(chained)
    assert int(env._state.timestep) == info["decision_start_frame"]


# ======================================================================
# Observation validity at the new reset
# ======================================================================


def test_reset_observation_finite_14d_at_decision_start(env):
    observation, info = env.reset(SINGLE_MANEUVER)

    assert observation.shape == (14,)
    assert np.all(np.isfinite(observation))
    assert info["decision_start_frame"] is not None


# ======================================================================
# Horizon preserves old absolute opportunity, safely capped
# ======================================================================


def test_episode_horizon_extends_by_start_shift(env):
    """_resolve_episode_horizon must add the start_shift on top of the
    base MAX_EPISODE_HORIZON_FRAMES, subject to the scenario's own
    logged-length cap -- i.e. it can never be LESS than
    min(100, max_steps_available), and if start_shift > 0 with room to
    spare it must be strictly greater than the base 100."""

    from src.environment.termination import MAX_EPISODE_HORIZON_FRAMES

    _, info = env.reset(SINGLE_MANEUVER)
    shift = info["merge_start_frame"] - info["decision_start_frame"]
    num_logged_frames = int(np.asarray(env._state.log_trajectory.x).shape[-1])
    max_steps_available = (num_logged_frames - 1) - info["decision_start_frame"]
    expected = min(MAX_EPISODE_HORIZON_FRAMES + max(shift, 0), max_steps_available)

    assert info["episode_horizon"] == expected
    assert info["episode_horizon"] <= max_steps_available


def test_episode_horizon_never_exceeds_scenario_frame_bound(env):
    """The environment must never be steppable past the scenario's
    own last logged frame index -- Waymax has no data beyond it."""

    _, info = env.reset(SINGLE_MANEUVER)
    num_logged_frames = int(np.asarray(env._state.log_trajectory.x).shape[-1])

    for _ in range(info["episode_horizon"]):
        _, _, terminated, truncated, info = env.step(BehaviorAction.KEEP)
        assert int(env._state.timestep) < num_logged_frames
        if terminated or truncated:
            break


# ======================================================================
# Canonical split / Phase 1 label preservation
# ======================================================================


def test_canonical_split_manifest_unchanged():
    """Stage B-2.8 must not rewrite data/manifests/phase2_dataset_split.csv.
    Checked via row counts and split-membership invariants rather than
    a hash (the file's exact bytes are not this test's concern, its
    canonical CONTENT is)."""

    rows = load_split_manifest()
    train = [r for r in rows if r.split == "train"]
    validation = [r for r in rows if r.split == "validation"]

    assert len(train) == 110
    assert len(validation) == 58

    train_scenes = {r.scene_key for r in train}
    validation_scenes = {r.scene_key for r in validation}
    assert train_scenes.isdisjoint(validation_scenes)


def test_phase1_canonical_maneuver_table_row_count_unchanged():
    """Phase 1's maneuver table must still contain exactly the
    canonical 168-maneuver pool -- Stage B-2.8 does not add, remove,
    or relabel maneuvers."""

    with open(MANEUVER_TABLE, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 168
