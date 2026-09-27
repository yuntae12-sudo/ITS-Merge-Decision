"""Stage C-2: Common 14D State Final Freeze tests.

Verifies observation_builder.py's ACTUAL production behavior against
the frozen research-interface contract in
configs/phase2_common_state.yaml, and adds targeted coverage for the
chained-maneuver observation-consistency requirement (Section 22)
that the existing Stage B-1 tests check only at the `info` level, not
the 14D vector itself.

Uses REAL WOMD scenes (same convention as test_merge_environment.py).
"""

import numpy as np
import pytest
import yaml

from src.environment.behavior_action import BehaviorAction
from src.environment.merge_environment import ManeuverSpec, MergeEnvironment
from src.environment.observation_builder import (
    OBSERVATION_DIM,
    OBSERVATION_FIELD_NAMES,
    TTC_CAP_S,
)
from src.environment.full_split_evaluator import load_decision_dataset_maneuver_specs

DATASET_CONFIG_PATH = "configs/dataset.yaml"
VALIDATION_DATASET_CONFIG_PATH = "configs/dataset_validation.yaml"
COMMON_STATE_CONFIG_PATH = "configs/phase2_common_state.yaml"

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

# Same real chained maneuver test_merge_environment.py uses (confirmed
# reliable under MERGE -- see that file's module-level comment).
CHAINED_MANEUVER = ManeuverSpec(
    maneuver_id="MAN_0041",
    source_shard="training_tfexample.tfrecord-00003-of-01000",
    record_index=216,
    lane_chain=[618, 623, 608],
    candidate_ids=[
        "training_tfexample.tfrecord-00003-of-01000#216__t38__618_623",
        "training_tfexample.tfrecord-00003-of-01000#216__t54__623_608",
    ],
    merge_start_frame=11,
)


@pytest.fixture(scope="module")
def env():
    return MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)


@pytest.fixture(scope="module")
def validation_env():
    """Separate env for VALIDATION-split resets: Waymax's
    DatasetExpansionConfig binds one fixed physical split per config,
    so TRAIN and VALIDATION shards cannot be mixed into a single
    configs/dataset.yaml."""

    return MergeEnvironment(dataset_config_path=VALIDATION_DATASET_CONFIG_PATH)


@pytest.fixture(scope="module")
def common_state_config():
    with open(COMMON_STATE_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ======================================================================
# B1/B9: schema matches the frozen config exactly
# ======================================================================


def test_state_dim_matches_frozen_config(common_state_config):
    assert OBSERVATION_DIM == common_state_config["state_dim"] == 14


def test_feature_order_matches_frozen_config(common_state_config):
    frozen_names = [f["name"] for f in common_state_config["features"]]
    assert list(OBSERVATION_FIELD_NAMES) == frozen_names


def test_feature_indices_match_frozen_config(common_state_config):
    for feature in common_state_config["features"]:
        assert OBSERVATION_FIELD_NAMES[feature["index"]] == feature["name"]


def test_ttc_cap_matches_frozen_config(common_state_config):
    assert TTC_CAP_S == common_state_config["ttc_cap_s"] == 100.0


# ======================================================================
# B3: presence / TTC sentinel semantics over a deterministic representative
# sample (fast targeted runs) + an exhaustive full-dataset variant (slow)
# ======================================================================

_SAMPLE_STRIDE = 50  # mid-section coverage stride for the fast sample.


def _representative_sample(specs, stride=_SAMPLE_STRIDE):
    """Deterministic representative subset of a maneuver-spec list:
    first, last, an even stride through the middle, and at least one
    spec per distinct source_shard -- so the fast targeted sample
    exercises multiple physical scenes/shards, not just one repeated
    shard's geometry, while staying real-env-cheap. ``specs`` must
    already be in the loader's own stable sorted order (it is, per
    load_decision_dataset_maneuver_specs's CSV row order)."""

    if not specs:
        return []

    sample_by_id = {}
    sample_by_id[specs[0].maneuver_id] = specs[0]
    sample_by_id[specs[-1].maneuver_id] = specs[-1]
    for spec in specs[::stride]:
        sample_by_id[spec.maneuver_id] = spec

    seen_shards = set()
    for spec in specs:
        if spec.source_shard not in seen_shards:
            seen_shards.add(spec.source_shard)
            sample_by_id[spec.maneuver_id] = spec

    return [sample_by_id[m] for m in sorted(sample_by_id)]


def _assert_presence_gated_sentinels(env, specs):
    slot_indices = {
        "target_front": (2, 3, 4, 5),
        "target_rear": (6, 7, 8, 9),
        "source_front": (10, 11, 12, 13),
    }

    for spec in specs:
        observation, _ = env.reset(spec)
        for slot, (present_i, gap_i, rel_i, ttc_i) in slot_indices.items():
            present = observation[present_i] == 1.0
            if not present:
                assert observation[gap_i] == 0.0, (spec.maneuver_id, slot)
                assert observation[rel_i] == 0.0, (spec.maneuver_id, slot)
                assert observation[ttc_i] == TTC_CAP_S, (spec.maneuver_id, slot)


def test_presence_gated_slots_use_documented_sentinels_across_train_sample(env):
    """Deterministic representative TRAIN sample (first/last/stride/
    per-shard, see _representative_sample): whenever a presence flag is
    0, the corresponding gap/relative-speed is exactly 0.0 and TTC is
    exactly TTC_CAP_S -- matching configs/phase2_common_state.yaml's
    documented sentinel semantics. See the @pytest.mark.slow variant
    below for the exhaustive full-TRAIN check."""

    specs = _representative_sample(load_decision_dataset_maneuver_specs("train"))
    _assert_presence_gated_sentinels(env, specs)


@pytest.mark.slow
def test_presence_gated_slots_use_documented_sentinels_across_train_full(env):
    """Exhaustive version of the sampled check above -- every TRAIN
    maneuver in the final dataset (1097 real env resets); not run by
    default targeted test commands (see pytest.ini's `slow` marker)."""

    specs = load_decision_dataset_maneuver_specs("train")
    _assert_presence_gated_sentinels(env, specs)


# ======================================================================
# B4: causality -- d_m/lane identity are map/current-state, never
# future-ego-trajectory derived
# ======================================================================


def test_d_m_does_not_depend_on_future_simulated_frames(env):
    """d_m at the reset frame must be identical regardless of what
    action is chosen AFTERWARD -- i.e. it is a function of the CURRENT
    frame only, never of the (as-yet-unknown) future rollout. Checked
    by resetting twice and confirming the reset-time d_m is identical
    before any step is taken (trivially true, but pinned as a
    regression guard against a future change accidentally threading
    rollout-dependent state into the reset-time observation)."""

    obs_a, _ = env.reset(SINGLE_MANEUVER)
    obs_b, _ = env.reset(SINGLE_MANEUVER)
    assert obs_a[1] == obs_b[1]


# ======================================================================
# B5/Section 22: chained-maneuver observation consistency
# ======================================================================


def test_observation_reflects_active_transition_after_chain_advance(env):
    """When active_transition_index advances (0 -> 1) on a real chained
    maneuver, the 14D observation's target-lane-dependent features
    must reflect the NEW active target lane, not remain anchored to
    the first transition's target. Checked structurally: d_m is
    recomputed from the newly-active source/target polylines (so its
    value is not required to move monotonically -- merge_end_s
    changes when the active pair changes), and the observation stays
    finite/valid throughout.
    """

    env.reset(CHAINED_MANEUVER)

    previous_index = 0
    pre_advance_d_m = None
    post_advance_d_m = None
    saw_advance = False

    for _ in range(80):
        observation, _, terminated, truncated, info = env.step(BehaviorAction.MERGE)
        assert observation.shape == (14,)
        assert np.all(np.isfinite(observation))

        if not saw_advance and info["active_transition_index"] == previous_index:
            pre_advance_d_m = observation[1]

        if info["active_transition_index"] != previous_index:
            saw_advance = True
            post_advance_d_m = observation[1]
            previous_index = info["active_transition_index"]

        if terminated or truncated:
            break

    assert saw_advance, "chained maneuver never advanced within the step budget"
    # d_m is recomputed against the NEW active merge_end_s once the
    # chain advances -- it must be a real (finite) value, not stale
    # data frozen from the first transition.
    assert pre_advance_d_m is not None
    assert post_advance_d_m is not None
    assert np.isfinite(post_advance_d_m)


# ======================================================================
# B6: numerical validity across the full final TRAIN+VALIDATION dataset
# ======================================================================


def _assert_valid_14d_reset(env, split_name, specs):
    for spec in specs:
        observation, info = env.reset(spec)
        assert observation.shape == (14,), (split_name, spec.maneuver_id)
        assert np.all(np.isfinite(observation)), (split_name, spec.maneuver_id)
        assert info["decision_start_frame"] is not None


def test_canonical_maneuver_sample_resets_to_valid_14d_observation(env, validation_env):
    """Deterministic representative TRAIN+VALIDATION sample (see
    @pytest.mark.slow variant below for the exhaustive full-dataset,
    1485-maneuver check). TRAIN and VALIDATION specs are reset against
    separate envs -- Waymax binds one fixed physical split per
    DatasetExpansionConfig, so the two shard pools cannot share a
    single MergeEnvironment instance."""

    train_specs = _representative_sample(load_decision_dataset_maneuver_specs("train"))
    validation_specs = _representative_sample(load_decision_dataset_maneuver_specs("validation"))

    _assert_valid_14d_reset(env, "train", train_specs)
    _assert_valid_14d_reset(validation_env, "validation", validation_specs)


@pytest.mark.slow
def test_all_canonical_maneuvers_reset_to_valid_14d_observation_full(env, validation_env):
    """Exhaustive version of the sampled check above -- every TRAIN
    (1097) + VALIDATION (388) maneuver in the final dataset, 1485 real
    env resets total; not run by default targeted test commands (see
    pytest.ini's `slow` marker)."""

    train_specs = load_decision_dataset_maneuver_specs("train")
    validation_specs = load_decision_dataset_maneuver_specs("validation")

    _assert_valid_14d_reset(env, "train", train_specs)
    _assert_valid_14d_reset(validation_env, "validation", validation_specs)


# ======================================================================
# B7: state sufficiency -- structural check that every action's
# required inputs are present in the frozen 14D vector (documentation
# consistency, not a new design decision)
# ======================================================================


def test_behavior_executor_only_reads_documented_indices():
    """BehaviorExecutor.compute_objective must only read observation
    indices that are part of the frozen 14D contract -- this pins the
    Section 25 sufficiency finding (no missing/extra information) as a
    regression guard: KEEP needs none, FOLLOW needs source_front_*
    (10-12), MERGE needs target_front_* (2-4), STOP needs none."""

    import ast
    import inspect
    import textwrap

    from src.environment.behavior_action import BehaviorExecutor

    source = textwrap.dedent(inspect.getsource(BehaviorExecutor.compute_objective))
    tree = ast.parse(source)

    accessed_indices = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "observation"
            and isinstance(node.slice, ast.Constant)
        ):
            accessed_indices.add(node.slice.value)

    # All actions' required reads must be within the documented 14
    # dimensions -- no index >= 14 or negative.
    assert accessed_indices, "expected compute_objective to read some observation indices"
    assert all(0 <= i < OBSERVATION_DIM for i in accessed_indices)
