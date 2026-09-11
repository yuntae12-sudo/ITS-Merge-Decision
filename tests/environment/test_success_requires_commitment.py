"""Stage B-2.5 regression test: `check_online_causal_merge_success`
must never be evaluated -- let alone reported as SUCCESS -- unless the
episode has actually committed to MERGE (`DecisionState.is_committed`).

Bug found during the Stage B-2.5 decision-problem validity audit: a
trivial AlwaysKeep sanity-check policy (never selects MERGE) reached
`termination_reason == "success"` on a real maneuver (MAN_0001) at
frame 13, purely from incidental lane-keeping drift -- source and
target lane centerlines are geometrically close/converging near the
merge point (true by definition of a merge scenario), so ego's stable
lane assignment can cross into the target lane's geometric envelope
without the policy ever choosing to merge.
`termination.py`'s own docstring already documented this check as
"only meaningful once the episode is in MERGE_COMMITTED phase", but
`MergeEnvironment._check_final_success`/`_maybe_advance_chain` never
enforced that precondition until this fix (both now short-circuit to
False unless `self._decision_state.is_committed`).
"""

from src.environment.behavior_action import BehaviorAction
from src.environment.merge_environment import ManeuverSpec, MergeEnvironment

DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"

# Same real single-candidate maneuver used throughout Stage B-1/B-2
# tests -- previously confirmed to spuriously "succeed" under pure
# KEEP at frame 13, before this fix.
SINGLE_MANEUVER = ManeuverSpec(
    maneuver_id="MAN_0001",
    source_shard="training_tfexample.tfrecord-00000-of-01000",
    record_index=12,
    lane_chain=[637, 628],
    candidate_ids=["training_tfexample.tfrecord-00000-of-01000#12__t49__637_628"],
    merge_start_frame=36,
)


def test_pure_keep_never_reports_success():
    """Negative regression guard: KEEP the whole episode (never
    selecting MERGE) must never report success, regardless of any
    incidental geometric drift toward the target lane."""

    env = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)
    observation, info = env.reset(SINGLE_MANEUVER)

    for _ in range(60):
        observation, reward, terminated, truncated, info = env.step(BehaviorAction.KEEP)
        assert info["termination_reason"] != "success"
        assert info["merge_committed"] is False
        assert info["decision_phase"] == "decision"
        if terminated or truncated:
            break


def test_merge_driven_rollout_still_succeeds():
    """Positive regression guard: the fix must not have broken the
    genuine MERGE-commitment success path -- a policy that actually
    selects MERGE on the same maneuver still reaches success."""

    env = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)
    observation, info = env.reset(SINGLE_MANEUVER)

    terminated = truncated = False
    for _ in range(60):
        observation, reward, terminated, truncated, info = env.step(BehaviorAction.MERGE)
        if terminated or truncated:
            break

    assert terminated
    assert info["termination_reason"] == "success"
    assert info["merge_committed"] is True
