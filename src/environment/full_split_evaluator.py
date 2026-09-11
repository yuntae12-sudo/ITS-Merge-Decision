"""Phase 2 Stage B-2.5: full-split deterministic policy evaluator.

Shared by the Rule-FSM and the trivial sanity-check policies
(AlwaysMerge/AlwaysKeep/AlwaysStop) so their results are directly
comparable: identical MergeEnvironment, identical maneuver iteration
order, identical horizon, identical metrics computation. No policy
gets privileged information or a different evaluation harness.
"""

import csv
import dataclasses
from typing import List, Optional

from src.environment.behavior_action import BehaviorAction
from src.environment.dataset_split import load_split_manifest
from src.environment.merge_environment import ManeuverSpec, MergeEnvironment

MANEUVER_TABLE = "outputs/phase1/training_10shard_pilot/training_visual_merge_maneuvers.csv"
CANDIDATE_MANIFEST = "outputs/phase1/training_10shard_pilot/merge_manifest_training_scratch.csv"
DEFAULT_MAX_STEPS = 100  # matches MAX_EPISODE_HORIZON_FRAMES


def load_maneuver_specs(which_split: str, split_manifest_path: Optional[str] = None) -> List[ManeuverSpec]:
    """Loads every ManeuverSpec belonging to one split (no subsampling
    -- Stage B-2.5 explicitly requires the FULL canonical split, unlike
    Stage B-1.5/B-2's representative-sample checks)."""

    if split_manifest_path is not None:
        split_rows = {r.maneuver_id: r.split for r in load_split_manifest(split_manifest_path)}
    else:
        split_rows = {r.maneuver_id: r.split for r in load_split_manifest()}

    with open(CANDIDATE_MANIFEST, newline="") as f:
        manifest_by_candidate_id = {row["candidate_id"]: row for row in csv.DictReader(f)}

    specs = []
    with open(MANEUVER_TABLE, newline="") as f:
        for row in csv.DictReader(f):
            if split_rows.get(row["maneuver_id"]) != which_split:
                continue
            try:
                spec = ManeuverSpec.from_csv_row(row, manifest_by_candidate_id)
            except KeyError:
                continue
            specs.append(spec)
    return specs


@dataclasses.dataclass
class EpisodeResult:
    maneuver_id: str
    outcome: str  # "success" | "failure_collision" | "failure_offroad" | "timeout" | "exception"
    steps_elapsed: int
    merge_commit_frame: Optional[int]  # None if never committed within horizon
    initial_action: Optional[str]
    action_counts: dict  # BehaviorAction.name -> count, DECISION-phase steps only
    fallback_count: int
    immediate_merge: bool  # committed on the very first decision step
    saw_keep_before_merge: bool
    saw_follow_before_merge: bool
    saw_stop_before_merge: bool
    reset_classification: dict  # Section 5 classification, computed once at reset


def _classify_reset_state(observation) -> dict:
    """Section 5: classifies one reset observation against the FSM's
    own safety thresholds -- shared here so both the diversity audit
    and any per-episode reporting use one definition."""

    from src.environment.fsm_policy import (
        FOLLOW_GAP_M,
        FOLLOW_TTC_S,
        GAP_SAFE_M,
        STOP_MARGIN_M,
        TTC_SAFE_S,
        _source_lead_is_close,
        _target_gap_safe,
    )

    d_m = observation[1]
    target_front_present = observation[2] == 1.0
    target_front_gap = observation[3]
    target_front_ttc = observation[5]
    target_rear_present = observation[6] == 1.0
    target_rear_gap = observation[7]
    target_rear_ttc = observation[9]
    source_front_present = observation[10] == 1.0

    merge_safe = _target_gap_safe(observation)
    target_front_blocking = target_front_present and not (
        target_front_ttc >= TTC_SAFE_S and target_front_gap >= GAP_SAFE_M
    )
    target_rear_blocking = target_rear_present and not (
        target_rear_ttc >= TTC_SAFE_S and target_rear_gap >= GAP_SAFE_M
    )
    stop_margin_state = d_m < STOP_MARGIN_M
    source_front_relevant = source_front_present
    follow_relevant = _source_lead_is_close(observation)

    return {
        "merge_safe": merge_safe,
        "merge_unsafe": not merge_safe,
        "target_front_blocking": target_front_blocking,
        "target_rear_blocking": target_rear_blocking,
        "stop_margin_state": stop_margin_state,
        "source_front_relevant": source_front_relevant,
        "follow_relevant": follow_relevant,
    }


def run_episode(env: MergeEnvironment, policy, spec: ManeuverSpec, max_steps: int = DEFAULT_MAX_STEPS) -> EpisodeResult:
    """Runs one episode under one policy (anything exposing
    ``.decide(observation) -> object-with-.action``). Only counts an
    action toward ``action_counts``/diversity flags while the episode
    is still in DECISION phase (before MERGE commitment) -- matches
    ``DecisionState.policy_choice_available``, so a policy is never
    credited/debited for "re-selections" the environment doesn't
    actually solicit once committed.
    """

    observation, info = env.reset(spec)
    reset_classification = _classify_reset_state(observation)

    action_counts = {a.name: 0 for a in BehaviorAction}
    fallback_count = 0
    initial_action = None
    merge_commit_frame = None
    saw_keep = saw_follow = saw_stop = False
    immediate_merge = False

    outcome = "timeout"
    steps_elapsed = 0

    try:
        for frame in range(max_steps):
            was_committed_before = info.get("merge_committed", False)

            if not was_committed_before:
                decision = policy.decide(observation)
                action = decision.action
            else:
                # Environment doesn't actually consult the policy once
                # committed (decision_state.py); re-issue MERGE so
                # env.step()'s API contract (an action argument) is
                # still satisfied, without crediting this as a fresh
                # policy decision.
                action = BehaviorAction.MERGE

            if initial_action is None:
                initial_action = action.name

            if not was_committed_before:
                action_counts[action.name] += 1
                if action == BehaviorAction.KEEP:
                    saw_keep = True
                elif action == BehaviorAction.FOLLOW:
                    saw_follow = True
                elif action == BehaviorAction.STOP:
                    saw_stop = True
                elif action == BehaviorAction.MERGE and merge_commit_frame is None:
                    merge_commit_frame = frame
                    if frame == 0:
                        immediate_merge = True

            observation, reward, terminated, truncated, info = env.step(action)
            steps_elapsed = frame + 1

            if info.get("fallback_applied"):
                fallback_count += 1

            if terminated:
                reason = info["termination_reason"]
                outcome = reason
                break
            if truncated:
                outcome = "timeout"
                break
        else:
            outcome = "timeout"
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: a
        # genuine environment/geometry exception on one maneuver must
        # not crash the whole full-split sweep; it is recorded as its
        # own outcome category instead of silently skipped.
        outcome = f"exception:{type(exc).__name__}:{exc}"

    return EpisodeResult(
        maneuver_id=spec.maneuver_id,
        outcome=outcome,
        steps_elapsed=steps_elapsed,
        merge_commit_frame=merge_commit_frame,
        initial_action=initial_action,
        action_counts=action_counts,
        fallback_count=fallback_count,
        immediate_merge=immediate_merge,
        saw_keep_before_merge=saw_keep,
        saw_follow_before_merge=saw_follow,
        saw_stop_before_merge=saw_stop,
        reset_classification=reset_classification,
    )


def run_full_split(env: MergeEnvironment, policy, specs: List[ManeuverSpec], max_steps: int = DEFAULT_MAX_STEPS) -> List[EpisodeResult]:
    return [run_episode(env, policy, spec, max_steps=max_steps) for spec in specs]
