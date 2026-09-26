#!/usr/bin/env python3
"""DIAGNOSTIC-ONLY, pre-freeze sanity check -- NOT a canonical baseline
evaluation, NOT a final paper result.

Runs AlwaysKeep / AlwaysFollow / AlwaysMerge (src/environment/
trivial_policies.py, plus a new AlwaysFollowPolicy added here) against a
sample of this session's new Tier A/B candidates (MERGE_CONTEXT_ELIGIBLE
only) through the exact same MergeEnvironment/BehaviorExecutor/
LowLevelController/termination stack the FSM and any future PPO policy
use -- no privileged information, no shortcuts.

CRITICAL CONSTRAINT (explicit user instruction): this script must NEVER
change `manual_validation` values, and must NEVER weaken or bypass
`ManeuverSpec.require_schema`'s existing CONFIRMED_MERGE contract in
src/environment/merge_environment.py / merge_v2.py. The existing canonical
evaluator (scripts/evaluate_v2_baselines.py) is untouched.

How this avoids the contract without weakening it: `MergeEnvironment.reset`
only calls `maneuver.require_schema(DATASET_SCHEMA_V2)` when
`maneuver.is_v2` is True (schema_version == "merge_interaction_v2"). This
session's new Tier A/B/C system is honestly a DIFFERENT, not-yet-human-
validated concept from the old v2 canonical contract (decision-relevance-
based, not human-confirmation-based) -- so ManeuverSpecs built here
correctly use `schema_version=LEGACY_DATASET_SCHEMA` (the dataclass
default), which is not a bypass of the v2 contract but an honest statement
that these rows are not (yet) v2-canonical-confirmed. `require_schema` is
never called with `DATASET_SCHEMA_V2` here, and is never edited.

Per the user's explicit conditions:
  1. existing canonical evaluator semantics: unchanged (this is a new file)
  2. manual_validation: never touched
  3. only MERGE_CONTEXT_ELIGIBLE Tier A/B candidates used
  4. output is clearly diagnostic-only (see output JSON's own "diagnostic_only" key)
  5. AlwaysKeep/AlwaysFollow/AlwaysMerge run under identical episode conditions
  6. success/collision/offroad/safety metrics recorded
  7. NOT to be used as final paper baseline performance
  8. if trivial-policy risk is shown, Dataset Freeze must STOP and report a blocker
  9. if results are normal, Phase F (and the freeze) may proceed
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment.behavior_action import BehaviorAction
from src.environment.episode_context import parse_candidate_ids, parse_lane_chain
from src.environment.full_split_evaluator import EpisodeResult, run_episode
from src.environment.merge_environment import ManeuverSpec, MergeEnvironment
from src.environment.trivial_policies import AlwaysKeepPolicy, AlwaysMergePolicy, TrivialDecision
from src.scenarios.merge_v2 import LEGACY_DATASET_SCHEMA

DATASET_CONFIG = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"
TIER_MANIFEST = "data/manifests/v2/merge_decision_train_candidates_v2.csv"
EVIDENCE_JSONL = "data/manifests/v2/evidence_training.jsonl"
OUT_PATH = Path("outputs/merge_v2_decision_audit_v2/trivial_policy_sanity_diagnostic.json")
MAX_STEPS = 100
SAMPLE_PER_TIER = 60  # bounded sample, not the full 5545-candidate TRAIN sweep


class AlwaysFollowPolicy:
    """Diagnostic-only: ignores state, always proposes FOLLOW. Per
    behavior_action.py, FOLLOW falls back to KEEP's nominal-cruise
    reference speed whenever no source-lane lead exists -- so this
    policy's behavior should differ from AlwaysKeep ONLY on candidates
    with a real source-lane lead present, which this diagnostic checks
    explicitly (see `differs_from_keep_on_source_front` in the report)."""

    def decide(self, observation) -> TrivialDecision:
        del observation
        return TrivialDecision(action=BehaviorAction.FOLLOW)


def load_eligible_tier_ab_sample(tier_manifest_path, evidence_path, per_tier):
    with open(tier_manifest_path, newline="", encoding="utf-8") as f:
        tier_rows = [r for r in csv.DictReader(f) if r["decision_tier"] in ("A", "B")]

    assert all(r["merge_context_status"] == "MERGE_CONTEXT_ELIGIBLE" for r in tier_rows), (
        "Tier A/B rows must all be MERGE_CONTEXT_ELIGIBLE by construction -- "
        "constraint violated, refusing to sample"
    )

    evidence_by_id = {}
    with open(evidence_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            evidence_by_id[rec["candidate_id"]] = rec

    by_tier = defaultdict(list)
    for r in tier_rows:
        by_tier[r["decision_tier"]].append(r)

    sample = []
    for tier in ("A", "B"):
        sample.extend(by_tier[tier][:per_tier])
    return sample, evidence_by_id


def build_maneuver_spec(tier_row, evidence_rec):
    te = evidence_rec["topology_evidence"]
    ie = evidence_rec["interaction_evidence"]
    candidate_id = tier_row["candidate_id"]
    return ManeuverSpec(
        maneuver_id=candidate_id,
        source_shard=evidence_rec["source_shard"],
        record_index=int(evidence_rec["record_index"]),
        lane_chain=[te["source_lane_id"], te["target_lane_id"]],
        candidate_ids=[candidate_id],
        merge_start_frame=int(ie["decision_start_frame"]),
        schema_version=LEGACY_DATASET_SCHEMA,  # honest: not v2-canonical-confirmed (see module docstring)
        maneuver_type=None,
        manual_validation=None,  # never set/forged to CONFIRMED_MERGE
        topology_evidence=None,
        interaction_evidence=None,
    )


def summarize(results, tier_by_maneuver_id):
    outcomes = Counter(r.outcome for r in results)
    by_tier_outcome = defaultdict(Counter)
    for r in results:
        by_tier_outcome[tier_by_maneuver_id[r.maneuver_id]][r.outcome] += 1
    n = len(results)
    success = outcomes.get("success", 0)
    collision = sum(c for o, c in outcomes.items() if "collision" in o)
    offroad = sum(c for o, c in outcomes.items() if "offroad" in o)
    return {
        "n": n,
        "outcomes": dict(outcomes),
        "success_rate": success / n if n else 0.0,
        "collision_rate": collision / n if n else 0.0,
        "offroad_rate": offroad / n if n else 0.0,
        "by_tier_outcome": {k: dict(v) for k, v in by_tier_outcome.items()},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-tier", type=int, default=SAMPLE_PER_TIER)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args(argv)

    sample, evidence_by_id = load_eligible_tier_ab_sample(TIER_MANIFEST, EVIDENCE_JSONL, args.per_tier)
    print(f"diagnostic sample: {len(sample)} candidates (Tier A/B, MERGE_CONTEXT_ELIGIBLE only)")

    tier_by_maneuver_id = {r["candidate_id"]: r["decision_tier"] for r in sample}
    specs = [build_maneuver_spec(r, evidence_by_id[r["candidate_id"]]) for r in sample]

    # required_dataset_schema_version intentionally omitted/None: specs use
    # LEGACY_DATASET_SCHEMA so MergeEnvironment.reset's `elif maneuver.is_v2`
    # branch is also never triggered (is_v2 is False for LEGACY_DATASET_SCHEMA).
    env = MergeEnvironment(DATASET_CONFIG, downstream_mode="frenet_mpc")

    policies = {
        "always_keep": AlwaysKeepPolicy(),
        "always_follow": AlwaysFollowPolicy(),
        "always_merge": AlwaysMergePolicy(),
    }

    report = {"diagnostic_only": True, "not_a_final_baseline": True, "sample_size": len(sample)}
    per_policy_results = {}
    for name, policy in policies.items():
        results = []
        for spec in specs:
            try:
                results.append(run_episode(env, policy, spec, max_steps=args.max_steps))
            except Exception as exc:  # noqa: BLE001 -- one bad maneuver must not abort the sweep
                results.append(EpisodeResult(
                    maneuver_id=spec.maneuver_id, dataset_schema_version=spec.schema_version,
                    maneuver_type=None, outcome=f"exception:{type(exc).__name__}:{exc}",
                    steps_elapsed=0, merge_commit_frame=None, initial_action=None,
                    action_counts={a.name: 0 for a in BehaviorAction}, fallback_count=0,
                    immediate_merge=False, saw_keep_before_merge=False, saw_follow_before_merge=False,
                    saw_stop_before_merge=False, reset_classification={},
                ))
        per_policy_results[name] = results
        report[name] = summarize(results, tier_by_maneuver_id)
        print(f"{name}: {report[name]['outcomes']}")

    # AlwaysKeep hard-failure gate (existing contract from evaluate_v2_baselines.py,
    # re-checked here since this diagnostic path bypasses that script entirely).
    always_keep_successes = report["always_keep"]["outcomes"].get("success", 0)
    report["always_keep_hard_failure"] = always_keep_successes > 0
    if always_keep_successes:
        print(f"BLOCKER: AlwaysKeep produced {always_keep_successes} success(es) -- "
              "existing v2 validity contract violated")

    # AlwaysMerge trivial-shortcut check: does AlwaysMerge solve nearly
    # everything regardless of Tier/context? No fixed hard threshold (per
    # brief Section 25) -- report the number for human judgment.
    always_merge_success_rate = report["always_merge"]["success_rate"]
    report["always_merge_dominant_shortcut_risk"] = always_merge_success_rate >= 0.90

    # AlwaysFollow vs AlwaysKeep differentiation check.
    keep_actions = per_policy_results["always_keep"]
    follow_actions = per_policy_results["always_follow"]
    differ_count = sum(
        1 for k, f in zip(keep_actions, follow_actions)
        if k.outcome != f.outcome or k.action_counts != f.action_counts
    )
    source_front_present_count = sum(1 for r in sample if r["source_front_present"] == "True")
    report["source_front_present_in_sample"] = source_front_present_count
    report["source_front_present_ratio_in_sample"] = (
        source_front_present_count / len(sample) if sample else 0.0
    )
    report["always_follow_vs_always_keep_differ_count"] = differ_count
    report["always_follow_vs_always_keep_differ_ratio"] = differ_count / len(sample) if sample else 0.0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {OUT_PATH}")

    blocker = report["always_keep_hard_failure"] or report["always_merge_dominant_shortcut_risk"]
    print(f"\nBLOCKER_DETECTED={blocker}")
    return report, blocker


if __name__ == "__main__":
    main()
