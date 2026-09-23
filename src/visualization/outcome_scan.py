"""Deterministic outcome scan + representative-episode selection for
the PPO visualization pipeline (docs/ppo/PPO_VISUALIZATION_GUIDE.md).

Runs a restored PPO checkpoint's policy, deterministically by default,
across a set of maneuvers and records each episode's real outcome
(Source of Truth: MergeEnvironment's own terminated/truncated/
termination_reason -- see src.visualization.ppo_rollout). Selection of
one representative episode per outcome category is a fixed, stable-
sorted-order rule -- never a "best representative" heuristic.
"""

import csv
import dataclasses
from typing import Dict, List, Optional

from src.visualization.ppo_rollout import (
    OUTCOME_COLLISION,
    OUTCOME_EXCEPTION,
    OUTCOME_OFFROAD,
    OUTCOME_SUCCESS,
    OUTCOME_TIMEOUT,
    PPOEpisodeResult,
)

# Fixed selection priority: which outcome categories the pipeline tries
# to find one representative episode for, and in what order they are
# reported. "offroad"/"exception" are optional/best-effort (Section 5
# of the task spec: only success/collision/timeout are guaranteed
# categories).
OUTCOME_CATEGORIES = (
    OUTCOME_SUCCESS,
    OUTCOME_COLLISION,
    OUTCOME_TIMEOUT,
    OUTCOME_OFFROAD,
)


@dataclasses.dataclass
class OutcomeIndexRow:
    maneuver_id: str
    dataset_schema_version: str
    maneuver_type: Optional[str]
    outcome: str
    termination_reason: Optional[str]
    steps: int
    merge_commit_step: Optional[int]
    policy_decision_count: int
    intervention_rate: float
    exception_repr: Optional[str] = None


def build_outcome_index(episode_results: List[PPOEpisodeResult]) -> List[OutcomeIndexRow]:
    """Builds the outcome_index.csv rows, one per evaluated maneuver, in
    the SAME order ``episode_results`` was produced (callers are
    responsible for iterating maneuvers in sorted maneuver_id order so
    this stays deterministic/reproducible -- see
    ``scripts/visualization/visualize_ppo_run.py``)."""

    schemas = {
        result.steps[0].dataset_schema_version
        for result in episode_results
        if result.steps
    }
    if len(schemas) > 1:
        raise ValueError(f"Mixed dataset schemas are not aggregatable: {sorted(schemas)}")

    rows = []
    for result in episode_results:
        rows.append(
            OutcomeIndexRow(
                maneuver_id=result.maneuver_id,
                dataset_schema_version=(
                    result.steps[0].dataset_schema_version if result.steps else "unknown"
                ),
                maneuver_type=(result.steps[0].maneuver_type if result.steps else None),
                outcome=result.outcome,
                termination_reason=result.termination_reason,
                steps=result.physical_step_count,
                merge_commit_step=result.merge_commit_step,
                policy_decision_count=result.policy_decision_count,
                intervention_rate=result.final_intervention_rate,
                exception_repr=result.exception_repr,
            )
        )
    return rows


def write_outcome_index_csv(rows: List[OutcomeIndexRow], path: str) -> None:
    fieldnames = [
        "maneuver_id",
        "dataset_schema_version",
        "maneuver_type",
        "outcome",
        "termination_reason",
        "steps",
        "merge_commit_step",
        "policy_decision_count",
        "intervention_rate",
        "exception_repr",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))


DEFAULT_MAX_PER_OUTCOME = 5


def select_representative_episodes(
    episode_results: List[PPOEpisodeResult],
    max_per_outcome: Optional[int] = DEFAULT_MAX_PER_OUTCOME,
) -> Dict[str, Dict]:
    """Deterministically selects up to ``max_per_outcome`` episodes (in
    the order given, which callers must produce in stable-sorted
    maneuver_id order) for each outcome category in
    ``OUTCOME_CATEGORIES``.

    ``max_per_outcome=None`` means "no cap" -- selects EVERY episode of
    that outcome in the evaluated scope (used by ``--render-all``).
    ``max_per_outcome=5`` (the default) selects the first 5, in stable
    sorted-maneuver_id order -- never a "best representative"
    heuristic, and never fewer than what exists if fewer than 5 exist
    (e.g. 3 timeout episodes -> all 3 selected).

    Returns a dict keyed by outcome category. Each value is
    ``{"maneuver_ids": [...], "reason": ...}``. An outcome with zero
    matching episodes gets an empty list and an explanatory reason --
    never raises/fails the whole run.
    """

    by_outcome: Dict[str, List[PPOEpisodeResult]] = {}
    for result in episode_results:
        by_outcome.setdefault(result.outcome, []).append(result)

    selected = {}
    for outcome in OUTCOME_CATEGORIES:
        candidates = by_outcome.get(outcome, [])
        if not candidates:
            selected[outcome] = {
                "maneuver_ids": [],
                "reason": f"{outcome} episode not found in evaluated scope",
            }
            continue

        if max_per_outcome is None:
            chosen = candidates
            reason = "all episodes for this outcome (--render-all)"
        else:
            chosen = candidates[:max_per_outcome]
            if len(candidates) <= max_per_outcome:
                reason = (
                    f"all {len(candidates)} episode(s) for this outcome "
                    f"(<= max_per_outcome={max_per_outcome})"
                )
            else:
                reason = (
                    f"first {max_per_outcome} of {len(candidates)} episodes "
                    "in stable sorted-maneuver_id order"
                )

        selected[outcome] = {
            "maneuver_ids": [c.maneuver_id for c in chosen],
            "reason": reason,
        }
    return selected


__all__ = [
    "OUTCOME_CATEGORIES",
    "DEFAULT_MAX_PER_OUTCOME",
    "OutcomeIndexRow",
    "build_outcome_index",
    "write_outcome_index_csv",
    "select_representative_episodes",
]
