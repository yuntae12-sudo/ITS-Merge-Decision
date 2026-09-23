"""summary.json writer for one selected episode (docs/ppo/
PPO_VISUALIZATION_GUIDE.md). Pure serialization of
``PPOEpisodeResult`` fields -- no new metric/derivation.
"""

import json

from src.visualization.ppo_rollout import PPOEpisodeResult


def build_episode_summary(episode: PPOEpisodeResult) -> dict:
    return {
        "maneuver_id": episode.maneuver_id,
        "outcome": episode.outcome,
        "termination_reason": episode.termination_reason,
        "physical_step_count": episode.physical_step_count,
        "policy_decision_count": episode.policy_decision_count,
        "merge_commit_step": episode.merge_commit_step,
        "final_intervention_rate": episode.final_intervention_rate,
        "exception_repr": episode.exception_repr,
        "evaluation_kind": "POST-TRAINING FINAL-CHECKPOINT EVALUATION",
        "dataset_schema_version": (
            episode.steps[0].dataset_schema_version if episode.steps else None
        ),
        "maneuver_type": episode.steps[0].maneuver_type if episode.steps else None,
    }


def write_episode_summary(episode: PPOEpisodeResult, path: str) -> None:
    with open(path, "w") as f:
        json.dump(build_episode_summary(episode), f, indent=2, default=str)


__all__ = ["build_episode_summary", "write_episode_summary"]
