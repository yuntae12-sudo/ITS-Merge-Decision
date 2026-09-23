"""trace.csv writer for one PPO visualization episode (docs/ppo/
PPO_VISUALIZATION_GUIDE.md). Pure serialization -- every field comes
straight off ``src.visualization.ppo_rollout.PPOStepRecord``, nothing
computed/derived here beyond flattening the 14D observation array into
named columns (``src.environment.observation_builder.OBSERVATION_FIELD_NAMES``
order, the same order the frozen environment itself uses).
"""

import csv
from typing import List

from src.environment.observation_builder import OBSERVATION_FIELD_NAMES
from src.visualization.ppo_rollout import PPOEpisodeResult

TRACE_FIELDNAMES = [
    "step",
    "maneuver_id",
    "is_policy_step",
    "merge_committed",
    "selected_action",
    "action_index",
    "dataset_schema_version",
    "maneuver_type",
    "active_source_lane_id",
    "active_target_lane_id",
    "current_stable_lane_id",
    "v2_saw_relevant_interaction",
    "keep_prob",
    "follow_prob",
    "merge_prob",
    "stop_prob",
    "value_estimate",
    "reward_total",
    "reward_terminal_component",
    "reward_decision_cost_component",
    *OBSERVATION_FIELD_NAMES,
    "downstream_status",
    "intervention_rate",
    "fallback_applied",
    "terminated",
    "truncated",
    "termination_reason",
    "ego_x",
    "ego_y",
    "ego_yaw",
    "ego_speed_mps",
]


def write_trace_csv(episode: PPOEpisodeResult, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_FIELDNAMES)
        writer.writeheader()
        for record in episode.steps:
            row = {
                "step": record.step_index,
                "maneuver_id": record.maneuver_id,
                "is_policy_step": record.is_policy_step,
                "merge_committed": record.merge_committed_before,
                "selected_action": record.selected_action,
                "action_index": record.action_index,
                "dataset_schema_version": record.dataset_schema_version,
                "maneuver_type": record.maneuver_type,
                "active_source_lane_id": record.active_source_lane_id,
                "active_target_lane_id": record.active_target_lane_id,
                "current_stable_lane_id": record.current_stable_lane_id,
                "v2_saw_relevant_interaction": record.v2_saw_relevant_interaction,
                "keep_prob": record.keep_prob,
                "follow_prob": record.follow_prob,
                "merge_prob": record.merge_prob,
                "stop_prob": record.stop_prob,
                "value_estimate": record.value_estimate,
                "reward_total": record.reward_total,
                "reward_terminal_component": record.reward_terminal_component,
                "reward_decision_cost_component": record.reward_decision_cost_component,
                "downstream_status": record.downstream_status,
                "intervention_rate": record.intervention_rate,
                "fallback_applied": record.fallback_applied,
                "terminated": record.terminated,
                "truncated": record.truncated,
                "termination_reason": record.termination_reason,
                "ego_x": record.ego_x,
                "ego_y": record.ego_y,
                "ego_yaw": record.ego_yaw,
                "ego_speed_mps": record.ego_speed_mps,
            }
            for name, value in zip(OBSERVATION_FIELD_NAMES, record.observation.tolist()):
                row[name] = value
            writer.writerow(row)


__all__ = ["TRACE_FIELDNAMES", "write_trace_csv"]
