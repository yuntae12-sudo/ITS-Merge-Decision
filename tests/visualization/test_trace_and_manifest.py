"""Tests for src.visualization.trace_writer / manifest / episode_summary
(task spec Section 13.D/E): trace.csv step order + 14D observation
mapping + action semantics, and manifest/output-structure generation."""

import csv
import json

import numpy as np
import pytest

from src.environment.observation_builder import OBSERVATION_FIELD_NAMES
from src.visualization.episode_summary import build_episode_summary, write_episode_summary
from src.visualization.manifest import build_manifest
from src.visualization.ppo_checkpoint_policy import RestoredPPOCheckpoint
from src.visualization.ppo_rollout import PPOEpisodeResult, PPOStepRecord
from src.visualization.trace_writer import TRACE_FIELDNAMES, write_trace_csv


def _make_step(step_index, action_name, action_index, is_policy_step, merge_committed_before):
    obs = np.arange(14, dtype=np.float64) + step_index
    return PPOStepRecord(
        step_index=step_index,
        maneuver_id="MAN_TEST",
        is_policy_step=is_policy_step,
        merge_committed_before=merge_committed_before,
        selected_action=action_name,
        action_index=action_index,
        keep_prob=0.1 if is_policy_step else float("nan"),
        follow_prob=0.2 if is_policy_step else float("nan"),
        merge_prob=0.6 if is_policy_step else float("nan"),
        stop_prob=0.1 if is_policy_step else float("nan"),
        value_estimate=0.5,
        reward_total=0.0,
        reward_terminal_component=0.0,
        reward_decision_cost_component=0.0,
        observation=obs,
        downstream_status="OK",
        intervention_rate=0.0,
        fallback_applied=None,
        terminated=False,
        truncated=False,
        termination_reason=None,
        ego_x=1.0 + step_index,
        ego_y=2.0,
        ego_yaw=0.0,
        ego_speed_mps=10.0,
        agents=[],
        active_source_lane_id=1,
        active_target_lane_id=2,
        info={},
    )


def _make_episode():
    steps = [
        _make_step(0, "MERGE", 2, True, False),
        _make_step(1, "MERGE", 2, False, True),
        _make_step(2, "MERGE", 2, False, True),
    ]
    steps[-1].terminated = True
    steps[-1].termination_reason = "success"
    return PPOEpisodeResult(
        maneuver_id="MAN_TEST",
        outcome="success",
        termination_reason="success",
        steps=steps,
        physical_step_count=3,
        policy_decision_count=1,
        merge_commit_step=0,
        final_intervention_rate=0.0,
    )


def test_trace_csv_step_order_and_observation_mapping(tmp_path):
    episode = _make_episode()
    path = tmp_path / "trace.csv"
    write_trace_csv(episode, str(path))

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    assert [int(r["step"]) for r in rows] == [0, 1, 2]

    for i, row in enumerate(rows):
        for j, name in enumerate(OBSERVATION_FIELD_NAMES):
            assert float(row[name]) == float(i + j)

    assert rows[0]["selected_action"] == "MERGE"
    assert rows[0]["is_policy_step"] == "True"
    assert rows[0]["merge_committed"] == "False"
    assert rows[1]["is_policy_step"] == "False"
    assert rows[1]["merge_committed"] == "True"
    # Post-commitment frames must serialize NaN probabilities, never a
    # real decision value.
    assert rows[1]["keep_prob"] == "nan"


def test_trace_fieldnames_cover_required_categories():
    required_substrings = [
        "step", "maneuver_id", "is_policy_step", "merge_committed",
        "selected_action", "action_index",
        "keep_prob", "follow_prob", "merge_prob", "stop_prob", "value_estimate",
        "reward_total", "reward_terminal_component", "reward_decision_cost_component",
        "reward_safety_component", "reward_progress_component", "reward_decision_component",
        "reward_ttc_penalty_raw", "reward_gap_penalty_raw",
        "downstream_status", "intervention_rate", "fallback_applied",
        "terminated", "truncated", "termination_reason",
        "ego_x", "ego_y", "ego_yaw", "ego_speed_mps",
    ]
    for name in required_substrings:
        assert name in TRACE_FIELDNAMES
    for obs_field in OBSERVATION_FIELD_NAMES:
        assert obs_field in TRACE_FIELDNAMES


def test_v0_trace_backward_compatible_zero_v1_components(tmp_path):
    """A V0-style PPOStepRecord (constructed without any V1 keyword
    argument, as _make_step above does) must serialize all five new
    columns as exactly 0.0 -- never missing, never NaN."""

    episode = _make_episode()
    path = tmp_path / "trace.csv"
    write_trace_csv(episode, str(path))

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        for col in (
            "reward_safety_component", "reward_progress_component",
            "reward_decision_component", "reward_ttc_penalty_raw",
            "reward_gap_penalty_raw",
        ):
            assert float(row[col]) == 0.0


def test_v1_trace_contains_all_four_reward_components(tmp_path):
    step = _make_step(0, "MERGE", 2, True, False)
    step.reward_terminal_component = 1.0
    step.reward_safety_component = -0.02
    step.reward_progress_component = 0.1
    step.reward_decision_component = -0.02
    step.reward_ttc_penalty_raw = -1.0
    step.reward_gap_penalty_raw = 0.0
    step.reward_total = 1.0 - 0.02 + 0.1 - 0.02
    step.terminated = True
    step.termination_reason = "success"

    episode = PPOEpisodeResult(
        maneuver_id="MAN_TEST", outcome="success", termination_reason="success",
        steps=[step], physical_step_count=1, policy_decision_count=1,
        merge_commit_step=0, final_intervention_rate=0.0,
    )
    path = tmp_path / "trace.csv"
    write_trace_csv(episode, str(path))

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    row = rows[0]
    assert float(row["reward_terminal_component"]) == 1.0
    assert float(row["reward_safety_component"]) == -0.02
    assert float(row["reward_progress_component"]) == 0.1
    assert float(row["reward_decision_component"]) == -0.02

    # component-sum consistency: terminal + safety + progress + decision
    # == total (V1's legacy decision_cost is 0 in this example, matching
    # the correctness patch's contract).
    reconstructed = (
        float(row["reward_terminal_component"])
        + float(row["reward_safety_component"])
        + float(row["reward_progress_component"])
        + float(row["reward_decision_component"])
        + float(row["reward_decision_cost_component"])
    )
    assert reconstructed == pytest.approx(float(row["reward_total"]))

    # raw diagnostics present but distinguishable from the weighted
    # safety component (never silently identical/double-counted).
    assert float(row["reward_ttc_penalty_raw"]) == -1.0
    assert float(row["reward_gap_penalty_raw"]) == 0.0


def test_episode_summary_fields(tmp_path):
    episode = _make_episode()
    summary = build_episode_summary(episode)
    assert summary["maneuver_id"] == "MAN_TEST"
    assert summary["outcome"] == "success"
    assert summary["evaluation_kind"] == "POST-TRAINING FINAL-CHECKPOINT EVALUATION"

    path = tmp_path / "summary.json"
    write_episode_summary(episode, str(path))
    loaded = json.loads(path.read_text())
    assert loaded["maneuver_id"] == "MAN_TEST"


def test_manifest_contains_required_fields():
    restored = RestoredPPOCheckpoint(
        checkpoint_path="outputs/checkpoints/fake.pkl",
        payload=None,
        policy=None,
        value_network=None,
        value_params=None,
        network_hidden_sizes=[256, 64, 32],
        maneuver_ids=["MAN_0001"],
        seed=0,
        reward_version="v0",
        checkpoint_git_sha="abc123",
        global_env_step=100,
        ppo_update_step=5,
        reward_config_path="configs/reward.yaml",
        ppo_config_path="configs/ppo/train.yaml",
    )

    manifest = build_manifest(
        restored=restored,
        run_id="test_run",
        evaluated_maneuver_ids=["MAN_0001"],
        policy_mode="deterministic",
        policy_seed=None,
        downstream_mode="frenet_mpc",
        max_episode_steps=100,
        selected_episodes={"success": ["MAN_0001"], "collision": [], "timeout": []},
        scope="checkpoint",
        render_policy={"mode": "representative_per_outcome", "max_per_outcome": 5},
    )

    for key in [
        "checkpoint_path", "run_id", "checkpoint_git_sha", "current_git_sha",
        "reward_version", "seed", "ppo_update_step", "global_env_step",
        "network_hidden_sizes", "evaluated_maneuver_ids", "policy_mode",
        "downstream_mode", "max_episode_steps", "selected_episodes", "render_policy",
    ]:
        assert key in manifest

    assert manifest["downstream_mode"] == "frenet_mpc"
    assert manifest["evaluation_kind"] == "POST-TRAINING FINAL-CHECKPOINT EVALUATION"
    # selected_episodes must be list-valued per category, never a scalar.
    assert isinstance(manifest["selected_episodes"]["success"], list)
    assert manifest["render_policy"]["mode"] == "representative_per_outcome"
    assert manifest["render_policy"]["max_per_outcome"] == 5


def test_manifest_render_policy_modes():
    restored = RestoredPPOCheckpoint(
        checkpoint_path="outputs/checkpoints/fake.pkl",
        payload=None, policy=None, value_network=None, value_params=None,
        network_hidden_sizes=[256, 64, 32], maneuver_ids=["MAN_0001"], seed=0,
        reward_version="v0", checkpoint_git_sha="abc123", global_env_step=100,
        ppo_update_step=5, reward_config_path="configs/reward.yaml",
        ppo_config_path="configs/ppo/train.yaml",
    )

    for render_policy in (
        {"mode": "all"},
        {"mode": "scan_only"},
        {"mode": "representative_per_outcome", "max_per_outcome": 5},
    ):
        manifest = build_manifest(
            restored=restored, run_id="test_run", evaluated_maneuver_ids=["MAN_0001"],
            policy_mode="deterministic", policy_seed=None, downstream_mode="frenet_mpc",
            max_episode_steps=100, selected_episodes={"success": ["MAN_0001"]},
            scope="checkpoint", render_policy=render_policy,
        )
        assert manifest["render_policy"] == render_policy
