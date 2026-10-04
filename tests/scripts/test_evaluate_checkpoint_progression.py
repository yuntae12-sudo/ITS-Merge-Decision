import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evaluate_checkpoint_progression.py"
SPEC = importlib.util.spec_from_file_location("evaluate_checkpoint_progression", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(mid, outcome, *, checkpoint=1, actions=(1, 1, 1, 1), ret=0.0):
    keep, follow, merge, stop = actions
    return {
        "checkpoint": checkpoint, "checkpoint_path": "x.pkl", "maneuver_id": mid,
        "outcome": outcome, "termination_reason": "", "episode_steps": 10,
        "policy_decision_count": sum(actions), "episode_return": ret,
        "reward_terminal": 0.0, "reward_decision_cost": 0.0, "reward_safety": 0.0,
        "reward_progress": 0.0, "reward_decision": 0.0, "keep_count": keep,
        "follow_count": follow, "merge_count": merge, "stop_count": stop,
        "intervention_count": 2, "collision_blocked_count": 1,
        "planner_infeasible_count": 1, "intervention_rate": 0.2,
        "collision_blocked_rate": 0.1, "planner_infeasible_rate": 0.1,
        "merge_commit_step": "", "exception_repr": "",
    }


def test_fixed_manifest_is_reproducible(tmp_path):
    source = tmp_path / "ids.txt"
    ids = [f"m{i:02d}" for i in range(64)]
    source.write_text("\n".join(ids) + "\n")
    first = MODULE.build_fixed_manifest(source, MODULE.read_fixed_ids(source), 20260928)
    second = MODULE.build_fixed_manifest(source, MODULE.read_fixed_ids(source), 20260928)
    assert first == second
    assert first["num_maneuvers"] == 64


def test_same_64_maneuvers_required_for_every_checkpoint():
    ids = [f"m{i:02d}" for i in range(64)]
    rows = [_row(mid, "timeout", checkpoint=6) for mid in ids]
    MODULE.validate_raw_rows(rows, ids, 6)
    with pytest.raises(ValueError, match="differs"):
        MODULE.validate_raw_rows(rows[::-1], ids, 6)


def test_metric_aggregation_and_action_ratios():
    rows = [
        _row("a", "success", actions=(2, 1, 1, 0), ret=2.0),
        _row("b", "collision", actions=(0, 1, 1, 2), ret=-2.0),
    ]
    result = MODULE.aggregate_rows(1, "x.pkl", rows)
    assert result["success_count"] == 1
    assert result["collision_count"] == 1
    assert result["success_rate"] == pytest.approx(0.5)
    assert result["mean_episode_return"] == pytest.approx(0.0)
    assert result["intervention_rate"] == pytest.approx(0.2)
    assert sum(result[f"{name.lower()}_ratio"] for name in MODULE.ACTION_NAMES) == pytest.approx(1.0)


def test_outcome_count_equals_evaluated_maneuvers():
    rows = [_row("a", "success"), _row("b", "timeout"), _row("c", "offroad")]
    result = MODULE.aggregate_rows(1, "x.pkl", rows)
    assert sum(result[f"{outcome}_count"] for outcome in MODULE.OUTCOMES) == 3


def test_checkpoint_summary_csv_schema(tmp_path):
    summary = MODULE.aggregate_rows(1, "x.pkl", [_row("a", "success")])
    path = tmp_path / "summary.csv"
    MODULE.write_csv(path, [summary], MODULE.SUMMARY_FIELDS)
    rows = MODULE.read_csv(path)
    assert tuple(rows[0].keys()) == MODULE.SUMMARY_FIELDS


def test_paired_transition_classification():
    early = [_row("a", "collision"), _row("b", "success"), _row("c", "timeout")]
    final = [_row("a", "success"), _row("b", "collision"), _row("c", "timeout")]
    transitions = {row["maneuver_id"]: row["transition"] for row in MODULE.transition_rows(early, final)}
    assert transitions == {
        "a": "improved_to_success", "b": "regressed_from_success", "c": "unchanged"
    }
