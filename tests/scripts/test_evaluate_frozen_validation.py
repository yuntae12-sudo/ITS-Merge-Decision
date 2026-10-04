import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evaluate_frozen_validation.py"
SPEC = importlib.util.spec_from_file_location("evaluate_frozen_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(mid, outcome, actions=(1, 1, 1, 1), ret=0.0):
    keep, follow, merge, stop = actions
    return {
        "checkpoint": 12, "checkpoint_path": "outputs/checkpoints/full_seed0_step000012.pkl",
        "maneuver_id": mid, "outcome": outcome, "termination_reason": "",
        "episode_steps": 10, "policy_decision_count": sum(actions), "episode_return": ret,
        "reward_terminal": 0, "reward_decision_cost": 0, "reward_safety": 0,
        "reward_progress": 0, "reward_decision": 0, "keep_count": keep,
        "follow_count": follow, "merge_count": merge, "stop_count": stop,
        "intervention_count": 2, "collision_blocked_count": 1,
        "planner_infeasible_count": 1, "intervention_rate": .2,
        "collision_blocked_rate": .1, "planner_infeasible_rate": .1,
        "merge_commit_step": "", "exception_repr": "",
    }


def test_canonical_validation_is_exactly_388_unique_and_disjoint():
    validation = sorted(MODULE.load_decision_dataset_maneuver_specs("validation"), key=lambda x: x.maneuver_id)
    train = MODULE.load_decision_dataset_maneuver_specs("train")
    manifest = MODULE.build_validation_manifest(validation, train)
    assert manifest["num_maneuvers"] == 388
    assert len(set(manifest["maneuver_ids"])) == 388
    assert manifest["maneuver_ids"] == sorted(manifest["maneuver_ids"])
    assert manifest["train_validation_maneuver_id_overlap"] == 0
    assert manifest == MODULE.build_validation_manifest(validation, train)


def test_resume_requires_exact_ordered_prefix():
    ids = ["a", "b", "c"]
    MODULE.validate_resume_prefix([_row("a", "success"), _row("b", "timeout")], ids)
    with pytest.raises(ValueError, match="ordered prefix"):
        MODULE.validate_resume_prefix([_row("b", "timeout")], ids)


def test_summary_consistency_median_and_action_ratios(monkeypatch):
    monkeypatch.setattr(MODULE, "EXPECTED_VALIDATION_COUNT", 4)
    rows = [_row("a", "success", (2, 0, 0, 0), 2),
            _row("b", "collision", (0, 2, 0, 0), -2),
            _row("c", "offroad", (0, 0, 2, 0), 1),
            _row("d", "timeout", (0, 0, 0, 2), -1)]
    summary = MODULE.aggregate_rows(rows, "x.pkl")
    assert sum(summary[f"{key}_count"] for key in MODULE.OUTCOMES) == 4
    assert summary["median_episode_return"] == pytest.approx(0)
    assert sum(summary[f"{key.lower()}_ratio"] for key in MODULE.ACTION_NAMES) == pytest.approx(1)


def test_representative_selector_is_deterministic_and_capped():
    rows = [_row(f"m{i:02d}", "success") for i in range(7)] + [_row("z", "timeout")]
    first = MODULE.representative_selection(rows)
    second = MODULE.representative_selection(rows)
    assert first == second
    assert first["selection"]["success"]["maneuver_ids"] == [f"m{i:02d}" for i in range(5)]
    assert first["selection"]["timeout"]["maneuver_ids"] == ["z"]


def test_freeze_constants_are_update12_train_only():
    assert MODULE.SELECTED_UPDATE == 12
    assert MODULE.TRAIN64_REFERENCE["success_rate"] == 0.3125
    assert MODULE.DEFAULT_VALIDATION_DATASET_CONFIG == "configs/dataset_validation.yaml"
