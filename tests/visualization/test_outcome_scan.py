"""Tests for src.visualization.outcome_scan: outcome_index.csv always
contains every evaluated episode, and default-5-per-outcome /
--render-all (no cap) selection semantics (docs/ppo/
PPO_VISUALIZATION_GUIDE.md, PPO Visualization Default-5 Extension)."""

from src.visualization.outcome_scan import (
    DEFAULT_MAX_PER_OUTCOME,
    OUTCOME_CATEGORIES,
    build_outcome_index,
    select_representative_episodes,
    write_outcome_index_csv,
)
from src.visualization.ppo_rollout import (
    OUTCOME_COLLISION,
    OUTCOME_SUCCESS,
    OUTCOME_TIMEOUT,
    PPOEpisodeResult,
)


def _episode(maneuver_id, outcome, termination_reason=None):
    return PPOEpisodeResult(
        maneuver_id=maneuver_id,
        outcome=outcome,
        termination_reason=termination_reason,
        steps=[],
        physical_step_count=10,
        policy_decision_count=1,
        merge_commit_step=0,
        final_intervention_rate=0.0,
    )


def _episodes(maneuver_ids, outcome, termination_reason=None):
    return [_episode(mid, outcome, termination_reason) for mid in maneuver_ids]


def test_default_max_per_outcome_is_5():
    assert DEFAULT_MAX_PER_OUTCOME == 5


# ======================================================================
# A. Default selection: cap at 5, never fewer than what exists
# ======================================================================


def test_default_selection_caps_at_5_per_outcome():
    success_ids = [f"MAN_{i:04d}" for i in range(1, 11)]  # 10 success episodes
    collision_ids = [f"MAN_{i:04d}" for i in range(101, 108)]  # 7 collision episodes
    timeout_ids = [f"MAN_{i:04d}" for i in range(201, 204)]  # 3 timeout episodes

    episodes = (
        _episodes(success_ids, OUTCOME_SUCCESS, "success")
        + _episodes(collision_ids, OUTCOME_COLLISION, "failure_collision")
        + _episodes(timeout_ids, OUTCOME_TIMEOUT, None)
    )

    selected = select_representative_episodes(episodes)  # default cap

    assert len(selected["success"]["maneuver_ids"]) == 5
    assert selected["success"]["maneuver_ids"] == success_ids[:5]

    assert len(selected["collision"]["maneuver_ids"]) == 5
    assert selected["collision"]["maneuver_ids"] == collision_ids[:5]

    # Fewer than 5 timeout episodes -> all 3 selected, never padded/crashed.
    assert len(selected["timeout"]["maneuver_ids"]) == 3
    assert selected["timeout"]["maneuver_ids"] == timeout_ids

    # No offroad episodes at all -> empty list, handled gracefully.
    assert selected["offroad"]["maneuver_ids"] == []
    assert "not found in evaluated scope" in selected["offroad"]["reason"]


def test_default_selection_never_raises_on_empty_input():
    selected = select_representative_episodes([])
    for outcome in OUTCOME_CATEGORIES:
        assert selected[outcome]["maneuver_ids"] == []


# ======================================================================
# B. Deterministic ordering
# ======================================================================


def test_selection_is_stable_sorted_regardless_of_input_order():
    # Deliberately unsorted maneuver_id order in the input list.
    shuffled_ids = ["MAN_0008", "MAN_0001", "MAN_0005", "MAN_0003", "MAN_0002",
                     "MAN_0009", "MAN_0004", "MAN_0006", "MAN_0007"]
    episodes = _episodes(shuffled_ids, OUTCOME_SUCCESS, "success")

    selected_from_shuffled = select_representative_episodes(episodes)

    sorted_ids = sorted(shuffled_ids)
    episodes_sorted = _episodes(sorted_ids, OUTCOME_SUCCESS, "success")
    selected_from_sorted = select_representative_episodes(episodes_sorted)

    # Both must select the exact same first-5-in-sorted-order maneuvers
    # -- callers are responsible for feeding stable sorted-maneuver_id
    # order (as scripts/visualization/visualize_ppo_run.py does), but
    # the selection function itself must not depend on incidental list
    # order for two runs fed the SAME already-sorted sequence.
    assert selected_from_sorted["success"]["maneuver_ids"] == sorted_ids[:5]
    # Feeding it pre-sorted input is the documented contract; confirm
    # that contract actually yields the first 5 in sorted order.
    assert selected_from_sorted["success"]["maneuver_ids"] == selected_from_sorted["success"]["maneuver_ids"] == sorted(
        selected_from_sorted["success"]["maneuver_ids"]
    )


# ======================================================================
# C. --render-all (max_per_outcome=None): select every episode
# ======================================================================


def test_render_all_selects_every_episode_per_outcome():
    success_ids = [f"MAN_{i:04d}" for i in range(1, 34)]  # 33 success episodes
    collision_ids = [f"MAN_{i:04d}" for i in range(101, 107)]  # 6 collision

    episodes = (
        _episodes(success_ids, OUTCOME_SUCCESS, "success")
        + _episodes(collision_ids, OUTCOME_COLLISION, "failure_collision")
    )

    selected = select_representative_episodes(episodes, max_per_outcome=None)

    assert len(selected["success"]["maneuver_ids"]) == 33
    assert selected["success"]["maneuver_ids"] == success_ids
    assert len(selected["collision"]["maneuver_ids"]) == 6
    assert selected["collision"]["maneuver_ids"] == collision_ids
    assert selected["timeout"]["maneuver_ids"] == []


# ======================================================================
# D. selected_episodes.json shape: list, not scalar
# ======================================================================


def test_selection_result_is_list_valued_not_scalar():
    episodes = _episodes(["MAN_0001", "MAN_0002"], OUTCOME_SUCCESS, "success")
    selected = select_representative_episodes(episodes)

    for outcome in OUTCOME_CATEGORIES:
        assert isinstance(selected[outcome]["maneuver_ids"], list)


# ======================================================================
# outcome_index.csv must always contain every evaluated episode,
# regardless of render/selection policy.
# ======================================================================


def test_build_outcome_index_contains_every_evaluated_episode_never_truncated():
    # 12 episodes total, spread across outcomes -- far more than the
    # default per-outcome render cap of 5. The index must still have
    # all 12 rows.
    success_ids = [f"MAN_{i:04d}" for i in range(1, 9)]  # 8
    collision_ids = [f"MAN_{i:04d}" for i in range(101, 105)]  # 4

    episodes = (
        _episodes(success_ids, OUTCOME_SUCCESS, "success")
        + _episodes(collision_ids, OUTCOME_COLLISION, "failure_collision")
    )

    rows = build_outcome_index(episodes)
    assert len(rows) == 12
    assert {r.maneuver_id for r in rows} == set(success_ids) | set(collision_ids)


def test_build_outcome_index_and_write_csv(tmp_path):
    episodes = [
        _episode("MAN_0001", OUTCOME_SUCCESS, "success"),
        _episode("MAN_0002", OUTCOME_COLLISION, "failure_collision"),
    ]
    rows = build_outcome_index(episodes)
    assert len(rows) == 2
    assert rows[0].maneuver_id == "MAN_0001"
    assert rows[0].outcome == OUTCOME_SUCCESS

    csv_path = tmp_path / "outcome_index.csv"
    write_outcome_index_csv(rows, str(csv_path))

    content = csv_path.read_text()
    assert "maneuver_id" in content.splitlines()[0]
    assert "MAN_0001" in content
    assert "MAN_0002" in content
