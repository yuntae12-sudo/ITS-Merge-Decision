"""Tests for src/environment/observation_builder.py using synthetic
geometry (no WOMD fixture dependency, matching the style of
tests/scenarios/test_scenario_features.py).
"""

import numpy as np
import pytest

from src.environment.observation_builder import (
    OBSERVATION_DIM,
    OBSERVATION_FIELD_NAMES,
    TTC_CAP_S,
    ObservationInputs,
    build_observation,
)
from src.scenarios.lane_geometry import LanePolyline, compute_arc_length
from src.scenarios.scenario_features import AgentSelectionConfig

DEFAULT_CONFIG = AgentSelectionConfig(
    max_target_lane_lateral_distance_m=5.0,
    max_target_lane_heading_difference_deg=45.0,
    max_distance_m=100.0,
    density_radius_m=50.0,
)


def _make_straight_lane(lane_id, length_m=100.0, y_offset=0.0):
    xs = np.linspace(0.0, length_m, int(length_m) + 1)
    ys = np.full_like(xs, y_offset)
    xy = np.stack([xs, ys], axis=1)
    return LanePolyline(
        lane_id=lane_id,
        lane_type=2,
        xy=xy,
        direction=np.tile([1.0, 0.0], (xy.shape[0], 1)),
        arc_length=compute_arc_length(xy),
    )


def _agents(entries):
    if not entries:
        empty = np.array([])
        return (empty,) * 9
    fields = list(zip(*entries))
    return tuple(np.asarray(f) for f in fields)


def _base_inputs(**overrides):
    source = _make_straight_lane(lane_id=10, y_offset=0.0)
    target = _make_straight_lane(lane_id=20, y_offset=0.0)

    (object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid) = (
        overrides.pop("agents", _agents([]))
    )

    defaults = dict(
        ego_id=1,
        ego_x=50.0,
        ego_y=0.0,
        ego_vel_x=15.0,
        ego_vel_y=0.0,
        ego_length_m=4.5,
        source_polyline=source,
        target_polyline=target,
        merge_end_s=90.0,
        ego_source_arc_length_m=50.0,
        object_ids=object_ids,
        object_types=object_types,
        valid=valid,
        x=x,
        y=y,
        yaw=yaw,
        vel_x=vel_x,
        vel_y=vel_y,
        length=length,
        agent_selection_config=DEFAULT_CONFIG,
    )
    defaults.update(overrides)
    return ObservationInputs(**defaults)


def test_observation_shape_and_field_count():

    inputs = _base_inputs()
    observation = build_observation(inputs)

    assert observation.shape == (14,)
    assert observation.shape == (OBSERVATION_DIM,)
    assert len(OBSERVATION_FIELD_NAMES) == OBSERVATION_DIM


def test_observation_field_order_matches_spec():

    assert OBSERVATION_FIELD_NAMES == (
        "v_e",
        "d_m",
        "target_front_present",
        "target_front_gap",
        "target_front_relative_speed",
        "target_front_ttc",
        "target_rear_present",
        "target_rear_gap",
        "target_rear_relative_speed",
        "target_rear_ttc",
        "source_front_present",
        "source_front_gap",
        "source_front_relative_speed",
        "source_front_ttc",
    )


def test_observation_is_always_finite_no_vehicles():

    inputs = _base_inputs(agents=_agents([]))
    observation = build_observation(inputs)

    assert np.all(np.isfinite(observation))
    # presence flags all 0, gaps/relative-speeds 0.0, TTC capped.
    assert observation[2] == 0.0  # target_front_present
    assert observation[3] == 0.0  # target_front_gap
    assert observation[4] == 0.0  # target_front_relative_speed
    assert observation[5] == pytest.approx(TTC_CAP_S)
    assert observation[6] == 0.0  # target_rear_present
    assert observation[10] == 0.0  # source_front_present
    assert observation[13] == pytest.approx(TTC_CAP_S)


def test_scalar_ego_speed_not_lane_projected():
    """v_e must be sqrt(vx^2+vy^2), NOT a lane-tangent projection --
    Stage B-0 Section 2's fix. Using a lateral-only velocity proves
    the two definitions would disagree if the wrong one were used:
    the lane-tangent projection of a purely-lateral velocity is 0,
    but the scalar speed is nonzero.
    """

    inputs = _base_inputs(
        ego_vel_x=0.0, ego_vel_y=7.0, agents=_agents([])
    )
    observation = build_observation(inputs)

    assert observation[0] == pytest.approx(7.0, abs=1e-6)


def test_d_m_uses_source_lane_arc_length():

    inputs = _base_inputs(
        merge_end_s=90.0, ego_source_arc_length_m=60.0, agents=_agents([])
    )
    observation = build_observation(inputs)

    assert observation[1] == pytest.approx(30.0, abs=1e-6)


def test_d_m_none_source_arc_length_yields_zero():

    inputs = _base_inputs(
        merge_end_s=90.0, ego_source_arc_length_m=None, agents=_agents([])
    )
    observation = build_observation(inputs)

    assert observation[1] == pytest.approx(0.0, abs=1e-6)


def test_target_front_present_and_values():

    agents = _agents(
        [
            (2, 70.0, 0.0, 0.0, 10.0, 0.0, 4.5, 1, True),  # target front
        ]
    )
    inputs = _base_inputs(agents=agents)
    observation = build_observation(inputs)

    assert observation[2] == 1.0  # target_front_present
    assert observation[3] == pytest.approx(70.0 - 50.0 - 4.5, abs=1e-3)
    assert observation[5] < TTC_CAP_S or observation[5] == pytest.approx(
        TTC_CAP_S
    )  # capped or below cap either way, never inf


def test_source_front_present_independent_of_target():
    """A vehicle only on the SOURCE lane must populate
    source_front_* without affecting target_front_* (they are
    computed by two independent extract_interaction_features calls,
    Stage B-0 Section 1)."""

    source = _make_straight_lane(lane_id=10, y_offset=0.0)
    target = _make_straight_lane(lane_id=20, y_offset=50.0)  # far away laterally

    agents = _agents(
        [
            (2, 70.0, 0.0, 0.0, 10.0, 0.0, 4.5, 1, True),  # only near source lane
        ]
    )
    inputs = _base_inputs(
        source_polyline=source, target_polyline=target, agents=agents
    )
    observation = build_observation(inputs)

    assert observation[10] == 1.0  # source_front_present
    assert observation[2] == 0.0  # target_front_present (vehicle too far laterally)


def test_ttc_never_infinite_in_observation():

    # Non-closing front (front faster than ego) -> raw TTC is inf.
    agents = _agents(
        [
            (2, 70.0, 0.0, 0.0, 25.0, 0.0, 4.5, 1, True),  # front, faster than ego
        ]
    )
    inputs = _base_inputs(ego_vel_x=15.0, agents=agents)
    observation = build_observation(inputs)

    assert observation[2] == 1.0
    assert observation[5] == pytest.approx(TTC_CAP_S)
    assert np.isfinite(observation[5])


def test_rear_ordering_fix_reflected_in_observation():
    """End-to-end proof that the Stage B-0 geometry fix (Section 1)
    is actually wired into the observation builder: ego positioned
    before the target lane's start, with a real rear vehicle also
    before the start -- must be detected, not silently dropped."""

    source = _make_straight_lane(lane_id=10, y_offset=0.0)
    target = _make_straight_lane(lane_id=20, y_offset=0.0)

    agents = _agents(
        [
            (2, -4.0, 0.0, 0.0, 20.0, 0.0, 4.5, 1, True),  # rear, before target start
        ]
    )
    inputs = _base_inputs(
        ego_x=-3.0,
        ego_y=0.0,
        source_polyline=source,
        target_polyline=target,
        ego_source_arc_length_m=None,
        agents=agents,
    )
    observation = build_observation(inputs)

    assert observation[6] == 1.0  # target_rear_present
    assert np.all(np.isfinite(observation))
