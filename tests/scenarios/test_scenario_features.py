"""Tests for src/scenarios/scenario_features.py using synthetic geometry.

Covers Front/Rear ordering, bumper-to-bumper gap, relative-speed
conventions, closing/non-closing TTC, no-front/no-rear handling, and
d_m pass-through (project plan Commit D section 24).
"""

import numpy as np
import pytest

from src.scenarios.lane_geometry import LanePolyline, compute_arc_length
from src.scenarios.scenario_features import (
    AgentSelectionConfig,
    extract_interaction_features,
    find_target_lane_front_rear,
)

DEFAULT_CONFIG = AgentSelectionConfig(
    max_target_lane_lateral_distance_m=5.0,
    max_distance_m=100.0,
    density_radius_m=50.0,
)


def _make_straight_target(lane_id=1, length_m=100.0):
    xs = np.linspace(0.0, length_m, int(length_m) + 1)
    xy = np.stack([xs, np.zeros_like(xs)], axis=1)
    return LanePolyline(
        lane_id=lane_id,
        lane_type=2,
        xy=xy,
        direction=np.tile([1.0, 0.0], (xy.shape[0], 1)),
        arc_length=compute_arc_length(xy),
    )


def _agents(entries):
    """entries: list of (id, x, y, vel_x, vel_y, length, type, valid)."""

    if not entries:
        empty = np.array([])
        return empty, empty, empty, empty, empty, empty, empty, empty

    fields = list(zip(*entries))
    return tuple(np.asarray(f) for f in fields)


def test_front_rear_ordering():

    target = _make_straight_target()

    object_ids, x, y, vel_x, vel_y, length, object_types, valid = _agents(
        [
            (2, 30.0, 0.0, 25.0, 0.0, 4.5, 1, True),  # rear
            (3, 70.0, 0.0, 15.0, 0.0, 4.5, 1, True),  # front
        ]
    )

    front_id, front_s, rear_id, rear_s = find_target_lane_front_rear(
        target,
        ego_s_m=50.0,
        ego_id=1,
        frame_index=0,
        object_ids=object_ids,
        object_types=object_types,
        valid=valid,
        x=x,
        y=y,
        config=DEFAULT_CONFIG,
    )

    assert front_id == 3
    assert rear_id == 2
    assert front_s == pytest.approx(70.0)
    assert rear_s == pytest.approx(30.0)


def test_bumper_to_bumper_gap_exact():

    target = _make_straight_target()

    object_ids, x, y, vel_x, vel_y, length, object_types, valid = _agents(
        [
            (2, 30.0, 0.0, 25.0, 0.0, 4.0, 1, True),
            (3, 70.0, 0.0, 15.0, 0.0, 6.0, 1, True),
        ]
    )

    features = extract_interaction_features(
        frame_index=0,
        target_polyline=target,
        merge_distance_m=10.0,
        ego_id=1,
        ego_x=50.0,
        ego_y=0.0,
        ego_vel_x=20.0,
        ego_vel_y=0.0,
        ego_length_m=5.0,
        object_ids=object_ids,
        object_types=object_types,
        valid=valid,
        x=x,
        y=y,
        vel_x=vel_x,
        vel_y=vel_y,
        length=length,
        config=DEFAULT_CONFIG,
    )

    # front_gap = (70 - 50) - (6.0 + 5.0)/2 = 20 - 5.5 = 14.5
    assert features.front_gap_m == pytest.approx(14.5)
    # rear_gap = (50 - 30) - (4.0 + 5.0)/2 = 20 - 4.5 = 15.5
    assert features.rear_gap_m == pytest.approx(15.5)


def test_relative_speed_convention():

    target = _make_straight_target()

    object_ids, x, y, vel_x, vel_y, length, object_types, valid = _agents(
        [
            (2, 30.0, 0.0, 25.0, 0.0, 4.5, 1, True),  # rear, faster than ego
            (3, 70.0, 0.0, 15.0, 0.0, 4.5, 1, True),  # front, slower than ego
        ]
    )

    features = extract_interaction_features(
        frame_index=0,
        target_polyline=target,
        merge_distance_m=10.0,
        ego_id=1,
        ego_x=50.0,
        ego_y=0.0,
        ego_vel_x=20.0,
        ego_vel_y=0.0,
        ego_length_m=4.5,
        object_ids=object_ids,
        object_types=object_types,
        valid=valid,
        x=x,
        y=y,
        vel_x=vel_x,
        vel_y=vel_y,
        length=length,
        config=DEFAULT_CONFIG,
    )

    # Delta v_f = v_ego - v_front = 20 - 15 = +5 (closing)
    assert features.front_relative_speed_mps == pytest.approx(5.0)
    # Delta v_r = v_rear - v_ego = 25 - 20 = +5 (closing)
    assert features.rear_relative_speed_mps == pytest.approx(5.0)


def test_non_closing_ttc_is_inf():
    """Ego slower than front (not closing) -> TTC_f = inf."""

    target = _make_straight_target()

    object_ids, x, y, vel_x, vel_y, length, object_types, valid = _agents(
        [
            (3, 70.0, 0.0, 25.0, 0.0, 4.5, 1, True),  # front, faster than ego
        ]
    )

    features = extract_interaction_features(
        frame_index=0,
        target_polyline=target,
        merge_distance_m=10.0,
        ego_id=1,
        ego_x=50.0,
        ego_y=0.0,
        ego_vel_x=15.0,
        ego_vel_y=0.0,
        ego_length_m=4.5,
        object_ids=object_ids,
        object_types=object_types,
        valid=valid,
        x=x,
        y=y,
        vel_x=vel_x,
        vel_y=vel_y,
        length=length,
        config=DEFAULT_CONFIG,
    )

    assert features.front_relative_speed_mps == pytest.approx(-10.0)
    assert features.front_ttc_s == float("inf")


def test_closing_ttc_exact():
    """gap = 20 m, closing = 5 m/s -> TTC = 4 s."""

    target = _make_straight_target()

    # front at s=70, ego at s=50, both length 0 (gap == center distance
    # for a clean, exactly-checkable number): gap = 20 - 0 = 20 m.
    object_ids, x, y, vel_x, vel_y, length, object_types, valid = _agents(
        [
            (3, 70.0, 0.0, 15.0, 0.0, 0.0, 1, True),
        ]
    )

    features = extract_interaction_features(
        frame_index=0,
        target_polyline=target,
        merge_distance_m=10.0,
        ego_id=1,
        ego_x=50.0,
        ego_y=0.0,
        ego_vel_x=20.0,
        ego_vel_y=0.0,
        ego_length_m=0.0,
        object_ids=object_ids,
        object_types=object_types,
        valid=valid,
        x=x,
        y=y,
        vel_x=vel_x,
        vel_y=vel_y,
        length=length,
        config=DEFAULT_CONFIG,
    )

    assert features.front_gap_m == pytest.approx(20.0)
    assert features.front_relative_speed_mps == pytest.approx(5.0)
    assert features.front_ttc_s == pytest.approx(4.0)


def test_no_front_no_rear_convention():

    target = _make_straight_target()

    object_ids, x, y, vel_x, vel_y, length, object_types, valid = _agents(
        [
            # A vehicle far outside max_distance_m, should not count.
            (2, 500.0, 0.0, 20.0, 0.0, 4.5, 1, True),
        ]
    )

    features = extract_interaction_features(
        frame_index=0,
        target_polyline=target,
        merge_distance_m=10.0,
        ego_id=1,
        ego_x=50.0,
        ego_y=0.0,
        ego_vel_x=20.0,
        ego_vel_y=0.0,
        ego_length_m=4.5,
        object_ids=object_ids,
        object_types=object_types,
        valid=valid,
        x=x,
        y=y,
        vel_x=vel_x,
        vel_y=vel_y,
        length=length,
        config=DEFAULT_CONFIG,
    )

    assert features.front_vehicle_id is None
    assert features.front_gap_m is None
    assert features.front_relative_speed_mps is None
    assert features.front_ttc_s == float("inf")

    assert features.rear_vehicle_id is None
    assert features.rear_gap_m is None
    assert features.rear_relative_speed_mps is None
    assert features.rear_ttc_s == float("inf")


def test_d_m_pass_through():

    target = _make_straight_target()

    object_ids, x, y, vel_x, vel_y, length, object_types, valid = _agents([])

    features = extract_interaction_features(
        frame_index=0,
        target_polyline=target,
        merge_distance_m=42.5,
        ego_id=1,
        ego_x=50.0,
        ego_y=0.0,
        ego_vel_x=20.0,
        ego_vel_y=0.0,
        ego_length_m=4.5,
        object_ids=object_ids,
        object_types=object_types,
        valid=valid,
        x=x,
        y=y,
        vel_x=vel_x,
        vel_y=vel_y,
        length=length,
        config=DEFAULT_CONFIG,
    )

    assert features.merge_distance_m == pytest.approx(42.5)


def test_overlapping_gap_not_clamped_and_ttc_zero():
    """Physical overlap (gap <= 0) is preserved, not clamped positive,
    and TTC is defined as 0.0 for that case.
    """

    target = _make_straight_target()

    # Front vehicle very close, large lengths -> negative gap.
    object_ids, x, y, vel_x, vel_y, length, object_types, valid = _agents(
        [
            (3, 51.0, 0.0, 15.0, 0.0, 6.0, 1, True),
        ]
    )

    features = extract_interaction_features(
        frame_index=0,
        target_polyline=target,
        merge_distance_m=10.0,
        ego_id=1,
        ego_x=50.0,
        ego_y=0.0,
        ego_vel_x=20.0,
        ego_vel_y=0.0,
        ego_length_m=6.0,
        object_ids=object_ids,
        object_types=object_types,
        valid=valid,
        x=x,
        y=y,
        vel_x=vel_x,
        vel_y=vel_y,
        length=length,
        config=DEFAULT_CONFIG,
    )

    # gap = (51-50) - (6+6)/2 = 1 - 6 = -5
    assert features.front_gap_m == pytest.approx(-5.0)
    assert features.front_ttc_s == pytest.approx(0.0)


def test_non_vehicle_type_excluded():

    target = _make_straight_target()

    object_ids, x, y, vel_x, vel_y, length, object_types, valid = _agents(
        [
            (2, 70.0, 0.0, 15.0, 0.0, 1.0, 2, True),  # pedestrian, excluded
        ]
    )

    features = extract_interaction_features(
        frame_index=0,
        target_polyline=target,
        merge_distance_m=10.0,
        ego_id=1,
        ego_x=50.0,
        ego_y=0.0,
        ego_vel_x=20.0,
        ego_vel_y=0.0,
        ego_length_m=4.5,
        object_ids=object_ids,
        object_types=object_types,
        valid=valid,
        x=x,
        y=y,
        vel_x=vel_x,
        vel_y=vel_y,
        length=length,
        config=DEFAULT_CONFIG,
    )

    assert features.front_vehicle_id is None
    assert features.traffic_density == 0
