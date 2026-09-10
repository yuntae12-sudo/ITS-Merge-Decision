"""Tests for src/scenarios/scenario_features.py using synthetic geometry.

Covers Front/Rear ordering, bumper-to-bumper gap, relative-speed
conventions, closing/non-closing TTC, no-front/no-rear handling, d_m
pass-through (project plan Commit D section 24), target-lane heading
filtering (crossing / opposite-direction rejection, fix commit section
5), and target-lane longitudinal velocity projection (fix commit
section 6).
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
    max_target_lane_heading_difference_deg=45.0,
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
    """entries: list of (id, x, y, yaw, vel_x, vel_y, length, type, valid)."""

    if not entries:
        empty = np.array([])
        return (empty,) * 9

    fields = list(zip(*entries))
    return tuple(np.asarray(f) for f in fields)


def test_front_rear_ordering():

    target = _make_straight_target()

    (
        object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid,
    ) = _agents(
        [
            (2, 30.0, 0.0, 0.0, 25.0, 0.0, 4.5, 1, True),  # rear
            (3, 70.0, 0.0, 0.0, 15.0, 0.0, 4.5, 1, True),  # front
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
        yaw=yaw,
        config=DEFAULT_CONFIG,
    )

    assert front_id == 3
    assert rear_id == 2
    assert front_s == pytest.approx(70.0)
    assert rear_s == pytest.approx(30.0)


def test_aligned_target_lane_vehicle_retained():
    """A vehicle whose yaw matches the target lane's heading (aligned
    traffic) must be retained as a Front/Rear candidate.
    """

    target = _make_straight_target()

    (
        object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid,
    ) = _agents(
        [
            (3, 70.0, 0.0, 0.0, 15.0, 0.0, 4.5, 1, True),  # yaw=0, aligned
        ]
    )

    front_id, front_s, _, _ = find_target_lane_front_rear(
        target,
        ego_s_m=50.0,
        ego_id=1,
        frame_index=0,
        object_ids=object_ids,
        object_types=object_types,
        valid=valid,
        x=x,
        y=y,
        yaw=yaw,
        config=DEFAULT_CONFIG,
    )

    assert front_id == 3
    assert front_s == pytest.approx(70.0)


def test_crossing_vehicle_at_target_centerline_excluded():
    """A vehicle sitting exactly on the target centerline (lateral
    distance 0) but facing ~90 degrees off the target's heading (a
    crossing vehicle at an intersection) must be excluded, even though
    it would pass the lateral-distance filter alone (fix commit
    section 5).
    """

    target = _make_straight_target()

    (
        object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid,
    ) = _agents(
        [
            # On the centerline, but yaw=90 deg (crossing perpendicular).
            (3, 70.0, 0.0, np.pi / 2, 0.0, 15.0, 4.5, 1, True),
        ]
    )

    front_id, _, rear_id, _ = find_target_lane_front_rear(
        target,
        ego_s_m=50.0,
        ego_id=1,
        frame_index=0,
        object_ids=object_ids,
        object_types=object_types,
        valid=valid,
        x=x,
        y=y,
        yaw=yaw,
        config=DEFAULT_CONFIG,
    )

    assert front_id is None
    assert rear_id is None


def test_opposite_direction_vehicle_excluded():
    """A vehicle on the target centerline but facing the opposite
    direction (~180 degrees off) must be excluded (fix commit
    section 5).
    """

    target = _make_straight_target()

    (
        object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid,
    ) = _agents(
        [
            (3, 70.0, 0.0, np.pi, -15.0, 0.0, 4.5, 1, True),  # yaw=180 deg
        ]
    )

    front_id, _, rear_id, _ = find_target_lane_front_rear(
        target,
        ego_s_m=50.0,
        ego_id=1,
        frame_index=0,
        object_ids=object_ids,
        object_types=object_types,
        valid=valid,
        x=x,
        y=y,
        yaw=yaw,
        config=DEFAULT_CONFIG,
    )

    assert front_id is None
    assert rear_id is None


def test_bumper_to_bumper_gap_exact():

    target = _make_straight_target()

    (
        object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid,
    ) = _agents(
        [
            (2, 30.0, 0.0, 0.0, 25.0, 0.0, 4.0, 1, True),
            (3, 70.0, 0.0, 0.0, 15.0, 0.0, 6.0, 1, True),
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
        yaw=yaw,
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

    (
        object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid,
    ) = _agents(
        [
            (2, 30.0, 0.0, 0.0, 25.0, 0.0, 4.5, 1, True),  # rear, faster
            (3, 70.0, 0.0, 0.0, 15.0, 0.0, 4.5, 1, True),  # front, slower
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
        yaw=yaw,
        vel_x=vel_x,
        vel_y=vel_y,
        length=length,
        config=DEFAULT_CONFIG,
    )

    # Delta v_f = v_ego - v_front = 20 - 15 = +5 (closing)
    assert features.front_relative_speed_mps == pytest.approx(5.0)
    # Delta v_r = v_rear - v_ego = 25 - 20 = +5 (closing)
    assert features.rear_relative_speed_mps == pytest.approx(5.0)


def test_longitudinal_velocity_ignores_lateral_component():
    """Fix (project plan section 6): velocity magnitude
    hypot(vx, vy) must NOT be used. With target tangent = (1, 0) and
    vehicle velocity = (15, 8), the longitudinal speed must be exactly
    15 m/s, not hypot(15, 8) (~17.0 m/s).
    """

    target = _make_straight_target()

    (
        object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid,
    ) = _agents(
        [
            (3, 70.0, 0.0, 0.0, 15.0, 8.0, 4.5, 1, True),
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
        yaw=yaw,
        vel_x=vel_x,
        vel_y=vel_y,
        length=length,
        config=DEFAULT_CONFIG,
    )

    # Delta v_f = v_ego_long - v_front_long = 20 - 15 = 5, NOT
    # 20 - hypot(15, 8) (~2.97).
    assert features.front_relative_speed_mps == pytest.approx(5.0)
    assert not np.isclose(
        features.front_relative_speed_mps, 20.0 - np.hypot(15.0, 8.0)
    )


def test_ego_longitudinal_speed_ignores_lateral_component():
    """Same fix, applied to ego's own speed: ego velocity (12, 9)
    against target tangent (1, 0) must yield 12.0, not hypot(12, 9)
    (=15.0).
    """

    target = _make_straight_target()

    object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid = (
        _agents([])
    )

    features = extract_interaction_features(
        frame_index=0,
        target_polyline=target,
        merge_distance_m=10.0,
        ego_id=1,
        ego_x=50.0,
        ego_y=0.0,
        ego_vel_x=12.0,
        ego_vel_y=9.0,
        ego_length_m=4.5,
        object_ids=object_ids,
        object_types=object_types,
        valid=valid,
        x=x,
        y=y,
        yaw=yaw,
        vel_x=vel_x,
        vel_y=vel_y,
        length=length,
        config=DEFAULT_CONFIG,
    )

    assert features.ego_longitudinal_speed_mps == pytest.approx(12.0)
    assert not np.isclose(
        features.ego_longitudinal_speed_mps, np.hypot(12.0, 9.0)
    )


def test_non_closing_ttc_is_inf():
    """Ego slower than front (not closing) -> TTC_f = inf."""

    target = _make_straight_target()

    (
        object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid,
    ) = _agents(
        [
            (3, 70.0, 0.0, 0.0, 25.0, 0.0, 4.5, 1, True),  # front, faster
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
        yaw=yaw,
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
    (
        object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid,
    ) = _agents(
        [
            (3, 70.0, 0.0, 0.0, 15.0, 0.0, 0.0, 1, True),
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
        yaw=yaw,
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

    (
        object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid,
    ) = _agents(
        [
            # A vehicle far outside max_distance_m, should not count.
            (2, 500.0, 0.0, 0.0, 20.0, 0.0, 4.5, 1, True),
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
        yaw=yaw,
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

    object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid = (
        _agents([])
    )

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
        yaw=yaw,
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
    (
        object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid,
    ) = _agents(
        [
            (3, 51.0, 0.0, 0.0, 15.0, 0.0, 6.0, 1, True),
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
        yaw=yaw,
        vel_x=vel_x,
        vel_y=vel_y,
        length=length,
        config=DEFAULT_CONFIG,
    )

    # gap = (51-50) - (6+6)/2 = 1 - 6 = -5
    assert features.front_gap_m == pytest.approx(-5.0)
    assert features.front_ttc_s == pytest.approx(0.0)


def test_target_rear_ordering_before_polyline_start():
    """Stage B-0 fix: ego and a real rear vehicle both project before
    the target polyline's own start (a realistic early pre-merge
    scenario, per the Stage A/B-0 diagnostic finding that
    merge_start_frame consistently lands here). Both points sit
    exactly on the polyline's own extended centerline (y=0) so
    project_point_to_polyline's nearest-point lateral-distance
    measurement at the boundary segment stays within
    ``max_target_lane_lateral_distance_m`` for both -- isolating the
    longitudinal-ordering bug this fix targets from that unrelated
    boundary lateral-distance quirk.

    Before the fix, both ego (-3.0) and the rear candidate (-4.0)
    clamped to the identical arc_length_m=0.0, making relative_s
    exactly 0.0 for both -- neither > 0 nor < 0 -- so the rear vehicle
    was silently dropped. After the fix
    (project_point_to_polyline_signed), their true 1 m longitudinal
    separation is preserved and rear is found.
    """

    target = _make_straight_target()

    # ego 3 m before the target lane's start (x=0), on-centerline.
    ego_s_m = -3.0

    (
        object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid,
    ) = _agents(
        [
            (2, -4.0, 0.0, 0.0, 20.0, 0.0, 4.5, 1, True),  # true rear, 1m behind ego
            (3, 20.0, 0.0, 0.0, 15.0, 0.0, 4.5, 1, True),  # true front, well ahead
        ]
    )

    front_id, front_s, rear_id, rear_s = find_target_lane_front_rear(
        target,
        ego_s_m=ego_s_m,
        ego_id=1,
        frame_index=0,
        object_ids=object_ids,
        object_types=object_types,
        valid=valid,
        x=x,
        y=y,
        yaw=yaw,
        config=DEFAULT_CONFIG,
    )

    assert rear_id == 2
    assert rear_s == pytest.approx(-4.0, abs=1e-3)
    assert front_id == 3
    assert front_s == pytest.approx(20.0, abs=1e-3)


def test_ego_signed_projection_used_end_to_end():
    """extract_interaction_features must use the SIGNED target-lane
    projection for ego (not the clamped one) when computing
    front/rear gaps, so a pre-polyline-start ego still gets a
    physically correct bumper-to-bumper gap rather than one computed
    against an artificially-clamped ego_s=0."""

    target = _make_straight_target()

    (
        object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid,
    ) = _agents(
        [
            (2, -4.0, 0.0, 0.0, 20.0, 0.0, 4.5, 1, True),  # rear
        ]
    )

    features = extract_interaction_features(
        frame_index=0,
        target_polyline=target,
        merge_distance_m=0.0,
        ego_id=1,
        ego_x=-3.0,
        ego_y=0.0,
        ego_vel_x=10.0,
        ego_vel_y=0.0,
        ego_length_m=4.5,
        object_ids=object_ids,
        object_types=object_types,
        valid=valid,
        x=x,
        y=y,
        yaw=yaw,
        vel_x=vel_x,
        vel_y=vel_y,
        length=length,
        config=DEFAULT_CONFIG,
    )

    assert features.rear_vehicle_id == 2
    # bumper-to-bumper: (ego_s - rear_s) - 0.5*(4.5+4.5) = (-3 - -4) - 4.5 = -3.5
    assert features.rear_gap_m == pytest.approx(-3.5, abs=1e-3)


def test_non_vehicle_type_excluded():

    target = _make_straight_target()

    (
        object_ids, x, y, yaw, vel_x, vel_y, length, object_types, valid,
    ) = _agents(
        [
            (2, 70.0, 0.0, 0.0, 15.0, 0.0, 1.0, 2, True),  # pedestrian
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
        yaw=yaw,
        vel_x=vel_x,
        vel_y=vel_y,
        length=length,
        config=DEFAULT_CONFIG,
    )

    assert features.front_vehicle_id is None
    assert features.traffic_density == 0
