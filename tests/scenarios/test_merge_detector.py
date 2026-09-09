"""Tests for src/scenarios/merge_detector.py using synthetic geometry.

Covers the minimum topology cases from the project plan (Commit D
section 23): true lane-ending merge, ordinary parallel lane change,
serial map-feature continuation, crossing lane, and weak/ambiguous
convergence.
"""

import numpy as np
import pytest

from src.scenarios.lane_assignment import LaneTransition
from src.scenarios.lane_geometry import LanePolyline, compute_arc_length
from src.scenarios.merge_detector import (
    REJECT_HEADING_MISMATCH,
    REJECT_INSUFFICIENT_CONVERGENCE,
    REJECT_PARALLEL_LANE_CHANGE,
    REJECT_SERIAL_CONTINUATION,
    MergeTopologyConfig,
    compute_merge_start_end_s,
    compute_remaining_merge_distance,
    detect_merge,
)

DEFAULT_CONFIG = MergeTopologyConfig(
    max_source_end_distance_m=15.0,
    max_endpoint_target_distance_m=5.0,
    max_heading_difference_deg=20.0,
    convergence_window_m=30.0,
    convergence_sample_count=7,
    min_separation_reduction_m=2.0,
    min_decreasing_fraction=0.6,
    serial_continuation_max_lateral_m=0.5,
    min_pre_merge_frames=5,
    min_target_lane_frames=5,
)


def _make_lane(lane_id, xy, lane_type=2):
    xy = np.asarray(xy, dtype=np.float64)
    direction = np.zeros_like(xy)
    diffs = np.diff(xy, axis=0)
    norms = np.hypot(diffs[:, 0], diffs[:, 1])
    norms[norms == 0] = 1.0
    unit = diffs / norms[:, None]
    direction[:-1] = unit
    direction[-1] = unit[-1]
    return LanePolyline(
        lane_id=lane_id,
        lane_type=lane_type,
        xy=xy,
        direction=direction,
        arc_length=compute_arc_length(xy),
    )


def _make_transition(source_id, target_id, transition_frame=10):
    return LaneTransition(
        source_lane_id=source_id,
        target_lane_id=target_id,
        transition_frame=transition_frame,
        source_start_frame=0,
        source_end_frame=transition_frame - 1,
        target_start_frame=transition_frame,
        target_end_frame=transition_frame + 9,
    )


def test_case1_true_lane_ending_merge_accepted():
    """Source curves toward and ends near a straight target; separation
    decreases monotonically over the approach.
    """

    xs = np.linspace(0.0, 30.0, 31)
    source_xy = np.stack([xs, 5.0 - (xs / 30.0) * 4.5], axis=1)
    source = _make_lane(1, source_xy)

    target_xs = np.linspace(-10.0, 40.0, 51)
    target_xy = np.stack([target_xs, np.zeros_like(target_xs)], axis=1)
    target = _make_lane(2, target_xy)

    transition = _make_transition(1, 2)

    diagnostic = detect_merge(
        transition, source, target, source.arc_length[-2], DEFAULT_CONFIG
    )

    assert diagnostic.is_merge_candidate is True
    assert diagnostic.reject_reason is None
    assert diagnostic.lanes_converge is True
    assert diagnostic.serial_continuation is False


def test_case2_ordinary_parallel_lane_change_rejected():
    """Source and target run parallel at a constant ~2 m separation --
    a normal lane change, not a merge (separation never meaningfully
    decreases).
    """

    xs = np.linspace(0.0, 30.0, 31)
    source_xy = np.stack([xs, np.full_like(xs, 2.0)], axis=1)
    source = _make_lane(1, source_xy)

    target_xy = np.stack([xs, np.zeros_like(xs)], axis=1)
    target = _make_lane(2, target_xy)

    transition = _make_transition(1, 2)

    diagnostic = detect_merge(
        transition, source, target, source.arc_length[-2], DEFAULT_CONFIG
    )

    assert diagnostic.is_merge_candidate is False
    assert diagnostic.reject_reason == REJECT_PARALLEL_LANE_CHANGE
    assert diagnostic.parallel_continuation is True


def test_case3_serial_map_feature_continuation_rejected():
    """Source A ends essentially exactly where target B starts, as
    consecutive serial segments of the same physical lane -- this must
    NOT be classified as a merge even though the endpoint is extremely
    close to the target (near-zero lateral offset, landing at the
    target's own start).
    """

    source_xy = np.stack(
        [np.linspace(0.0, 20.0, 21), np.zeros(21)], axis=1
    )
    source = _make_lane(1, source_xy)

    # Target begins exactly where source ends and continues collinearly.
    target_xy = np.stack(
        [np.linspace(20.0, 50.0, 31), np.zeros(31)], axis=1
    )
    target = _make_lane(2, target_xy)

    transition = _make_transition(1, 2)

    diagnostic = detect_merge(
        transition, source, target, source.arc_length[-2], DEFAULT_CONFIG
    )

    assert diagnostic.is_merge_candidate is False
    assert diagnostic.reject_reason == REJECT_SERIAL_CONTINUATION
    assert diagnostic.serial_continuation is True


def test_case4_crossing_lane_rejected():
    """Source ends physically near the target, but the target crosses
    at ~90 degrees -- heading incompatible, not a merge.
    """

    source_xy = np.stack(
        [np.linspace(0.0, 20.0, 21), np.full(21, 0.1)], axis=1
    )
    source = _make_lane(1, source_xy)

    # Target runs perpendicular (+y direction) through x=20.
    target_xy = np.stack(
        [np.full(21, 20.0), np.linspace(-10.0, 10.0, 21)], axis=1
    )
    target = _make_lane(2, target_xy)

    transition = _make_transition(1, 2)

    diagnostic = detect_merge(
        transition, source, target, source.arc_length[-2], DEFAULT_CONFIG
    )

    assert diagnostic.is_merge_candidate is False
    assert diagnostic.reject_reason == REJECT_HEADING_MISMATCH


def test_case5_weak_ambiguous_convergence_rejected():
    """Endpoint is close to the target and separation does trend
    downward overall, but the reduction is small -- below
    min_separation_reduction_m even though a majority of steps
    decrease. This models a real observed case (WOMD record 7,
    lane 256->257: reduction ~0.94 m, decreasing_fraction ~0.67) that
    is genuinely ambiguous rather than clearly parallel or clearly
    converging, and must not be accepted as a confirmed merge.
    """

    xs = np.linspace(0.0, 30.0, 31)
    # Separation decreases mostly monotonically by ~1.5 m total
    # (6.0 -> 4.5): more than half of min_separation_reduction_m=2.0
    # (so not "near-zero"/clearly parallel) but still short of the
    # 2.0 m required to confirm convergence, with one small uptick so
    # decreasing_fraction < 1.0 but still >= min_decreasing_fraction
    # =0.6.
    separation_profile = np.linspace(6.0, 4.5, 31)
    separation_profile[15] += 0.15  # one non-decreasing step
    source_xy = np.stack([xs, separation_profile], axis=1)
    source = _make_lane(1, source_xy)

    target_xy = np.stack([xs, np.zeros_like(xs)], axis=1)
    target = _make_lane(2, target_xy)

    transition = _make_transition(1, 2)

    diagnostic = detect_merge(
        transition, source, target, source.arc_length[-2], DEFAULT_CONFIG
    )

    assert diagnostic.is_merge_candidate is False
    assert diagnostic.reject_reason == REJECT_INSUFFICIENT_CONVERGENCE
    assert diagnostic.parallel_continuation is False


def test_merge_start_end_uses_source_arc_length():

    xs = np.linspace(0.0, 30.0, 31)
    source_xy = np.stack([xs, 5.0 - (xs / 30.0) * 4.5], axis=1)
    source = _make_lane(1, source_xy)

    target_xs = np.linspace(-10.0, 40.0, 51)
    target_xy = np.stack([target_xs, np.zeros_like(target_xs)], axis=1)
    target = _make_lane(2, target_xy)

    merge_start_s, merge_end_s = compute_merge_start_end_s(
        source, target, DEFAULT_CONFIG
    )

    assert merge_end_s == pytest.approx(source.arc_length[-1])
    assert merge_start_s < merge_end_s
    assert merge_start_s >= 0.0


def test_remaining_merge_distance_nonnegative_and_uses_source_arc_length():

    assert compute_remaining_merge_distance(30.0, 10.0) == pytest.approx(20.0)
    # Ego already past merge_end_s: clipped to 0, not negative.
    assert compute_remaining_merge_distance(30.0, 35.0) == pytest.approx(0.0)
    # No ego source-lane position available: documented as 0.0.
    assert compute_remaining_merge_distance(30.0, None) == pytest.approx(0.0)
