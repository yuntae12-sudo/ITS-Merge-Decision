"""Stage B-1.5 Section A4 regression tests for
src/environment/merge_reference.py's degenerate-blend-window fix.

Synthetic straight-line polylines (no real WOMD data needed -- this
tests pure geometry, matching the convention used by
tests/scenarios/test_lane_geometry.py).

Two real-data bugs motivated this fix, both traced directly during
Stage B-1.5:
  1. Stage B-1's original bug: a short TARGET lane made later blend
     samples clamp to nearly the same terminal point (fixed then).
  2. Stage B-1.5's bug: whenever ego's remaining SOURCE-lane distance
     shrank near zero (at reset or mid-episode), the SOURCE-side blend
     sample collapsed the same way. An intermediate fix (windowed
     sampling along the target lane's own arc length) reintroduced the
     identical degeneracy once ego traveled past the TARGET lane's own
     end too (found on a real chained maneuver, MAN_0062, where ego
     cruises well past the transition boundary before advancing). The
     final fix removes the bounded window entirely below threshold:
     it hands back the target polyline directly.
"""

import numpy as np

from src.scenarios.lane_geometry import LanePolyline, compute_arc_length
from src.environment.merge_reference import (
    MIN_SOURCE_BLEND_WINDOW_M,
    build_merge_reference,
)


def _straight_polyline(lane_id, start_xy, end_xy, num_points=10):
    start = np.array(start_xy, dtype=np.float64)
    end = np.array(end_xy, dtype=np.float64)
    t = np.linspace(0.0, 1.0, num_points)[:, None]
    xy = start[None, :] + t * (end - start)[None, :]
    diffs = np.diff(xy, axis=0)
    unit = diffs / np.linalg.norm(diffs, axis=1, keepdims=True)
    direction = np.vstack([unit, unit[-1:]])
    return LanePolyline(
        lane_id=lane_id,
        lane_type=1,
        xy=xy,
        direction=direction,
        arc_length=compute_arc_length(xy),
    )


SOURCE = _straight_polyline(1, (0.0, 0.0), (20.0, 0.0))
SHORT_TARGET = _straight_polyline(2, (0.0, 3.0), (12.0, 3.0))
LONG_TARGET = _straight_polyline(2, (0.0, 3.0), (100.0, 3.0))


def test_mid_source_blend_produces_smoothly_varying_points():
    """Ample source_remaining_m -> normal blended window, no
    degenerate/duplicate points."""

    reference = build_merge_reference(SOURCE, LONG_TARGET, ego_x=2.0, ego_y=0.0)
    xy = reference.polyline.xy
    step_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    assert np.all(step_lengths > 1e-6), "blend produced near-duplicate points"


def test_near_source_end_returns_target_polyline_directly():
    """Ego within MIN_SOURCE_BLEND_WINDOW_M of the source lane's own
    end -> reference IS the target polyline (no bounded sample window
    that could itself degenerate)."""

    ego_x = SOURCE.arc_length[-1] - (MIN_SOURCE_BLEND_WINDOW_M / 2.0)
    reference = build_merge_reference(SOURCE, LONG_TARGET, ego_x=ego_x, ego_y=0.0)
    assert reference.polyline is LONG_TARGET


def test_ego_past_source_end_returns_target_polyline_directly():
    ego_x = SOURCE.arc_length[-1] + 5.0  # past the source's own reconstructed end
    reference = build_merge_reference(SOURCE, LONG_TARGET, ego_x=ego_x, ego_y=0.0)
    assert reference.polyline is LONG_TARGET


def test_ego_far_past_short_target_end_still_returns_target_polyline():
    """The originally-failing scenario (MAN_0062): ego has traveled far
    along a SHORT target lane, well past its own reconstructed end.
    The fix must not attempt any bounded target-side sampling here --
    it must still just be the (short) target polyline itself, which
    the low-level controller's own clamped projection then tracks
    (clamping to the target's real endpoint, not a synthetic
    extrapolation) rather than a separately-degenerating window."""

    ego_x = SOURCE.arc_length[-1] - 1.0  # inside the near-end threshold
    reference = build_merge_reference(SOURCE, SHORT_TARGET, ego_x=ego_x, ego_y=0.0)
    assert reference.polyline is SHORT_TARGET
    # The short target's own geometry is unmodified (no degenerate
    # collapse introduced by this function).
    assert reference.polyline.arc_length[-1] == SHORT_TARGET.arc_length[-1]


def test_blend_window_is_capped_by_short_target_length():
    """Mid-source blend (not yet near the end) but the TARGET lane
    itself is short -- the blend window must not exceed the target's
    own total length (Stage B-1's original fix, still covered)."""

    reference = build_merge_reference(SOURCE, SHORT_TARGET, ego_x=2.0, ego_y=0.0)
    total_blend_span = reference.polyline.arc_length[-1]
    assert total_blend_span <= SHORT_TARGET.arc_length[-1] + 1e-6
