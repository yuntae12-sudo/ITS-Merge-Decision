"""Tests for src/scenarios/validation_viz.py.

Prefers testing the pure helper functions directly (no matplotlib
figure needed). The renderer test builds a minimal synthetic
ScenarioRecord-like object rather than depending on real WOMD data.
"""

import types
from pathlib import Path

import numpy as np
import pytest

from src.scenarios.lane_geometry import LanePolyline, compute_arc_length
from src.scenarios.validation_viz import (
    compute_obb_corners,
    compute_viewport_bounds,
    render_candidate_figure,
)


def test_compute_obb_corners_axis_aligned():
    corners = compute_obb_corners(x=0.0, y=0.0, yaw=0.0, length=4.0, width=2.0)
    assert corners.shape == (4, 2)

    xs = corners[:, 0]
    ys = corners[:, 1]
    assert np.isclose(xs.max(), 2.0)
    assert np.isclose(xs.min(), -2.0)
    assert np.isclose(ys.max(), 1.0)
    assert np.isclose(ys.min(), -1.0)


def test_compute_obb_corners_rotated_quarter_turn():
    # yaw = pi/2: the "length" axis now points along +y, "width" along -x.
    corners = compute_obb_corners(x=0.0, y=0.0, yaw=np.pi / 2, length=4.0, width=2.0)
    xs = corners[:, 0]
    ys = corners[:, 1]
    assert np.isclose(xs.max(), 1.0, atol=1e-6)
    assert np.isclose(xs.min(), -1.0, atol=1e-6)
    assert np.isclose(ys.max(), 2.0, atol=1e-6)
    assert np.isclose(ys.min(), -2.0, atol=1e-6)


def test_compute_obb_corners_translated():
    corners = compute_obb_corners(x=10.0, y=-5.0, yaw=0.0, length=2.0, width=2.0)
    center = corners.mean(axis=0)
    assert np.isclose(center[0], 10.0)
    assert np.isclose(center[1], -5.0)


def test_compute_viewport_bounds_symmetric():
    (xmin, xmax), (ymin, ymax) = compute_viewport_bounds(100.0, -50.0, 25.0)
    assert xmin == pytest.approx(75.0)
    assert xmax == pytest.approx(125.0)
    assert ymin == pytest.approx(-75.0)
    assert ymax == pytest.approx(-25.0)
    assert (xmax - xmin) == pytest.approx(ymax - ymin)


def _make_lane(lane_id, xy):
    xy = np.asarray(xy, dtype=np.float64)
    direction = np.zeros_like(xy)
    diffs = np.diff(xy, axis=0)
    norms = np.hypot(diffs[:, 0], diffs[:, 1])
    norms[norms == 0] = 1.0
    unit = diffs / norms[:, None]
    direction[:-1] = unit
    direction[-1] = unit[-1]
    return LanePolyline(
        lane_id=lane_id, lane_type=2, xy=xy, direction=direction,
        arc_length=compute_arc_length(xy),
    )


def _make_fake_state(num_frames=20, num_objects=3):
    x = np.zeros((num_objects, num_frames))
    y = np.zeros((num_objects, num_frames))
    yaw = np.zeros((num_objects, num_frames))
    length = np.full((num_objects, num_frames), 4.5)
    width = np.full((num_objects, num_frames), 2.0)
    valid = np.ones((num_objects, num_frames), dtype=bool)

    # ego (index 0) moves along x.
    x[0] = np.linspace(0.0, 19.0, num_frames)

    # another vehicle (index 1) sits nearby.
    x[1] = np.linspace(5.0, 24.0, num_frames)
    y[1] = 2.0

    # a third (index 2), unused (invalid).
    valid[2] = False

    log_trajectory = types.SimpleNamespace(
        x=x, y=y, yaw=yaw, length=length, width=width, valid=valid,
    )

    object_metadata = types.SimpleNamespace(ids=np.array([1, 2, 3]))

    roadgraph_points = types.SimpleNamespace(
        x=np.array([0.0, 10.0]),
        y=np.array([5.0, 5.0]),
        types=np.array([15, 15]),
        valid=np.array([True, True]),
    )

    return types.SimpleNamespace(
        log_trajectory=log_trajectory,
        object_metadata=object_metadata,
        roadgraph_points=roadgraph_points,
    )


def test_renderer_produces_nonempty_png(tmp_path):
    source_xy = np.stack([np.linspace(0.0, 10.0, 11), np.zeros(11)], axis=1)
    target_xy = np.stack([np.linspace(10.0, 20.0, 11), np.zeros(11)], axis=1)
    source = _make_lane(1, source_xy)
    target = _make_lane(2, target_xy)

    state = _make_fake_state()
    record = types.SimpleNamespace(state=state, sdc_index=0, sdc_id=1)

    candidate = types.SimpleNamespace(
        candidate_id="shard#0__t10__1_2",
        scene_key="shard#0",
        transition_frame=10,
        source_lane_id=1,
        target_lane_id=2,
        decision="accept",
        reason=None,
        endpoint_target_distance_m=0.6,
        heading_difference_deg=0.1,
        separation_reduction_m=5.0,
        decreasing_fraction=0.9,
        source_remaining_distance_m=0.0,
        target_lane_persistent=True,
        merge_start_s=2.0,
        merge_end_s=10.0,
        front_vehicle_id=2,
        front_gap_m=10.0,
        front_ttc_s=5.0,
        rear_vehicle_id=None,
        rear_gap_m=None,
        rear_ttc_s=float("inf"),
    )

    output_path = tmp_path / "candidate.png"

    render_candidate_figure(
        record,
        candidate,
        source,
        target,
        [source, target],
        None,
        output_path,
        viewport_radius_m=30.0,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_review_decision_directory_naming():
    from src.scenarios.dataset_builder import sanitize_candidate_id_for_filename

    output_dir = Path("outputs/phase1/merge_validation")
    decision = "review"
    candidate_id = "shard#0__t10__1_2"

    path = output_dir / decision / (
        sanitize_candidate_id_for_filename(candidate_id) + ".png"
    )

    assert path.parent.name == "review"
    assert "#" not in path.name
