"""Stage 3-D: vehicle model correctness tests.

The critical test in this file, ``test_matches_installed_waymax_bicycle_model``,
verifies our Python ``vehicle_model.step`` EXACTLY (near machine
precision) matches the installed ``waymax.dynamics.bicycle_model.
InvertibleBicycleModel.compute_update`` for one step, across
representative Phase-2-realistic samples. This is Stage 3-D's single
hardest correctness gate (Stage 3-0 audit) -- see
``src/control/vehicle_model.py``'s module docstring for the exact
quoted equation this reproduces.
"""

import numpy as np
import pytest

from src.control import vehicle_model
from src.control.mpc_types import MAX_ACCEL_MPS2, MAX_STEERING_CURVATURE, MPC_DT_S

waymax = pytest.importorskip("waymax")
from waymax import datatypes  # noqa: E402
from waymax.dynamics.bicycle_model import InvertibleBicycleModel  # noqa: E402


def _waymax_step(x, y, yaw, speed, accel, steering, dt=MPC_DT_S):
    """Builds minimal valid Waymax Trajectory/Action objects (shape
    (1, 1), matching the (num_objects, num_timesteps=1) convention
    compute_update expects) and calls the REAL installed dynamics
    model directly."""

    vel_x = speed * np.cos(yaw)
    vel_y = speed * np.sin(yaw)

    traj = datatypes.Trajectory(
        x=np.array([[x]], dtype=np.float32),
        y=np.array([[y]], dtype=np.float32),
        z=np.zeros((1, 1), dtype=np.float32),
        vel_x=np.array([[vel_x]], dtype=np.float32),
        vel_y=np.array([[vel_y]], dtype=np.float32),
        yaw=np.array([[yaw]], dtype=np.float32),
        valid=np.array([[True]], dtype=bool),
        timestamp_micros=np.zeros((1, 1), dtype=np.int32),
        length=np.ones((1, 1), dtype=np.float32),
        width=np.ones((1, 1), dtype=np.float32),
        height=np.ones((1, 1), dtype=np.float32),
    )
    action = datatypes.Action(
        data=np.array([[accel, steering]], dtype=np.float32),
        valid=np.array([[True]], dtype=bool),
    )
    model = InvertibleBicycleModel(
        dt=dt, max_accel=MAX_ACCEL_MPS2, max_steering=MAX_STEERING_CURVATURE,
        normalize_actions=False,
    )
    update = model.compute_update(action, traj)
    new_yaw = float(np.asarray(update.yaw)[0, 0])
    new_vel_x = float(np.asarray(update.vel_x)[0, 0])
    new_vel_y = float(np.asarray(update.vel_y)[0, 0])
    # Waymax's own new_vel = speed + accel*t is SIGNED (can go
    # negative -- no floor at zero; see vehicle_model.py's module
    # docstring point 2) and then re-expanded as
    # new_vel_x = new_vel*cos(new_yaw), new_vel_y = new_vel*sin(new_yaw).
    # A naive hypot(vel_x, vel_y) is always non-negative and would
    # silently discard that sign whenever new_vel < 0 (vel_x/vel_y
    # then point OPPOSITE new_yaw) -- recover the signed magnitude by
    # projecting (vel_x, vel_y) onto the (cos(new_yaw), sin(new_yaw))
    # unit vector instead.
    signed_speed = new_vel_x * np.cos(new_yaw) + new_vel_y * np.sin(new_yaw)
    return (
        float(np.asarray(update.x)[0, 0]),
        float(np.asarray(update.y)[0, 0]),
        new_yaw,
        float(signed_speed),
    )


# Representative Phase-2-realistic samples: speed 0..20 m/s, accel
# within [-6,6], steering within [-0.3,0.3], several yaw values
# (including near +-pi to exercise the wrap).
SAMPLES = [
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (10.0, -5.0, 0.5, 5.0, 2.0, 0.1),
    (0.0, 0.0, 1.2, 15.0, -3.0, -0.15),
    (-20.0, 30.0, -2.0, 20.0, 6.0, 0.3),
    (100.0, -50.0, 3.0, 12.0, -6.0, -0.3),
    (5.0, 5.0, np.pi - 0.01, 8.0, 1.0, 0.05),
    (0.0, 0.0, -np.pi + 0.02, 3.0, -1.5, -0.05),
    (0.0, 0.0, 0.0, 20.0, 0.0, 0.3),
    (0.0, 0.0, 0.0, 1.0, -6.0, 0.0),  # decel could drive speed toward/below 0
    (0.0, 0.0, 0.0, 0.5, -6.0, 0.2),  # decel large relative to small speed
]


@pytest.mark.parametrize("x,y,yaw,speed,accel,steering", SAMPLES)
def test_matches_installed_waymax_bicycle_model(x, y, yaw, speed, accel, steering):
    state = np.array([x, y, yaw, speed], dtype=np.float64)
    control = np.array([accel, steering], dtype=np.float64)

    ours = vehicle_model.step(state, control, dt=MPC_DT_S, clip_control=True)
    wx, wy, wyaw, wspeed = _waymax_step(x, y, yaw, speed, accel, steering)

    # Near machine precision: both are the SAME equation, not an
    # approximation. float32 (Waymax's own dtype) vs float64 (ours)
    # limits agreement to ~1e-5-1e-6 relative, not full float64
    # precision -- documented, not silently loosened beyond that.
    assert ours[0] == pytest.approx(wx, abs=1e-4)
    assert ours[1] == pytest.approx(wy, abs=1e-4)
    assert ours[2] == pytest.approx(wyaw, abs=1e-4)
    assert ours[3] == pytest.approx(wspeed, abs=1e-4)


def test_no_floor_at_zero_speed_matches_waymax():
    """Confirms deceleration past zero speed is NOT clamped, matching
    Waymax's own unclamped `new_vel = speed + accel*t` exactly."""

    state = np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float64)
    control = np.array([-6.0, 0.0], dtype=np.float64)
    ours = vehicle_model.step(state, control, dt=MPC_DT_S, clip_control=True)
    _, _, _, wspeed = _waymax_step(0.0, 0.0, 0.0, 0.5, -6.0, 0.0)
    assert ours[3] < 0.0
    assert ours[3] == pytest.approx(wspeed, abs=1e-4)


def test_clip_command_matches_waymax_bounds():
    accel, steering = vehicle_model.clip_command(100.0, -100.0)
    assert accel == MAX_ACCEL_MPS2
    assert steering == -MAX_STEERING_CURVATURE


def test_step_output_finite_for_all_samples():
    for x, y, yaw, speed, accel, steering in SAMPLES:
        state = np.array([x, y, yaw, speed], dtype=np.float64)
        control = np.array([accel, steering], dtype=np.float64)
        out = vehicle_model.step(state, control)
        assert np.all(np.isfinite(out))
