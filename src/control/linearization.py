"""Phase 3 Stage 3-D: analytic Jacobians of the Waymax-native forward
model, for LTV (successive-linearization) MPC.

State ``x = [x, y, yaw, speed]``, control ``u = [accel, steering]``.
Forward model (see ``vehicle_model.py`` for the full, verbatim-quoted
Waymax equation this differentiates):

    vel_x = speed * cos(yaw)
    vel_y = speed * sin(yaw)
    new_x    = x + vel_x*t + 0.5*accel*cos(yaw)*t^2
    new_y    = y + vel_y*t + 0.5*accel*sin(yaw)*t^2
    delta_yaw = steering * (speed*t + 0.5*accel*t^2)
    new_yaw  = yaw + delta_yaw           (mod 2*pi wrap -- wrap has
                                           zero local derivative almost
                                           everywhere, see note below)
    new_speed = speed + accel*t

Substituting vel_x/vel_y explicitly:

    new_x = x + speed*cos(yaw)*t + 0.5*accel*cos(yaw)*t^2
          = x + cos(yaw) * (speed*t + 0.5*accel*t^2)
    new_y = y + sin(yaw) * (speed*t + 0.5*accel*t^2)

Let ``k = speed*t + 0.5*accel*t^2`` (the same "arc length" term
steering multiplies for yaw). Then:

    new_x = x + cos(yaw) * k
    new_y = y + sin(yaw) * k

Partial derivatives (evaluated at an arbitrary operating point
``x_bar = [x, y, yaw, speed]``, ``u_bar = [accel, steering]``):

  d(new_x)/dx     = 1
  d(new_x)/dy     = 0
  d(new_x)/dyaw   = -sin(yaw) * k
  d(new_x)/dspeed = cos(yaw) * t              (dk/dspeed = t)
  d(new_x)/daccel = cos(yaw) * 0.5*t^2         (dk/daccel = 0.5*t^2)
  d(new_x)/dsteering = 0

  d(new_y)/dx     = 0
  d(new_y)/dy     = 1
  d(new_y)/dyaw   = cos(yaw) * k
  d(new_y)/dspeed = sin(yaw) * t
  d(new_y)/daccel = sin(yaw) * 0.5*t^2
  d(new_y)/dsteering = 0

  new_yaw = yaw + steering*k  (k as above; wrap only relocates by a
  constant 2*pi multiple and has zero derivative except on a measure
  -zero set at the +-pi branch cut, which no interior linearization
  point in these tests ever lands exactly on):
  d(new_yaw)/dx = d(new_yaw)/dy = 0
  d(new_yaw)/dyaw     = 1                      (k does not depend on yaw)
  d(new_yaw)/dspeed   = steering * t
  d(new_yaw)/daccel   = steering * 0.5*t^2
  d(new_yaw)/dsteering = k

  new_speed = speed + accel*t:
  d(new_speed)/dx = d(new_speed)/dy = d(new_speed)/dyaw = 0
  d(new_speed)/dspeed = 1
  d(new_speed)/daccel = t
  d(new_speed)/dsteering = 0

This gives the 4x4 state Jacobian A = d(new_x_vec)/d(x_vec) and the
4x2 control Jacobian B = d(new_x_vec)/d(u_vec), both evaluated at
(x_bar, u_bar). The affine offset (for the standard LTV form
``x_next ~= A@x + B@u + c``) is recovered as
``c = f(x_bar, u_bar) - A@x_bar - B@u_bar`` (exact for this smooth,
almost-everywhere-differentiable model at any point off the wrap
branch cut).
"""

import numpy as np

from src.control import vehicle_model
from src.control.mpc_types import MPC_DT_S


def linearize(state: np.ndarray, control: np.ndarray, dt: float = MPC_DT_S):
    """Analytic Jacobians of ``vehicle_model.step`` at one operating
    point.

    Args:
        state: ``[x, y, yaw, speed]`` (shape (4,)).
        control: ``[accel, steering]`` (shape (2,)) -- the UNCLIPPED
            control the Jacobian is evaluated at (the model clips
            internally; this linearization assumes the operating
            point is strictly inside the box, which the finite
            -difference test below verifies is exact at all such
            points -- if a caller linearizes exactly AT a bound, the
            clip's corner is non-differentiable, same as any bound
            -constrained smooth function).

    Returns:
        (A, B, c): A is (4,4), B is (4,2), c is (4,).
    """

    state = np.asarray(state, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)

    yaw = state[2]
    speed = state[3]
    accel = control[0]
    steering = control[1]
    t = dt

    k = speed * t + 0.5 * accel * t**2
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    A = np.zeros((4, 4), dtype=np.float64)
    A[0, 0] = 1.0
    A[0, 2] = -sin_yaw * k
    A[0, 3] = cos_yaw * t

    A[1, 1] = 1.0
    A[1, 2] = cos_yaw * k
    A[1, 3] = sin_yaw * t

    A[2, 2] = 1.0
    A[2, 3] = steering * t

    A[3, 3] = 1.0

    B = np.zeros((4, 2), dtype=np.float64)
    B[0, 0] = cos_yaw * 0.5 * t**2
    B[1, 0] = sin_yaw * 0.5 * t**2
    B[2, 0] = steering * 0.5 * t**2
    B[2, 1] = k
    B[3, 0] = t

    f_bar = vehicle_model.step(state, control, dt=dt, clip_control=False)
    c = f_bar - A @ state - B @ control

    return A, B, c
