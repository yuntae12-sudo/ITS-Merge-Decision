"""Phase 3 Stage 3-D: MPC quadratic cost + box constraints.

Cost and constraints are folded into one module (per the task brief's
"merge small modules together if genuinely cleaner" guidance): the
constraint set here is a simple pair of box bounds (no rate constraint
-- see below), which is naturally expressed alongside the cost as
"what ``scipy.optimize.minimize(..., bounds=...)`` needs", rather than
forcing an artificially separate file for two constants.

Cost terms (all weights loaded from ``configs/phase3_downstream.yaml``
``mpc:`` section -- see that file's comments for the "initial, not
final" caveat), over a horizon of N steps:

  - position tracking error: w_pos * ((x-x_ref)^2 + (y-y_ref)^2)
  - heading tracking error (angle-wrapped): w_yaw * wrap(yaw-yaw_ref)^2
  - speed tracking error: w_speed * (speed-speed_ref)^2
  - control effort: w_accel_effort * accel^2 + w_steering_effort * steering^2
  - control-rate smoothness: w_accel_rate * (accel_k - accel_{k-1})^2 +
    w_steering_rate * (steering_k - steering_{k-1})^2 (rate at k=0 is
    taken against the PREVIOUS applied command, if given, else the
    first horizon-step's own rate term is simply omitted -- there is
    no "previous" control before the first solve of an episode)
  - terminal cost: heavier w_terminal_pos/w_terminal_yaw/w_terminal_speed
    multipliers applied ONLY at the final horizon step, added on top of
    (not instead of) that step's regular per-step tracking cost.

Constraints: simple box bounds on (accel, steering) per horizon step,
matching Waymax's own ``InvertibleBicycleModel`` bounds exactly (see
``mpc_types.py``). No control-RATE constraint is added: the synthetic
tracking tests (``tests/control/test_ltv_mpc.py``) are stable with box
bounds + the rate-smoothness cost term alone (see
``docs/phase3/OVERNIGHT_PROGRESS.md`` for the empirical check), so per
the task brief's explicit instruction not to add one speculatively,
none is added here.
"""

import dataclasses
from typing import Optional

import numpy as np

from src.control.mpc_types import MAX_ACCEL_MPS2, MAX_STEERING_CURVATURE
from src.control import vehicle_model


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclasses.dataclass(frozen=True)
class CostWeights:
    w_pos: float
    w_yaw: float
    w_speed: float
    w_accel_effort: float
    w_steering_effort: float
    w_accel_rate: float
    w_steering_rate: float
    w_terminal_pos: float
    w_terminal_yaw: float
    w_terminal_speed: float


def bounds_for_horizon(horizon: int):
    """``scipy.optimize.minimize(bounds=...)`` list for a flattened
    ``[accel_0, steering_0, accel_1, steering_1, ...]`` decision
    vector of length ``2*horizon``."""

    single_step = [
        (-MAX_ACCEL_MPS2, MAX_ACCEL_MPS2),
        (-MAX_STEERING_CURVATURE, MAX_STEERING_CURVATURE),
    ]
    return single_step * horizon


def rollout(
    initial_state: np.ndarray, controls: np.ndarray, dt: float
) -> np.ndarray:
    """Forward-simulates the ACTUAL nonlinear model (not the LTV
    linearization) over the full horizon, given a flattened or (N,2)
    control sequence. Returns states of shape (N+1, 4): index 0 is
    ``initial_state``, index i+1 is the state after applying
    ``controls[i]``.

    Cost is evaluated against this true nonlinear rollout (see
    ``ltv_mpc.py`` module docstring for why: simpler and still
    correct, per the task brief's explicit permission to do so instead
    of rolling out the LTV-linearized model for cost evaluation)."""

    controls = np.asarray(controls, dtype=np.float64).reshape(-1, 2)
    horizon = controls.shape[0]
    states = np.empty((horizon + 1, 4), dtype=np.float64)
    states[0] = initial_state
    for k in range(horizon):
        states[k + 1] = vehicle_model.step(states[k], controls[k], dt=dt, clip_control=True)
    return states


def total_cost(
    decision_vector: np.ndarray,
    initial_state: np.ndarray,
    reference_states: np.ndarray,
    dt: float,
    weights: CostWeights,
    previous_command: Optional[np.ndarray] = None,
) -> float:
    """Scalar total cost over the horizon.

    Args:
        decision_vector: flattened (2*N,) control sequence.
        initial_state: (4,) current ``[x,y,yaw,speed]``.
        reference_states: (N, 4) ``[x_ref,y_ref,yaw_ref,speed_ref]``
            per horizon step (already aligned 1:1 with the MPC's own
            dt/horizon -- see ``ltv_mpc.py``).
        dt: MPC time step.
        weights: ``CostWeights``.
        previous_command: (2,) or None -- the previously APPLIED
            command (for the rate term at horizon index 0); None at
            episode start / after ``reset()``.
    """

    controls = np.asarray(decision_vector, dtype=np.float64).reshape(-1, 2)
    horizon = controls.shape[0]
    states = rollout(initial_state, controls, dt)  # (N+1, 4)

    cost = 0.0
    for k in range(horizon):
        s = states[k + 1]
        ref = reference_states[k]
        is_terminal = k == horizon - 1

        pos_err_sq = (s[0] - ref[0]) ** 2 + (s[1] - ref[1]) ** 2
        yaw_err_sq = _wrap_angle(s[2] - ref[2]) ** 2
        speed_err_sq = (s[3] - ref[3]) ** 2

        cost += weights.w_pos * pos_err_sq
        cost += weights.w_yaw * yaw_err_sq
        cost += weights.w_speed * speed_err_sq

        if is_terminal:
            cost += weights.w_terminal_pos * pos_err_sq
            cost += weights.w_terminal_yaw * yaw_err_sq
            cost += weights.w_terminal_speed * speed_err_sq

        accel_k, steering_k = controls[k]
        cost += weights.w_accel_effort * accel_k**2
        cost += weights.w_steering_effort * steering_k**2

        if k == 0:
            prev = previous_command
        else:
            prev = controls[k - 1]
        if prev is not None:
            d_accel = accel_k - prev[0]
            d_steering = steering_k - prev[1]
            cost += weights.w_accel_rate * d_accel**2
            cost += weights.w_steering_rate * d_steering**2

    return float(cost)
