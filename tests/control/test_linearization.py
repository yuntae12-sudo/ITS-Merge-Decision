"""Stage 3-D: analytic-vs-finite-difference Jacobian correctness test.

Hard correctness gate (per the task brief): analytic Jacobians (A, B)
from ``linearization.linearize`` must match central-difference
numerical Jacobians of the SAME ``vehicle_model.step`` function,
across multiple random valid (state, control) samples.
"""

import numpy as np
import pytest

from src.control import linearization, vehicle_model
from src.control.mpc_types import MAX_ACCEL_MPS2, MAX_STEERING_CURVATURE, MPC_DT_S

EPS = 1e-6


def _numerical_jacobian_state(state, control, dt):
    n = state.shape[0]
    jac = np.zeros((4, n), dtype=np.float64)
    for i in range(n):
        perturbation = np.zeros(n)
        perturbation[i] = EPS
        f_plus = vehicle_model.step(state + perturbation, control, dt=dt, clip_control=False)
        f_minus = vehicle_model.step(state - perturbation, control, dt=dt, clip_control=False)
        jac[:, i] = (f_plus - f_minus) / (2 * EPS)
    return jac


def _numerical_jacobian_control(state, control, dt):
    n = control.shape[0]
    jac = np.zeros((4, n), dtype=np.float64)
    for i in range(n):
        perturbation = np.zeros(n)
        perturbation[i] = EPS
        f_plus = vehicle_model.step(state, control + perturbation, dt=dt, clip_control=False)
        f_minus = vehicle_model.step(state, control - perturbation, dt=dt, clip_control=False)
        jac[:, i] = (f_plus - f_minus) / (2 * EPS)
    return jac


def _random_samples(n_samples=25, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        x = rng.uniform(-100, 100)
        y = rng.uniform(-100, 100)
        yaw = rng.uniform(-np.pi + 0.05, np.pi - 0.05)  # avoid the wrap branch cut
        speed = rng.uniform(0.0, 20.0)
        accel = rng.uniform(-MAX_ACCEL_MPS2, MAX_ACCEL_MPS2)
        steering = rng.uniform(-MAX_STEERING_CURVATURE, MAX_STEERING_CURVATURE)
        samples.append((np.array([x, y, yaw, speed]), np.array([accel, steering])))
    return samples


@pytest.mark.parametrize("state,control", _random_samples())
def test_analytic_matches_finite_difference_jacobian(state, control):
    A, B, c = linearization.linearize(state, control, dt=MPC_DT_S)

    A_numerical = _numerical_jacobian_state(state, control, MPC_DT_S)
    B_numerical = _numerical_jacobian_control(state, control, MPC_DT_S)

    np.testing.assert_allclose(A, A_numerical, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(B, B_numerical, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("state,control", _random_samples(n_samples=10, seed=1))
def test_affine_offset_reproduces_exact_forward_step(state, control):
    """A@x + B@u + c must equal the exact nonlinear f(x,u) at the
    linearization point itself (by construction, since c is solved
    for exactly that)."""

    A, B, c = linearization.linearize(state, control, dt=MPC_DT_S)
    reconstructed = A @ state + B @ control + c
    exact = vehicle_model.step(state, control, dt=MPC_DT_S, clip_control=False)
    np.testing.assert_allclose(reconstructed, exact, atol=1e-9)
