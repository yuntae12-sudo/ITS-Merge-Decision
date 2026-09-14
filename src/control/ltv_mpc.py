"""Phase 3 Stage 3-D: LTV-MPC solve loop.

Optimizer choice: ``scipy.optimize.minimize`` with method
``L-BFGS-B``. Justification (per the task brief's request to document
the choice against the actual constraint formulation): the only
constraints here are simple per-step box bounds on
(accel, steering) -- no linear/nonlinear equality or inequality
constraints, no coupling between horizon steps other than through the
cost/rollout itself. ``L-BFGS-B`` is the standard SciPy method for
exactly this shape (smooth objective, box bounds only, moderate
dimensionality -- here 2*N decision variables, N typically <= ~20-30
for a few-second horizon at dt=0.1s) and is materially cheaper than a
general-purpose constrained method like SLSQP or trust-constr since it
never needs to handle general constraint Jacobians.

Gradient: the cost (``mpc_cost.total_cost``) is evaluated by
forward-simulating the ACTUAL nonlinear vehicle model (not the LTV
-linearized one) over the horizon -- simpler and still correct, per
the task brief's explicit permission to do so instead of requiring the
LTV rollout for cost evaluation. This module supplies an ANALYTIC
gradient of that cost via reverse-mode chain rule through the
horizon, built directly from ``linearization.py``'s analytic per-step
Jacobians (evaluated along the current-iterate's own nonlinear
rollout at each cost evaluation) -- this is standard direct
-shooting/discrete-adjoint MPC gradient computation, preferred here
over SciPy's own finite-difference gradient (``jac=None``) since it is
faster (O(N) analytic Jacobian evaluations instead of O(N) additional
finite-difference cost evaluations per gradient call) and more
numerically robust, and the analytic Jacobians were already required
and validated by Stage 3-D.2's own correctness gate. Where "LTV" in
this module's name comes from: at each solver iterate, the
prediction model is re-linearized (Jacobians A/B/c re-evaluated) along
the current nonlinear trajectory guess -- exactly the
successive-linearization pattern the task brief specifies -- even
though the cost itself is evaluated on the true nonlinear rollout for
correctness.

Warm-start: the previous solve's full horizon-length solution is
shifted by one step (drop control[0], append a repeat of control[-1])
and used as the next solve's initial guess. With no warm-start (episode
start / after ``reset()``), the default initial guess is all-zero
(zero acceleration, zero steering) for every horizon step.

First-control-only convention: only ``horizon_commands[0]`` is ever
returned as the "command to apply now" (``MpcResult.command``); the
full solved sequence is retained internally as the next warm-start
(standard receding-horizon MPC).

Failure handling: if the solver's returned decision vector is
non-finite, or (beyond a small numerical tolerance) violates the box
bounds, this returns ``ControllerStatus.SOLVER_FAILURE`` explicitly --
no hidden fallback command is substituted. Non-finite
``ControllerState``/reference input is rejected up front with
``ControllerStatus.INVALID_INPUT``, before any solve is attempted.
"""

import dataclasses
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

from src.control import linearization, mpc_cost, vehicle_model
from src.control.mpc_cost import CostWeights, bounds_for_horizon
from src.control.mpc_types import (
    ControllerCommand,
    ControllerState,
    ControllerStatus,
    MpcResult,
    ReferencePoint,
)

# Small numerical tolerance for post-solve bound verification -- the
# optimizer's own bounds handling can leave a result a few ULPs outside
# the nominal bound; anything beyond this is treated as a genuine
# solver failure, not float noise.
_BOUND_TOLERANCE = 1e-6


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _analytic_gradient(
    decision_vector: np.ndarray,
    initial_state: np.ndarray,
    reference_states: np.ndarray,
    dt: float,
    weights: CostWeights,
    previous_command: Optional[np.ndarray],
) -> np.ndarray:
    """Reverse-mode (discrete-adjoint) gradient of ``total_cost`` with
    respect to the flattened control decision vector, via the chain
    rule through the horizon using ``linearization.linearize``'s
    analytic per-step Jacobians, re-evaluated at the CURRENT iterate's
    own nonlinear rollout (successive-linearization / LTV pattern).

    Standard discrete-time adjoint recursion: define per-step running
    cost ``l_k(x_{k+1}, u_k)``; let ``lambda_k = d(cost)/d(x_{k+1})``
    (direct cost gradient w.r.t. that step's resulting state, since
    each per-step term depends only on its own step's state, plus the
    terminal bonus at k=N-1). Then:

        d(cost)/d(u_k) = dl_k/du_k + B_k^T @ (lambda_k + A_{k+1}^T @ ... )

    Implemented here as a standard backward pass accumulating a
    costate ``p`` (d(cost)/d(x_k), propagated backward via A^T),
    initialized at 0 after the last step (no cost depends on any state
    beyond the horizon).
    """

    controls = np.asarray(decision_vector, dtype=np.float64).reshape(-1, 2)
    horizon = controls.shape[0]
    states = mpc_cost.rollout(initial_state, controls, dt)  # (N+1, 4)

    grad = np.zeros((horizon, 2), dtype=np.float64)
    # costate: d(total_cost)/d(states[k]) accumulated from steps >= k.
    costate_next = np.zeros(4, dtype=np.float64)

    for k in reversed(range(horizon)):
        s_next = states[k + 1]
        ref = reference_states[k]
        is_terminal = k == horizon - 1

        pos_scale = 2.0 * (weights.w_pos + (weights.w_terminal_pos if is_terminal else 0.0))
        yaw_scale = 2.0 * (weights.w_yaw + (weights.w_terminal_yaw if is_terminal else 0.0))
        speed_scale = 2.0 * (weights.w_speed + (weights.w_terminal_speed if is_terminal else 0.0))

        d_cost_d_snext = np.array(
            [
                pos_scale * (s_next[0] - ref[0]),
                pos_scale * (s_next[1] - ref[1]),
                yaw_scale * _wrap_angle(s_next[2] - ref[2]),
                speed_scale * (s_next[3] - ref[3]),
            ]
        )

        # Total sensitivity of cost to this step's resulting state:
        # direct term (this step's own tracking error) plus whatever
        # flows back from later steps via A (state-to-state Jacobian).
        total_d_cost_d_snext = d_cost_d_snext + costate_next

        A, B, _c = linearization.linearize(states[k], controls[k], dt=dt)

        # d(cost)/d(u_k) via B^T, plus direct control-cost terms
        # (effort + rate, w.r.t. this step's own control).
        d_cost_d_uk = B.T @ total_d_cost_d_snext

        accel_k, steering_k = controls[k]
        d_cost_d_uk[0] += 2.0 * weights.w_accel_effort * accel_k
        d_cost_d_uk[1] += 2.0 * weights.w_steering_effort * steering_k

        if k == 0:
            prev = previous_command
        else:
            prev = controls[k - 1]
        if prev is not None:
            d_accel = accel_k - prev[0]
            d_steering = steering_k - prev[1]
            d_cost_d_uk[0] += 2.0 * weights.w_accel_rate * d_accel
            d_cost_d_uk[1] += 2.0 * weights.w_steering_rate * d_steering
            if k > 0:
                # this step's rate term also depends on controls[k-1]
                # (the "prev" for step k); handled when k-1 is
                # processed via its own d_cost_d_uk direct term below,
                # by symmetry we add the cross term directly here to
                # controls[k-1]'s gradient slot.
                grad[k - 1, 0] += -2.0 * weights.w_accel_rate * d_accel
                grad[k - 1, 1] += -2.0 * weights.w_steering_rate * d_steering

        grad[k] += d_cost_d_uk

        # Propagate costate backward: d(cost)/d(x_k) = A^T @ (total
        # sensitivity to x_{k+1}).
        costate_next = A.T @ total_d_cost_d_snext

    return grad.reshape(-1)


@dataclasses.dataclass(frozen=True)
class MpcConfig:
    horizon: int
    dt_s: float
    weights: CostWeights
    max_iterations: int = 100


def load_mpc_config(path: str = "configs/phase3_downstream.yaml") -> MpcConfig:
    """Loads the ``mpc:`` section Stage 3-D added to
    ``configs/phase3_downstream.yaml`` (additive to Stage 3-C's
    existing ``planner:``/``feasibility:``/``collision:`` sections)."""

    import yaml

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    mpc = raw["mpc"]
    w = mpc["weights"]

    return MpcConfig(
        horizon=int(mpc["horizon_steps"]),
        dt_s=float(mpc["dt_s"]),
        max_iterations=int(mpc["max_iterations"]),
        weights=CostWeights(
            w_pos=float(w["w_pos"]),
            w_yaw=float(w["w_yaw"]),
            w_speed=float(w["w_speed"]),
            w_accel_effort=float(w["w_accel_effort"]),
            w_steering_effort=float(w["w_steering_effort"]),
            w_accel_rate=float(w["w_accel_rate"]),
            w_steering_rate=float(w["w_steering_rate"]),
            w_terminal_pos=float(w["w_terminal_pos"]),
            w_terminal_yaw=float(w["w_terminal_yaw"]),
            w_terminal_speed=float(w["w_terminal_speed"]),
        ),
    )


class LtvMpcController:
    """Stateful (warm-start only) LTV-MPC controller. Not thread-safe
    (mirrors the rest of this repo's per-episode-instance controller
    pattern)."""

    def __init__(self, config: MpcConfig):
        self._config = config
        self._warm_start: Optional[np.ndarray] = None  # (N, 2) or None
        self._previous_command: Optional[np.ndarray] = None  # (2,) or None

    def reset(self) -> None:
        """Clears all internal warm-start state. Must be called at
        episode boundaries (and, later, Stage 3-F chain-transition
        boundaries). No stuck-recovery/fail-safe heuristic -- purely
        an explicit state clear."""

        self._warm_start = None
        self._previous_command = None

    def _default_initial_guess(self) -> np.ndarray:
        return np.zeros((self._config.horizon, 2), dtype=np.float64)

    def solve(
        self,
        state: ControllerState,
        reference: List[ReferencePoint],
    ) -> MpcResult:
        """Solves for a horizon-length control sequence minimizing
        ``mpc_cost.total_cost`` subject to box bounds, returning only
        the first-step command (full sequence retained as the next
        warm-start).

        Args:
            state: current ego ``ControllerState``.
            reference: list of exactly ``config.horizon``
                ``ReferencePoint``s, already aligned to the MPC's own
                dt (see module docstring / ``docs/phase3/
                OVERNIGHT_PROGRESS.md`` for why no resampling
                machinery is implemented: Stage 3-C's planner already
                produces trajectories at dt=0.1s, matching the MPC's
                own locked dt 1:1, so direct indexing suffices and
                resampling would be unneeded machinery).
        """

        horizon = self._config.horizon

        if not state.is_finite():
            return MpcResult(
                status=ControllerStatus.INVALID_INPUT,
                command=None, horizon_commands=None, predicted_states=None,
                cost=None, solver_iterations=None,
                error=f"Non-finite ControllerState: {state!r}",
            )
        if len(reference) != horizon:
            return MpcResult(
                status=ControllerStatus.INVALID_INPUT,
                command=None, horizon_commands=None, predicted_states=None,
                cost=None, solver_iterations=None,
                error=(
                    f"Expected {horizon} reference points (config.horizon), "
                    f"got {len(reference)}."
                ),
            )
        reference_states = np.array(
            [[r.x, r.y, r.yaw, r.speed] for r in reference], dtype=np.float64
        )
        if not np.all(np.isfinite(reference_states)):
            return MpcResult(
                status=ControllerStatus.INVALID_INPUT,
                command=None, horizon_commands=None, predicted_states=None,
                cost=None, solver_iterations=None,
                error="Non-finite reference trajectory input.",
            )

        initial_state = state.as_array()

        if self._warm_start is not None and self._warm_start.shape == (horizon, 2):
            initial_guess = self._warm_start
        else:
            initial_guess = self._default_initial_guess()

        bounds = bounds_for_horizon(horizon)

        result = minimize(
            fun=mpc_cost.total_cost,
            x0=initial_guess.reshape(-1),
            args=(
                initial_state,
                reference_states,
                self._config.dt_s,
                self._config.weights,
                self._previous_command,
            ),
            method="L-BFGS-B",
            jac=_analytic_gradient,
            bounds=bounds,
            options={"maxiter": self._config.max_iterations},
        )

        solution = np.asarray(result.x, dtype=np.float64).reshape(horizon, 2)

        if not np.all(np.isfinite(solution)):
            return MpcResult(
                status=ControllerStatus.SOLVER_FAILURE,
                command=None, horizon_commands=None, predicted_states=None,
                cost=None, solver_iterations=int(getattr(result, "nit", 0)),
                error=f"Solver returned non-finite solution: {result.message}",
            )

        lower = np.array([b[0] for b in bounds], dtype=np.float64).reshape(horizon, 2)
        upper = np.array([b[1] for b in bounds], dtype=np.float64).reshape(horizon, 2)
        if np.any(solution < lower - _BOUND_TOLERANCE) or np.any(solution > upper + _BOUND_TOLERANCE):
            return MpcResult(
                status=ControllerStatus.SOLVER_FAILURE,
                command=None, horizon_commands=None, predicted_states=None,
                cost=None, solver_iterations=int(getattr(result, "nit", 0)),
                error="Solver returned a command outside the box bounds.",
            )

        # Clip away any tiny numerical slack from the tolerance check
        # above so the returned command is always strictly in-bounds.
        solution = np.clip(solution, lower, upper)

        predicted_states = mpc_cost.rollout(initial_state, solution, self._config.dt_s)

        first_command = ControllerCommand.from_array(solution[0])

        # Warm-start update: shift by one step, repeat the last control.
        shifted = np.empty_like(solution)
        shifted[:-1] = solution[1:]
        shifted[-1] = solution[-1]
        self._warm_start = shifted
        self._previous_command = solution[0].copy()

        return MpcResult(
            status=ControllerStatus.OK,
            command=first_command,
            horizon_commands=solution,
            predicted_states=predicted_states,
            cost=float(result.fun),
            solver_iterations=int(getattr(result, "nit", 0)),
        )
