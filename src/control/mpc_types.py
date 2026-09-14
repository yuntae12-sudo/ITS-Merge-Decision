"""Phase 3 Stage 3-D: MPC data types.

Minimal, dependency-light dataclasses shared by ``vehicle_model.py``,
``linearization.py``, ``mpc_cost.py``, ``constraints.py``, and
``ltv_mpc.py``. Nothing here depends on Waymax's own datatypes module
-- the controller only needs to match Waymax's *dynamics equation*
(see ``vehicle_model.py``), not its runtime types.

State/control convention (locked, Stage 3-0, restated here since this
is the module every other file imports it from):

  - ``ControllerState``: ``x, y, yaw, speed`` (4 scalars). This is the
    SAME shape ASMC's own state used, but here it must exactly mirror
    Waymax's own ``InvertibleBicycleModel`` trajectory-update
    semantics -- no wheelbase, no ASMC bicycle-model variant.
  - ``ControllerCommand``: ``acceleration_mps2, steering_curvature``
    (2 scalars) -- Waymax's own action space, unnormalized physical
    units (``normalize_actions=False``, matching
    ``src/environment/low_level_controller.py`` and
    ``src/environment/merge_environment.py``).
  - Box bounds: acceleration in [-6.0, +6.0] m/s^2, steering curvature
    in [-0.3, +0.3] 1/m -- see ``constraints.py``.
"""

import dataclasses
import enum
from typing import List, Optional

import numpy as np

# Waymax's own InvertibleBicycleModel defaults, reused directly (see
# module docstring; also src/environment/low_level_controller.py).
MAX_ACCEL_MPS2 = 6.0
MAX_STEERING_CURVATURE = 0.3
MPC_DT_S = 0.1  # Locked Stage 3-0 decision #2: matches Waymax sim dt 1:1.


@dataclasses.dataclass(frozen=True)
class ControllerState:
    """Current (never future/logged) ego state, Waymax-native."""

    x: float
    y: float
    yaw: float
    speed: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.speed], dtype=np.float64)

    @staticmethod
    def from_array(arr: np.ndarray) -> "ControllerState":
        return ControllerState(
            x=float(arr[0]), y=float(arr[1]), yaw=float(arr[2]), speed=float(arr[3])
        )

    def is_finite(self) -> bool:
        return bool(np.all(np.isfinite(self.as_array())))


@dataclasses.dataclass(frozen=True)
class ControllerCommand:
    """One control input: Waymax's own ``(acceleration, steering
    curvature)`` action space, unnormalized physical units."""

    acceleration_mps2: float
    steering_curvature: float

    def as_array(self) -> np.ndarray:
        return np.array([self.acceleration_mps2, self.steering_curvature], dtype=np.float64)

    @staticmethod
    def from_array(arr: np.ndarray) -> "ControllerCommand":
        return ControllerCommand(
            acceleration_mps2=float(arr[0]), steering_curvature=float(arr[1])
        )

    def is_finite(self) -> bool:
        return bool(np.all(np.isfinite(self.as_array())))


@dataclasses.dataclass(frozen=True)
class ReferencePoint:
    """One horizon-step reference target: x, y, yaw, speed. Sourced
    from Stage 3-C's ``CartesianTrajectory`` (Stage 3-E's job to wire
    up) -- this module only needs the four tracked quantities, not the
    curvature/acceleration fields ``CartesianTrajectory`` also
    carries."""

    x: float
    y: float
    yaw: float
    speed: float


class ControllerStatus(str, enum.Enum):
    """Explicit outcome status, consistent in spirit with Stage 3-C's
    ``PlannerStatus`` pattern (see ``src.planning.candidate_evaluator.
    PlannerStatus``): a failure must be reported explicitly, never
    silently papered over with a fallback that changes behavior
    semantics (Stage 3-0 audit, restated in the task brief)."""

    OK = "OK"
    INVALID_INPUT = "INVALID_INPUT"
    """Non-finite ControllerState or reference input was rejected
    before any solve was attempted."""

    SOLVER_FAILURE = "SOLVER_FAILURE"
    """The optimizer ran but produced a non-finite result, or the
    returned command violates the box bounds beyond a tiny numerical
    tolerance."""


@dataclasses.dataclass(frozen=True)
class MpcResult:
    """Result of one ``LtvMpcController.solve()`` call."""

    status: ControllerStatus
    command: Optional[ControllerCommand]
    """The FIRST horizon-step command only -- never None when
    ``status == OK``."""

    horizon_commands: Optional[np.ndarray]
    """Shape (N, 2): the full solved control sequence, kept as the
    next warm-start. None unless ``status == OK``."""

    predicted_states: Optional[np.ndarray]
    """Shape (N+1, 4): the nonlinear forward rollout under
    ``horizon_commands``, starting at the given ``ControllerState``.
    Diagnostic only. None unless ``status == OK``."""

    cost: Optional[float]
    solver_iterations: Optional[int]
    error: Optional[str] = None
