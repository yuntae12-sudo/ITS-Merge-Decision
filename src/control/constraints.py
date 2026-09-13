"""Phase 3 Stage 3-D: MPC box constraints.

The actual bound values and the ``scipy.optimize.minimize(bounds=...)``
builder live in ``mpc_cost.py`` (see that module's docstring for why
cost and constraints were folded together: the constraint set here is
just a pair of box bounds, naturally expressed alongside the cost
rather than forced into its own near-empty file). This module exists
as a thin, explicit re-export so ``from src.control.constraints import
...`` works for callers/tests that expect a dedicated constraints
module, and so the box-bound VALUES have one clearly-named place to
look them up without needing to know they happen to live next to the
cost code.
"""

from src.control.mpc_cost import bounds_for_horizon
from src.control.mpc_types import MAX_ACCEL_MPS2, MAX_STEERING_CURVATURE

ACCEL_BOUNDS = (-MAX_ACCEL_MPS2, MAX_ACCEL_MPS2)
STEERING_BOUNDS = (-MAX_STEERING_CURVATURE, MAX_STEERING_CURVATURE)

__all__ = [
    "bounds_for_horizon",
    "ACCEL_BOUNDS",
    "STEERING_BOUNDS",
    "MAX_ACCEL_MPS2",
    "MAX_STEERING_CURVATURE",
]
