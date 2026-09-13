"""Phase 3 Stage 3-B: Frenet-frame data types.

Minimal, action-agnostic dataclasses used by the Frenet trajectory
core (``polynomial.py``, ``frenet_transform.py``) and, later, Stage
3-C's BehaviorAction-specific candidate generation. Nothing in this
module depends on ``BehaviorAction`` or ``MergeEnvironment`` -- these
are pure math containers.

Sign/units convention (pinned here, consumed by ``frenet_transform.py``
and re-verified against Stage 3-A's ``ReferenceLine`` in
``tests/planning/test_frenet_transform.py``):
  - ``s`` is arc length along the reference line (m), same convention
    as ``ReferenceLine.s``.
  - ``d`` is signed lateral offset from the reference line (m):
    positive = LEFT of the reference line's direction of travel,
    matching ``lane_geometry.py``'s own
    ``project_point_to_polyline_signed`` convention (see Stage 3-A
    docstring), which Stage 3-A's ``ReferenceLine.project`` reuses
    directly.
  - Dotted names (``s_d``, ``d_d``, ...) are time derivatives; double-
    dotted (``s_dd``, ``d_dd``) are second time derivatives
    (acceleration-like); ``s_ddd`` is the third time derivative
    (jerk-like), kept only on ``FrenetPath`` for later jerk-limit
    checks in Stage 3-C -- not implemented/enforced in this stage.
"""

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass(frozen=True)
class FrenetState:
    """A single instant in the Frenet frame relative to some
    ``ReferenceLine``.

    Longitudinal: ``s`` (position, m), ``s_d`` (speed, m/s), ``s_dd``
    (acceleration, m/s^2).
    Lateral: ``d`` (offset, m), ``d_d`` (rate, m/s), ``d_dd``
    (m/s^2). Lateral derivatives here are with respect to TIME (not
    arc length) -- consistent with what ``frenet_transform.py``
    produces/consumes.
    """

    s: float
    s_d: float
    s_dd: float
    d: float
    d_d: float
    d_dd: float


@dataclasses.dataclass(frozen=True)
class FrenetPath:
    """A time-sampled Frenet-frame trajectory.

    All arrays share the same length (one entry per sample time
    ``t[i]``). ``s_ddd`` (longitudinal jerk) is optional and may be
    left as ``None`` when not computed by a given generator.

    ``valid`` and ``rejection_reason`` exist for Stage 3-C's candidate
    evaluator: this stage only defines the fields (defaulting to
    ``True``/``None``, i.e. "not yet evaluated / assumed valid") and
    does not implement any rejection logic.
    """

    t: np.ndarray
    s: np.ndarray
    s_d: np.ndarray
    s_dd: np.ndarray
    d: np.ndarray
    d_d: np.ndarray
    d_dd: np.ndarray
    s_ddd: Optional[np.ndarray] = None
    valid: bool = True
    rejection_reason: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class CartesianTrajectory:
    """A time-sampled Cartesian-frame trajectory, the output of
    converting a ``FrenetPath`` via ``frenet_transform.py`` against a
    ``ReferenceLine``.

    All arrays share the same length (one entry per sample time
    ``t[i]``). ``curvature`` is the path curvature (1/m) implied by
    the trajectory's heading rate over arc length; ``acceleration``
    is the scalar (tangential) acceleration, matching the convention
    ``frenet_transform.frenet_to_cartesian`` produces.
    """

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    curvature: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
