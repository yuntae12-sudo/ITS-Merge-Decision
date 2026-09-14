"""Phase 3 Stage 3-B: Frenet <-> Cartesian coordinate transforms.

Implements the standard Werling et al. (2010, "Optimal Trajectory
Generation for Dynamic Street Scenarios in a Frenet Frame") style
transform equations, adapted to:

  - consume Stage 3-A's ``ReferenceLine`` (its ``interpolate`` gives
    position/yaw/curvature/curvature_derivative at an arc length
    ``s``, and its ``project`` gives ``s``/lateral offset for the
    inverse transform) instead of a bespoke reference representation;
  - use TIME derivatives throughout (``s_d``, ``d_d``, ...), matching
    ``FrenetState``'s field convention, rather than Werling's original
    arc-length-parameterized primed notation (``d' = dd/ds``). The
    two are related by the chain rule ``d' = d_d / s_d`` wherever a
    conversion is needed internally; this module never exposes an
    arc-length-derivative quantity externally.

Sign convention (pinned in ``frenet_types.py``, re-verified here
against Stage 3-A's own ``ReferenceLine``/``project_point_to_polyline_signed``
convention): ``d > 0`` is LEFT of the reference line's direction of
travel. The left-normal unit vector at heading ``theta_r`` is
``(-sin(theta_r), cos(theta_r))``, so a Cartesian point is
``(rx, ry) + d * (-sin(theta_r), cos(theta_r))``.

Low-speed singularity
----------------------
Both directions of this transform divide by longitudinal speed
(``s_d``) in the general-case formulas (heading offset uses
``atan2(d_d, s_d * (1 - kappa_r * d))``, and curvature/acceleration
composition divide by ``s_d`` or ``s_d^2``). Frenet-frame heading is
genuinely undefined at exactly zero speed (a stationary point has no
direction of travel), and Phase 3's own STOP behavior
(``STOP_TARGET_SPEED_MPS = 0.0`` in ``src/environment/behavior_action.py``)
means this path WILL be exercised at the end of every STOP maneuver,
not just as a rare/defensive edge case.

Threshold: ``LOW_SPEED_THRESHOLD_MPS = 0.5`` m/s. Rationale, reasoned
from this repo's own numbers rather than copied from another
codebase: Phase 3's MPC operates at ``dt = 0.1`` s (locked design
decision, see ``docs/phase3/OVERNIGHT_PROGRESS.md``) and this repo's
own comfortable-deceleration constant is ``2.0 m/s^2``
(``comfortable_deceleration`` in the same locked decisions). At that
deceleration, one control step (0.1 s) changes speed by 0.2 m/s, so
0.5 m/s is about 2.5 control steps' worth of margin -- comfortably
above the "one step of integration noise" scale (avoids flapping
between the two formula branches across consecutive planner ticks
near a STOP), while remaining far below ``NOMINAL_CRUISE_SPEED_MPS =
15.0`` (3.3% of cruise speed), so the geometric fallback is never
mistakenly used during normal-speed driving. Below this threshold, we
use a purely geometric fallback that never divides by ``s_d``: heading
is taken directly from the reference line's own heading at ``s``
(``theta = theta_r``, i.e. assume the vehicle is aligned with the lane
when it is nearly stationary -- reasonable for a STOP maneuver
converging on a lane-aligned stop point, and avoids the ill-defined
"direction of travel of a motionless point" question entirely), and
curvature is taken directly as the reference line's curvature at that
point (``kappa = kappa_r``, since with zero relative heading offset
the path curvature reduces to the reference curvature).
"""

import dataclasses

import numpy as np

from src.planning.frenet_types import FrenetState
from src.planning.reference import ReferenceLine

LOW_SPEED_THRESHOLD_MPS = 0.5


@dataclasses.dataclass(frozen=True)
class CartesianState:
    """A single Cartesian-frame state, the output of
    ``frenet_to_cartesian``."""

    x: float
    y: float
    yaw: float
    curvature: float
    velocity: float
    acceleration: float


def _wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2 * np.pi) - np.pi)


def frenet_to_cartesian(
    frenet_state: FrenetState, reference: ReferenceLine
) -> CartesianState:
    """Converts a ``FrenetState`` to Cartesian, using ``reference``'s
    position/heading/curvature at ``frenet_state.s``.

    Position uses the standard Werling et al. offset formula directly:

      x = rx - d * sin(theta_r)
      y = ry + d * cos(theta_r)

    For heading, curvature, velocity, and acceleration, rather than
    transcribing Werling et al.'s higher-order closed-form heading/
    curvature/acceleration composition formulas from memory (a fragile
    multi-term expression, easy to mis-transcribe silently), this
    implementation takes a first-principles route that is equivalent
    but easier to verify:

      1. Decompose Frenet velocity/acceleration into components
         tangential and normal to the reference line at ``s``
         (``tangential_v = s_d * (1 - kappa_r * d)``, ``normal_v =
         d_d`` -- the standard Frenet-Serret arc-length-parameterized
         position-derivative decomposition, see e.g. Werling et al.
         Eq. 1-4), then rotate those into world (x, y) using the
         reference heading ``theta_r``, differentiating fully via the
         product rule (this picks up the frame's own rotation --
         centripetal/Coriolis-style cross terms -- automatically,
         since ``theta_r`` itself changes with time as ``s`` advances:
         ``d(theta_r)/dt = kappa_r * s_d``).
      2. Recover yaw, curvature, and (tangential) acceleration from
         the resulting Cartesian velocity/acceleration vectors
         (vx, vy, ax, ay) via the standard EXACT planar identities
         ``yaw = atan2(vy, vx)``, ``kappa = (vx*ay - vy*ax) / v^3``,
         ``a = (vx*ax + vy*ay) / v``. These identities hold for ANY
         smooth planar curve/motion -- they do not depend on step 1's
         Frenet decomposition being correct, only on (vx, vy, ax, ay)
         being the correct Cartesian velocity/acceleration -- so a
         round-trip test validates both steps at once, rather than
         trusting a memorized closed-form formula in isolation.
    """

    s = frenet_state.s
    d = frenet_state.d
    ref = reference.interpolate(s)
    theta_r = ref.yaw
    kappa_r = ref.curvature
    kappa_r_dot = ref.curvature_derivative

    cos_r, sin_r = np.cos(theta_r), np.sin(theta_r)
    x = ref.x - d * sin_r
    y = ref.y + d * cos_r

    s_d, s_dd = frenet_state.s_d, frenet_state.s_dd
    d_d, d_dd = frenet_state.d_d, frenet_state.d_dd

    if abs(s_d) < LOW_SPEED_THRESHOLD_MPS:
        # Low-speed geometric fallback: assume lane-aligned heading,
        # reference curvature, avoid dividing by s_d. See module
        # docstring for the threshold rationale.
        yaw = _wrap_angle(theta_r)
        curvature = float(kappa_r)
        velocity = float(np.hypot(s_d, d_d))
        acceleration = float(s_dd)
        return CartesianState(
            x=float(x), y=float(y), yaw=yaw, curvature=curvature,
            velocity=velocity, acceleration=acceleration,
        )

    one_minus_kd = 1.0 - kappa_r * d

    # Reference-frame (tangential, normal) velocity/acceleration
    # components, then rotated into world (x, y) using the reference
    # heading theta_r -- this is exactly the Frenet-frame basis
    # (tangent = (cos theta_r, sin theta_r), left-normal =
    # (-sin theta_r, cos theta_r)) evaluated with time derivatives of
    # the Frenet coordinates. This is a direct, first-principles
    # application of the Frenet-Serret frame (position derivative
    # decomposes into tangential progress s_d*(1-kappa_r*d) along the
    # rotating tangent, plus lateral rate d_d along the co-rotating
    # normal -- see e.g. Werling et al. Eq. 1-4), rather than the
    # higher-order curvature/acceleration formulas which are easy to
    # mis-transcribe; those are instead recovered below via the exact
    # planar (vx, vy, ax, ay) identities.
    tangential_v = s_d * one_minus_kd
    normal_v = d_d

    vx = tangential_v * cos_r - normal_v * sin_r
    vy = tangential_v * sin_r + normal_v * cos_r

    # d/dt of the tangential/normal decomposition. theta_r itself
    # changes with time as the arc-length parameter advances:
    # d(theta_r)/dt = kappa_r * s_d.
    theta_r_dot = kappa_r * s_d
    one_minus_kd_dot = -(kappa_r_dot * s_d * d + kappa_r * d_d)
    tangential_a = s_dd * one_minus_kd + s_d * one_minus_kd_dot
    normal_a = d_dd

    # Rotate (tangential_a, normal_a) into world frame, INCLUDING the
    # centripetal/Coriolis-style cross terms from the frame's own
    # rotation (d/dt of cos_r, sin_r is -sin_r*theta_r_dot,
    # cos_r*theta_r_dot respectively) -- i.e. differentiate
    # vx = tangential_v*cos_r - normal_v*sin_r fully via the product
    # rule.
    ax = (
        tangential_a * cos_r - tangential_v * sin_r * theta_r_dot
        - normal_a * sin_r - normal_v * cos_r * theta_r_dot
    )
    ay = (
        tangential_a * sin_r + tangential_v * cos_r * theta_r_dot
        + normal_a * cos_r - normal_v * sin_r * theta_r_dot
    )

    v = float(np.hypot(vx, vy))
    yaw = _wrap_angle(float(np.arctan2(vy, vx)))

    if v < LOW_SPEED_THRESHOLD_MPS:
        # Velocity vector itself is tiny even though s_d wasn't (can
        # happen right at a sign-change/cusp); fall back to the same
        # geometric-heading behavior for curvature/accel to avoid
        # dividing by a near-zero v^3.
        curvature = float(kappa_r)
        acceleration = float(s_dd)
    else:
        curvature = float((vx * ay - vy * ax) / (v ** 3))
        acceleration = float((vx * ax + vy * ay) / v)

    return CartesianState(
        x=float(x), y=float(y), yaw=yaw, curvature=curvature,
        velocity=v, acceleration=acceleration,
    )


def cartesian_to_frenet(
    x: float, y: float, yaw: float, velocity: float, acceleration: float,
    reference: ReferenceLine,
) -> FrenetState:
    """Converts a Cartesian state to Frenet, projecting onto
    ``reference`` to find ``s`` and the signed lateral offset ``d``,
    then inverting the same tangential/normal decomposition used by
    ``frenet_to_cartesian``.

    Uses ``ReferenceLine.project`` (Stage 3-A, itself built on the
    already-proven ``project_point_to_polyline_signed``) to get
    ``s``/``d`` directly -- consistent by construction with this
    module's sign convention (both ultimately derive from the same
    ``lane_geometry.py`` "positive = left" convention).

    ``acceleration`` convention: SCALAR, tangential to the current
    heading (``d(velocity)/dt`` along the direction of travel, i.e.
    ``(ax, ay) = acceleration * (cos(yaw), sin(yaw))``), NOT a full 2D
    Cartesian acceleration vector. This matches Waymax's own
    ``(acceleration_mps2, steering_curvature)`` action convention
    (locked Phase 3 design decision #1, see
    ``docs/phase3/OVERNIGHT_PROGRESS.md``) and, by construction, is
    exactly what ``frenet_to_cartesian``'s ``acceleration`` output
    field means too (it is likewise defined as the tangential/speed-
    rate component only, via ``(vx*ax + vy*ay) / v``) -- so the two
    functions are round-trip-consistent by definition, not by
    coincidence. A caller with an independently-measured lateral
    (centripetal) acceleration component should not pass it here; it
    is not part of this transform's input contract.
    """

    projection = reference.project(x, y)
    s = float(projection["arc_length_m"])
    d = float(projection["lateral_distance_m"])

    ref = reference.interpolate(s)
    theta_r = ref.yaw
    kappa_r = ref.curvature
    kappa_r_dot = ref.curvature_derivative
    cos_r, sin_r = np.cos(theta_r), np.sin(theta_r)

    delta_theta = _wrap_angle(yaw - theta_r)
    vx = velocity * np.cos(yaw)
    vy = velocity * np.sin(yaw)

    # Invert the world-frame rotation to recover tangential/normal
    # velocity components (inverse of the rotation used in
    # frenet_to_cartesian: tangential = vx*cos_r + vy*sin_r,
    # normal = -vx*sin_r + vy*cos_r).
    tangential_v = vx * cos_r + vy * sin_r
    normal_v = -vx * sin_r + vy * cos_r

    d_d = float(normal_v)

    if velocity < LOW_SPEED_THRESHOLD_MPS:
        # Low-speed fallback mirrors frenet_to_cartesian's: do not
        # divide tangential_v by (1 - kappa_r*d) since both can be
        # ill-conditioned together at near-zero speed. s_d is taken
        # directly as the tangential component (accurate to first
        # order and finite by construction), s_dd as the scalar
        # acceleration, d_dd as 0 (no reliable second-order lateral
        # information at near-zero speed without dividing by a small
        # number).
        s_d = float(tangential_v)
        s_dd = float(acceleration)
        d_dd = 0.0
        return FrenetState(s=s, s_d=s_d, s_dd=s_dd, d=d, d_d=d_d, d_dd=d_dd)

    one_minus_kd = 1.0 - kappa_r * d
    # Guard the geometric (1 - kappa_r*d) singularity too (see module
    # docstring: unlikely for typical highway-merge d ranges, but
    # guarded defensively so this never silently produces inf/NaN).
    if abs(one_minus_kd) < 1e-3:
        one_minus_kd = 1e-3 if one_minus_kd >= 0.0 else -1e-3

    s_d = float(tangential_v / one_minus_kd)

    ax = acceleration * np.cos(yaw)
    ay = acceleration * np.sin(yaw)
    tangential_a = ax * cos_r + ay * sin_r
    normal_a = -ax * sin_r + ay * cos_r

    d_dd = float(normal_a)

    theta_r_dot = kappa_r * s_d
    one_minus_kd_dot = -(kappa_r_dot * s_d * d + kappa_r * d_d)
    # tangential_a (as reconstructed from world ax,ay via the inverse
    # rotation above) already reflects the true tangential
    # acceleration component INCLUDING the frame-rotation
    # cross-terms; recovering s_dd from tangential_a requires
    # inverting tangential_a = s_dd*one_minus_kd + s_d*one_minus_kd_dot.
    s_dd = float((tangential_a - s_d * one_minus_kd_dot) / one_minus_kd)

    return FrenetState(s=s, s_d=s_d, s_dd=s_dd, d=d, d_d=d_d, d_dd=d_dd)
