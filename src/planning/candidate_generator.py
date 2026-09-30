"""Phase 3 Stage 3-C: per-BehaviorAction Frenet trajectory generation.

Turns one already-decided ``BehaviorAction`` + ``BehaviorObjective``
(Stage 2, frozen -- see ``src.environment.behavior_action``) into ONE
well-defined candidate ``FrenetPath`` (Stage 3-B type), evaluated
against whichever ``ReferenceLine`` (Stage 3-A) is appropriate for
that action.

This module deliberately generates exactly ONE candidate per action,
not a family to search over: each of the four actions already has one
well-defined terminal condition (a fixed lateral target, and either a
fixed terminal speed or a fixed terminal stop position) -- there is no
"which of several trajectories is best" decision to make here, only
"what is the one trajectory that executes this already-chosen
action". Feasibility/collision REJECTION of that one candidate is
``candidate_evaluator.py``'s job, not this module's.

Hard fairness constraint (Stage 3-0 audit, locked): nothing in this
module may decide WHETHER an action should be taken, only HOW to
execute an action that has already been chosen upstream. In
particular:
  - MERGE is generated exactly as MERGE asks (target-frame lateral
    convergence, current-lead-relative or cruise longitudinal
    objective) -- there is no gap-safety check here, and no
    fallback that silently reshapes a MERGE candidate into KEEP/
    FOLLOW/STOP geometry. If the ONE generated MERGE candidate later
    fails a kinematic/collision feasibility check
    (``candidate_evaluator.py``), the caller gets an explicit failure
    status back, never a silently substituted trajectory.
  - FOLLOW/MERGE longitudinal profiles consume gap/relative-speed
    numbers the CALLER supplies (sourced from the already-causal 14D
    observation's own front-vehicle selection --
    ``src.scenarios.scenario_features``) -- this module never
    reselects a different leader or re-derives gap/TTC itself.
"""

import dataclasses
from typing import Optional

import numpy as np

from src.planning.frenet_transform import cartesian_to_frenet
from src.planning.frenet_types import FrenetPath, FrenetState
from src.planning.polynomial import QuarticPolynomial, QuinticPolynomial
from src.planning.reference import ReferenceLine

# Terminal lateral target for every action: centered on the ego's
# CURRENT active reference (source lane for KEEP/FOLLOW/STOP, target
# lane for MERGE -- see frenet_planner.py for which ReferenceLine is
# passed in for which action). Not configurable: "return to lane
# center" is a geometric constant of lane-following, not a tunable.
LATERAL_TARGET_D_M = 0.0
LATERAL_TARGET_D_DOT = 0.0
LATERAL_TARGET_D_DDOT = 0.0


@dataclasses.dataclass(frozen=True)
class CandidateGenerationResult:
    """Either a generated ``FrenetPath`` + the ``ReferenceLine`` it is
    expressed against, or an explicit failure (never both)."""

    frenet_path: Optional[FrenetPath]
    reference: Optional[ReferenceLine]
    ok: bool
    error: Optional[str] = None
    # Diagnostics, always populated when ok=True:
    stop_target_clamped: bool = False
    terminal_frenet_state: Optional[FrenetState] = None


def _sample_times(horizon_s: float, dt_s: float) -> np.ndarray:
    n_steps = max(int(round(horizon_s / dt_s)), 1)
    return np.linspace(0.0, n_steps * dt_s, n_steps + 1)


def _lateral_quintic(
    d0: float, d_d0: float, d_dd0: float, horizon_s: float
) -> QuinticPolynomial:
    """Lateral quintic used identically by KEEP/FOLLOW/STOP (source
    frame, d_target=0) and MERGE (target frame, d_target=0) -- the
    terminal target is the same constant in both cases; only the
    reference frame (and hence d0/d_d0/d_dd0's meaning) differs,
    decided entirely by the caller's choice of which ReferenceLine to
    project the current state into before calling this."""

    return QuinticPolynomial.solve(
        p0=d0, v0=d_d0, a0_acc=d_dd0,
        pT=LATERAL_TARGET_D_M, vT=LATERAL_TARGET_D_DOT, aT=LATERAL_TARGET_D_DDOT,
        T=horizon_s,
    )


MAX_TERMINAL_SPEED_SEARCH_ITERATIONS = 40
# Bisection iteration cap for _solve_feasible_terminal_speed. 2^-40 of
# any physically reasonable initial search bracket (bounded by
# NOMINAL_CRUISE_SPEED_MPS-scale speeds, i.e. at most ~tens of m/s) is
# far below floating-point-meaningful precision, so this is a
# deterministic, generous cap, never actually exhausted in practice
# (see TERMINAL_SPEED_SEARCH_TOLERANCE_MPS below for the real stopping
# condition).
TERMINAL_SPEED_SEARCH_TOLERANCE_MPS = 1e-6


@dataclasses.dataclass(frozen=True)
class FeasibleTerminalSpeedResult:
    """Diagnostic-only (read-only) record of how
    ``_solve_feasible_terminal_speed`` arrived at the terminal speed
    actually used for one quartic velocity-keeping candidate. Never
    consulted by any behavior/training/reward decision -- surfaced
    purely for observability (see frenet_planner.py's diagnostics
    dict)."""

    requested_reference_speed_mps: float
    feasible_terminal_speed_mps: float
    terminal_speed_was_limited: bool
    terminal_speed_limit_reason: Optional[str]
    original_predicted_max_abs_accel_mps2: float
    final_predicted_max_abs_accel_mps2: float
    acceleration_limit_mps2: float


def _exact_max_abs_longitudinal_accel(
    s_d0: float, s_dd0: float, vT: float, horizon_s: float,
) -> float:
    """Exact continuous-time max|acceleration(t)| over t in [0, T] for
    the quartic velocity-keeping profile solved with
    (v0=s_d0, a0_acc=s_dd0, vT=vT, aT=0.0, T=horizon_s).

    acceleration(t) = 2*a2 + 6*a3*t + 12*a4*t^2 is QUADRATIC in t (see
    QuarticPolynomial.acceleration), so its exact extrema over a closed
    interval are just the two endpoints plus (if it falls inside the
    interval) the single vertex t* = -a3/(4*a4) -- no sampling/search
    over t is needed, only over the 3 candidate points below."""

    poly = QuarticPolynomial.solve(
        p0=0.0, v0=s_d0, a0_acc=s_dd0, vT=vT, aT=0.0, T=horizon_s,
    )
    a2, a3, a4 = poly.coeffs[2], poly.coeffs[3], poly.coeffs[4]

    candidate_times = [0.0, horizon_s]
    if abs(a4) > 1e-12:
        t_star = -a3 / (4.0 * a4)
        if 0.0 <= t_star <= horizon_s:
            candidate_times.append(t_star)

    return max(abs(2.0 * a2 + 6.0 * a3 * t + 12.0 * a4 * t * t) for t in candidate_times)


def _solve_feasible_terminal_speed(
    s_d0: float, s_dd0: float, desired_vT: float, horizon_s: float,
    max_longitudinal_accel_mps2: float,
) -> FeasibleTerminalSpeedResult:
    """Finds the terminal speed closest to ``desired_vT`` (on the
    segment between ``s_d0`` and ``desired_vT``) whose quartic
    velocity-keeping profile satisfies
    ``max|acceleration(t)| <= max_longitudinal_accel_mps2`` over the
    full continuous horizon -- handles BOTH speed-up
    (desired_vT > s_d0) and speed-down (desired_vT < s_d0) symmetrically.

    Deterministic bisection (not a general-purpose optimizer): for
    fixed (s_d0, s_dd0, horizon_s), ``_exact_max_abs_longitudinal_accel``
    is monotonically non-decreasing as vT moves AWAY from s_d0 in
    either direction (verified: the quartic's t=0/t=T/vertex-extrema
    accelerations are each affine in vT for fixed s_d0/s_dd0/horizon_s,
    so the max of their absolute values is a convex, unimodal function
    of vT with its minimum at/near vT=s_d0) -- so a simple bisection
    between s_d0 (always the least-demanding endpoint) and desired_vT
    converges to the unique feasible boundary point on that segment.
    """

    original_accel = _exact_max_abs_longitudinal_accel(s_d0, s_dd0, desired_vT, horizon_s)
    if original_accel <= max_longitudinal_accel_mps2:
        return FeasibleTerminalSpeedResult(
            requested_reference_speed_mps=desired_vT,
            feasible_terminal_speed_mps=desired_vT,
            terminal_speed_was_limited=False,
            terminal_speed_limit_reason=None,
            original_predicted_max_abs_accel_mps2=original_accel,
            final_predicted_max_abs_accel_mps2=original_accel,
            acceleration_limit_mps2=max_longitudinal_accel_mps2,
        )

    # desired_vT is infeasible; s_d0 itself (zero net speed change) is
    # always at least as feasible as any point further from it, so the
    # feasible boundary lies strictly between s_d0 and desired_vT.
    lo, hi = s_d0, desired_vT  # lo: feasible side, hi: infeasible side
    for _ in range(MAX_TERMINAL_SPEED_SEARCH_ITERATIONS):
        if abs(hi - lo) <= TERMINAL_SPEED_SEARCH_TOLERANCE_MPS:
            break
        mid = 0.5 * (lo + hi)
        mid_accel = _exact_max_abs_longitudinal_accel(s_d0, s_dd0, mid, horizon_s)
        if mid_accel <= max_longitudinal_accel_mps2:
            lo = mid
        else:
            hi = mid

    feasible_vT = lo
    final_accel = _exact_max_abs_longitudinal_accel(s_d0, s_dd0, feasible_vT, horizon_s)
    reason = (
        "requested reference speed exceeds max_longitudinal_accel_mps2 "
        f"over trajectory_horizon_s={horizon_s}s from current "
        f"s_d0={s_d0:.3f} m/s (speed-up)" if desired_vT > s_d0 else
        "requested reference speed exceeds max_longitudinal_accel_mps2 "
        f"over trajectory_horizon_s={horizon_s}s from current "
        f"s_d0={s_d0:.3f} m/s (speed-down)"
    )
    return FeasibleTerminalSpeedResult(
        requested_reference_speed_mps=desired_vT,
        feasible_terminal_speed_mps=feasible_vT,
        terminal_speed_was_limited=True,
        terminal_speed_limit_reason=reason,
        original_predicted_max_abs_accel_mps2=original_accel,
        final_predicted_max_abs_accel_mps2=final_accel,
        acceleration_limit_mps2=max_longitudinal_accel_mps2,
    )


def _longitudinal_quartic_velocity_keeping(
    s0: float, s_d0: float, s_dd0: float,
    target_speed_mps: float, horizon_s: float,
    max_longitudinal_accel_mps2: Optional[float] = None,
) -> "tuple[QuarticPolynomial, Optional[FeasibleTerminalSpeedResult]]":
    """KEEP/FOLLOW/MERGE-with-no-lead longitudinal profile: converge
    toward a target speed, terminal position free (Stage 3-B's own
    quartic primitive, used exactly as documented for velocity-
    keeping).

    When ``max_longitudinal_accel_mps2`` is given, the ACTUAL terminal
    speed used is shaped to the closest physically feasible speed (see
    ``_solve_feasible_terminal_speed``) -- ``target_speed_mps`` itself
    (the caller's desired/requested speed, i.e.
    ``BehaviorObjective.reference_speed_mps``) is never mutated by this
    function; only the trajectory generated FROM it is shaped. When
    ``None`` (default, e.g. STOP's own quintic path never calls this),
    the terminal speed is used exactly as given, matching this
    function's original unconditional behavior."""

    if max_longitudinal_accel_mps2 is None:
        return QuarticPolynomial.solve(
            p0=s0, v0=s_d0, a0_acc=s_dd0,
            vT=target_speed_mps, aT=0.0, T=horizon_s,
        ), None

    feasibility_result = _solve_feasible_terminal_speed(
        s_d0, s_dd0, target_speed_mps, horizon_s, max_longitudinal_accel_mps2,
    )
    return QuarticPolynomial.solve(
        p0=s0, v0=s_d0, a0_acc=s_dd0,
        vT=feasibility_result.feasible_terminal_speed_mps, aT=0.0, T=horizon_s,
    ), feasibility_result


def generate_keep_candidate(
    ego_frenet: FrenetState,
    reference_speed_mps: float,
    horizon_s: float,
    dt_s: float,
    max_longitudinal_accel_mps2: Optional[float] = None,
) -> "tuple[FrenetPath, Optional[FeasibleTerminalSpeedResult]]":
    """KEEP: lateral quintic to d=0 in the SOURCE frame; longitudinal
    quartic velocity-keeping toward ``reference_speed_mps`` (terminal
    position free -- no fixed cruise target position exists).

    ``reference_speed_mps`` is never mutated (see
    ``_longitudinal_quartic_velocity_keeping``'s docstring) -- when
    ``max_longitudinal_accel_mps2`` is given, the returned
    ``FrenetPath`` is shaped toward the closest physically feasible
    terminal speed instead, and the second return value documents how.
    """

    longitudinal_poly, feasibility_result = _longitudinal_quartic_velocity_keeping(
        ego_frenet.s, ego_frenet.s_d, ego_frenet.s_dd,
        reference_speed_mps, horizon_s, max_longitudinal_accel_mps2,
    )
    frenet_path = _build_frenet_path(
        lateral_poly=_lateral_quintic(
            ego_frenet.d, ego_frenet.d_d, ego_frenet.d_dd, horizon_s
        ),
        longitudinal_poly=longitudinal_poly,
        horizon_s=horizon_s, dt_s=dt_s,
    )
    return frenet_path, feasibility_result


def generate_follow_or_merge_candidate(
    ego_frenet: FrenetState,
    reference_speed_mps: float,
    horizon_s: float,
    dt_s: float,
    max_longitudinal_accel_mps2: Optional[float] = None,
) -> "tuple[FrenetPath, Optional[FeasibleTerminalSpeedResult]]":
    """FOLLOW (source frame) and MERGE-longitudinal (target frame):
    identical shape to KEEP's longitudinal profile -- a quartic
    velocity-keeping trajectory toward ``reference_speed_mps``.

    ``reference_speed_mps`` here is whatever
    ``BehaviorExecutor.compute_objective`` already decided (a time-gap
    -style follow speed against the current causal lead, or the
    nominal cruise speed if no lead/objective calls for it) -- this
    function does not re-derive that speed, only executes it (the
    DESIRED speed is unchanged; only the generated trajectory is
    feasibility-shaped, exactly as ``generate_keep_candidate``
    documents). Lateral handling is identical to KEEP's (quintic to
    d=0), just expressed in whichever frame the caller has already
    projected ``ego_frenet`` into (source for FOLLOW, target for MERGE
    -- see frenet_planner.py).
    """

    return generate_keep_candidate(
        ego_frenet, reference_speed_mps, horizon_s, dt_s, max_longitudinal_accel_mps2,
    )


MIN_STOP_HORIZON_S = 1.0
# Floor on the longitudinal stop horizon (see below): prevents a
# near-zero natural stopping time (ego already nearly stationary) from
# collapsing the quintic's T toward 0, which would make the boundary
# -value solve ill-conditioned. 1.0s is far below any realistic
# highway-speed stopping time at comfortable_deceleration_mps2 = 2.0
# (e.g. even a modest 2 m/s current speed implies a natural stop time
# of 1.0s exactly), so this floor only ever binds for an already
# -near-stationary ego, where "how long the stop trajectory nominally
# takes" is close to moot anyway.


def generate_stop_candidate(
    ego_frenet: FrenetState,
    comfortable_deceleration_mps2: float,
    horizon_s: float,
    dt_s: float,
    reference: ReferenceLine,
) -> CandidateGenerationResult:
    """STOP: lateral quintic to d=0 in the source frame (using the
    caller's ``horizon_s``, same as every other action's lateral
    profile -- lateral convergence has no dependency on the stopping
    physics below); longitudinal QUINTIC (fixed terminal position,
    unlike KEEP/FOLLOW/MERGE's quartic) to a causal stop target:

        braking_distance = v_current^2 / (2 * comfortable_deceleration)
        s_stop = s_current + braking_distance

    terminal [s_stop, v=0, a=0]. If s_stop exceeds the reference's
    valid domain, it is CLAMPED to the reference's own extent and
    ``stop_target_clamped=True`` is set (visible diagnostic, not a
    silent change) -- this is geometry-domain handling (the reference
    line simply does not extend that far), not a behavior decision.

    Longitudinal horizon: the physically NATURAL stopping duration
    ``v_current / comfortable_deceleration`` (floored at
    ``MIN_STOP_HORIZON_S``), NOT the caller's generic ``horizon_s``.
    A constant-deceleration stop from speed v0 to 0 at exactly
    ``comfortable_deceleration_mps2`` takes exactly
    ``v0 / comfortable_deceleration_mps2`` seconds; forcing that same
    stop into an unrelated fixed horizon (e.g. the other actions'
    generic 3.0s trajectory_horizon_s) makes the quintic boundary
    -value problem badly matched to the requested deceleration whenever
    the natural stopping time differs from that fixed horizon --
    concretely, a quintic solved to land exactly on [s_stop, 0, 0] at
    a horizon shorter than the natural stopping time is forced into a
    sharp overshoot-and-correct S-curve (verified directly: v0=12 m/s
    forced into a 3.0s horizon peaks at approximately -15.7 m/s^2,
    over 2.5x the 6.0 m/s^2 limit, whereas the same stop given its
    natural ~6.0s horizon peaks at exactly 3.0 m/s^2, i.e. the
    constant-deceleration profile itself, well within bounds). Using
    the natural stopping time is not a behavior decision (it does not
    change WHAT the stop target is, only how long the already-fixed
    terminal condition is given to reach), and keeps STOP's terminal
    condition physically consistent with the comfortable_deceleration
    constant that defines it in the first place.
    """

    v_current = ego_frenet.s_d
    braking_distance = (v_current ** 2) / (2.0 * comfortable_deceleration_mps2)
    s_stop = ego_frenet.s + braking_distance

    stop_target_clamped = False
    max_s = float(reference.s[-1])
    min_s = float(reference.s[0])
    if s_stop > max_s:
        s_stop = max_s
        stop_target_clamped = True
    elif s_stop < min_s:
        s_stop = min_s
        stop_target_clamped = True

    longitudinal_horizon_s = max(
        abs(v_current) / comfortable_deceleration_mps2, MIN_STOP_HORIZON_S
    )

    lateral_poly = _lateral_quintic(
        ego_frenet.d, ego_frenet.d_d, ego_frenet.d_dd, horizon_s
    )
    longitudinal_poly = QuinticPolynomial.solve(
        p0=ego_frenet.s, v0=ego_frenet.s_d, a0_acc=ego_frenet.s_dd,
        pT=s_stop, vT=0.0, aT=0.0, T=longitudinal_horizon_s,
    )

    # The reported trajectory spans the LONGER of the two horizons
    # (the caller's lateral horizon_s, and the stop's own natural
    # longitudinal_horizon_s), so the full stop maneuver is always
    # represented even when it takes longer than the generic
    # cross-action horizon_s. Each polynomial is evaluated only up to
    # its OWN horizon and then held at its converged terminal state
    # for any remaining samples (both terminal conditions are already
    # exactly [target, 0, 0]-style fixed points, so holding is exact,
    # not an approximation -- a quintic's OWN polynomial curve beyond
    # its solved T is not physically meaningful and must not be
    # evaluated there).
    common_horizon_s = max(horizon_s, longitudinal_horizon_s)
    t = _sample_times(common_horizon_s, dt_s)

    t_lateral = np.minimum(t, horizon_s)
    t_longitudinal = np.minimum(t, longitudinal_horizon_s)

    frenet_path = FrenetPath(
        t=t,
        s=longitudinal_poly.position(t_longitudinal),
        s_d=longitudinal_poly.velocity(t_longitudinal),
        s_dd=longitudinal_poly.acceleration(t_longitudinal),
        d=lateral_poly.position(t_lateral),
        d_d=lateral_poly.velocity(t_lateral),
        d_dd=lateral_poly.acceleration(t_lateral),
        s_ddd=longitudinal_poly.jerk(t_longitudinal),
    )

    return CandidateGenerationResult(
        frenet_path=frenet_path,
        reference=reference,
        ok=True,
        stop_target_clamped=stop_target_clamped,
        terminal_frenet_state=FrenetState(
            s=s_stop, s_d=0.0, s_dd=0.0,
            d=LATERAL_TARGET_D_M, d_d=LATERAL_TARGET_D_DOT, d_dd=LATERAL_TARGET_D_DDOT,
        ),
    )


def project_cartesian_to_frame(
    x: float, y: float, yaw: float, speed_mps: float,
    reference: ReferenceLine,
) -> FrenetState:
    """Projects the ego's CURRENT Cartesian state into ``reference``'s
    Frenet frame. Used by ``frenet_planner.py`` to obtain
    ``ego_frenet`` for whichever frame an action needs (source frame
    for KEEP/FOLLOW/STOP; TARGET frame for MERGE, per the task's
    preferred approach (b): generate the MERGE candidate directly in
    the target reference's own Frenet frame rather than inventing a
    blended reference).

    ``acceleration`` is passed as 0.0 -- the planner's own request
    type does not carry a measured current scalar acceleration (Stage
    3-C's request surface only requires x/y/yaw/speed, matching what
    the frozen 14D observation and Waymax sim state expose plainly);
    this only affects the derived s_dd/d_dd terms of the CURRENT
    Frenet state used as the boundary condition, and both KEEP/FOLLOW/
    MERGE/STOP already treat "current acceleration" as an initial
    condition to converge away from, not a target -- a 0.0 starting
    assumption is standard and conservative here (see
    ``frenet_planner.py``'s request docstring for exactly what an
    ego_acceleration_mps2 field, if ever added later, would flow to).
    """

    return cartesian_to_frenet(
        x=x, y=y, yaw=yaw, velocity=speed_mps, acceleration=0.0,
        reference=reference,
    )


def _build_frenet_path(
    lateral_poly: QuinticPolynomial,
    longitudinal_poly,
    horizon_s: float,
    dt_s: float,
) -> FrenetPath:
    t = _sample_times(horizon_s, dt_s)
    return FrenetPath(
        t=t,
        s=longitudinal_poly.position(t),
        s_d=longitudinal_poly.velocity(t),
        s_dd=longitudinal_poly.acceleration(t),
        d=lateral_poly.position(t),
        d_d=lateral_poly.velocity(t),
        d_dd=lateral_poly.acceleration(t),
        s_ddd=longitudinal_poly.jerk(t),
    )
