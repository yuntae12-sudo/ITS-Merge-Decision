"""Phase 3 Stage 3-B: 1D quintic/quartic polynomial trajectory
primitives.

These are the standard Werling et al.-style polynomial primitives used
independently for lateral ``d(t)`` and longitudinal ``s(t)`` in a
Frenet-frame planner:

  - ``QuinticPolynomial``: fully-constrained boundary value problem --
    initial [position, velocity, acceleration] at t=0 AND terminal
    [position, velocity, acceleration] at t=T. Used where a target
    terminal position is known (e.g. lateral lane-centering, or a
    longitudinal STOP target position).
  - ``QuarticPolynomial``: "velocity-keeping" boundary value problem --
    initial [position, velocity, acceleration] at t=0, but only
    terminal [velocity, acceleration] at t=T (terminal position is
    free). Used where only a target terminal speed matters (e.g.
    KEEP/FOLLOW longitudinal profiles in Stage 3-C).

Both solve their boundary-value problem EXACTLY by setting up the
small linear system implied by the boundary conditions (3x3 for
quintic's higher-order coefficients, 2x2 for quartic's) and solving it
with ``numpy.linalg.solve``. This is deliberately not a hand-expanded
closed-form matrix inverse transcribed from memory -- the boundary
conditions ARE the closed-form problem statement, and solving the
resulting (tiny, well-conditioned for any physically reasonable T> 0)
linear system exactly is mathematically identical to substituting a
memorized inverse formula, without the risk of a transcription error.
The lower-order coefficients (a0, a1, a2) follow directly from the
initial conditions in both cases (p(0)=p0 => a0=p0; p'(0)=v0 => a1=v0;
p''(0)=a0_acc => a2=a0_acc/2), which is standard and not subject to
any solve.

Jerk cost ``J = integral_0^T jerk(t)^2 dt`` is computed in closed form
by symbolically integrating the (polynomial) square of the jerk
polynomial term-by-term (exact for any polynomial), rather than
numerically -- see ``_integrate_squared_polynomial``. Tests
cross-check this against ``scipy.integrate.quad`` numerical
integration across multiple random boundary-condition sets.
"""

import dataclasses

import numpy as np


def _integrate_squared_polynomial(coeffs: np.ndarray, T: float) -> float:
    """Exact closed-form integral_0^T [sum_i coeffs[i] * t^i]^2 dt.

    ``coeffs`` is ordered lowest-degree-first (coeffs[0] is the
    constant term). Squaring the polynomial and integrating
    term-by-term is exact: for p(t) = sum_i c_i t^i,
    p(t)^2 = sum_i sum_j c_i c_j t^(i+j), and
    integral_0^T t^k dt = T^(k+1) / (k+1).
    """

    n = len(coeffs)
    total = 0.0
    for i in range(n):
        if coeffs[i] == 0.0:
            continue
        for j in range(n):
            if coeffs[j] == 0.0:
                continue
            k = i + j
            total += coeffs[i] * coeffs[j] * (T ** (k + 1)) / (k + 1)
    return float(total)


@dataclasses.dataclass(frozen=True)
class QuinticPolynomial:
    """1D quintic (5th-order) polynomial trajectory primitive.

    p(t) = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5

    fully constrained by [position, velocity, acceleration] at both
    t=0 and t=T.
    """

    p0: float
    v0: float
    a0_acc: float
    pT: float
    vT: float
    aT: float
    T: float
    coeffs: np.ndarray  # (6,) lowest-degree-first: [a0..a5]

    @staticmethod
    def solve(
        p0: float, v0: float, a0_acc: float,
        pT: float, vT: float, aT: float, T: float,
    ) -> "QuinticPolynomial":
        if T <= 0.0:
            raise ValueError(f"QuinticPolynomial requires T > 0, got T={T}")

        a0 = p0
        a1 = v0
        a2 = a0_acc / 2.0

        # Remaining boundary conditions at t=T, with the known a0/a1/a2
        # contributions moved to the RHS:
        #   a3 T^3 + a4 T^4 + a5 T^5 = pT - (a0 + a1 T + a2 T^2)
        #   3 a3 T^2 + 4 a4 T^3 + 5 a5 T^4 = vT - (a1 + 2 a2 T)
        #   6 a3 T + 12 a4 T^2 + 20 a5 T^3 = aT - 2 a2
        T2, T3, T4, T5 = T**2, T**3, T**4, T**5
        A = np.array([
            [T3, T4, T5],
            [3 * T2, 4 * T3, 5 * T4],
            [6 * T, 12 * T2, 20 * T3],
        ], dtype=np.float64)
        b = np.array([
            pT - (a0 + a1 * T + a2 * T2),
            vT - (a1 + 2 * a2 * T),
            aT - 2 * a2,
        ], dtype=np.float64)
        a3, a4, a5 = np.linalg.solve(A, b)

        coeffs = np.array([a0, a1, a2, a3, a4, a5], dtype=np.float64)
        return QuinticPolynomial(
            p0=p0, v0=v0, a0_acc=a0_acc, pT=pT, vT=vT, aT=aT, T=T,
            coeffs=coeffs,
        )

    def position(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        a = self.coeffs
        return a[0] + a[1] * t + a[2] * t**2 + a[3] * t**3 + a[4] * t**4 + a[5] * t**5

    def velocity(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        a = self.coeffs
        return a[1] + 2 * a[2] * t + 3 * a[3] * t**2 + 4 * a[4] * t**3 + 5 * a[5] * t**4

    def acceleration(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        a = self.coeffs
        return 2 * a[2] + 6 * a[3] * t + 12 * a[4] * t**2 + 20 * a[5] * t**3

    def jerk(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        a = self.coeffs
        return 6 * a[3] + 24 * a[4] * t + 60 * a[5] * t**2

    def jerk_cost(self) -> float:
        """J = integral_0^T jerk(t)^2 dt, exact closed form."""
        a = self.coeffs
        jerk_coeffs = np.array([6 * a[3], 24 * a[4], 60 * a[5]], dtype=np.float64)
        return _integrate_squared_polynomial(jerk_coeffs, self.T)


@dataclasses.dataclass(frozen=True)
class QuarticPolynomial:
    """1D quartic (4th-order) polynomial trajectory primitive
    ("velocity-keeping": terminal position is free).

    p(t) = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4

    constrained by [position, velocity, acceleration] at t=0 and only
    [velocity, acceleration] at t=T.
    """

    p0: float
    v0: float
    a0_acc: float
    vT: float
    aT: float
    T: float
    coeffs: np.ndarray  # (5,) lowest-degree-first: [a0..a4]

    @staticmethod
    def solve(
        p0: float, v0: float, a0_acc: float,
        vT: float, aT: float, T: float,
    ) -> "QuarticPolynomial":
        if T <= 0.0:
            raise ValueError(f"QuarticPolynomial requires T > 0, got T={T}")

        a0 = p0
        a1 = v0
        a2 = a0_acc / 2.0

        # Remaining boundary conditions at t=T:
        #   3 a3 T^2 + 4 a4 T^3 = vT - (a1 + 2 a2 T)
        #   6 a3 T + 12 a4 T^2 = aT - 2 a2
        T2, T3 = T**2, T**3
        A = np.array([
            [3 * T2, 4 * T3],
            [6 * T, 12 * T2],
        ], dtype=np.float64)
        b = np.array([
            vT - (a1 + 2 * a2 * T),
            aT - 2 * a2,
        ], dtype=np.float64)
        a3, a4 = np.linalg.solve(A, b)

        coeffs = np.array([a0, a1, a2, a3, a4], dtype=np.float64)
        return QuarticPolynomial(
            p0=p0, v0=v0, a0_acc=a0_acc, vT=vT, aT=aT, T=T, coeffs=coeffs,
        )

    def position(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        a = self.coeffs
        return a[0] + a[1] * t + a[2] * t**2 + a[3] * t**3 + a[4] * t**4

    def velocity(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        a = self.coeffs
        return a[1] + 2 * a[2] * t + 3 * a[3] * t**2 + 4 * a[4] * t**3

    def acceleration(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        a = self.coeffs
        return 2 * a[2] + 6 * a[3] * t + 12 * a[4] * t**2

    def jerk(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        a = self.coeffs
        return 6 * a[3] + 24 * a[4] * t

    def jerk_cost(self) -> float:
        """J = integral_0^T jerk(t)^2 dt, exact closed form."""
        a = self.coeffs
        jerk_coeffs = np.array([6 * a[3], 24 * a[4]], dtype=np.float64)
        return _integrate_squared_polynomial(jerk_coeffs, self.T)
