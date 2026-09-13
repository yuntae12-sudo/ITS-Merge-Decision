"""Stage 3-B tests: quintic/quartic polynomial trajectory primitives."""

import numpy as np
import pytest
from scipy.integrate import quad

from src.planning.polynomial import QuarticPolynomial, QuinticPolynomial

TOL = 1e-8

# Jerk-cost cross-check tolerance: analytical closed-form integration
# is exact (up to float64 rounding); scipy.integrate.quad on a smooth
# low-degree polynomial converges to essentially machine precision as
# well, so a fairly tight relative tolerance is appropriate here. This
# is looser than TOL because quad's adaptive quadrature and the
# closed-form sum accumulate rounding differently, but both are
# well within 1e-6 relative for these smooth low-degree integrands.
JERK_COST_RTOL = 1e-6
JERK_COST_ATOL = 1e-9


def _random_quintic_bounds(rng):
    return dict(
        p0=rng.uniform(-50, 50),
        v0=rng.uniform(-20, 20),
        a0_acc=rng.uniform(-5, 5),
        pT=rng.uniform(-50, 50),
        vT=rng.uniform(-20, 20),
        aT=rng.uniform(-5, 5),
        T=rng.uniform(0.5, 8.0),
    )


def _random_quartic_bounds(rng):
    return dict(
        p0=rng.uniform(-50, 50),
        v0=rng.uniform(-20, 20),
        a0_acc=rng.uniform(-5, 5),
        vT=rng.uniform(-20, 20),
        aT=rng.uniform(-5, 5),
        T=rng.uniform(0.5, 8.0),
    )


class TestQuinticPolynomial:
    def test_boundary_conditions_exact(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            bounds = _random_quintic_bounds(rng)
            poly = QuinticPolynomial.solve(**bounds)
            T = bounds["T"]

            assert poly.position(0.0) == pytest.approx(bounds["p0"], abs=TOL)
            assert poly.velocity(0.0) == pytest.approx(bounds["v0"], abs=TOL)
            assert poly.acceleration(0.0) == pytest.approx(bounds["a0_acc"], abs=TOL)

            assert poly.position(T) == pytest.approx(bounds["pT"], abs=TOL)
            assert poly.velocity(T) == pytest.approx(bounds["vT"], abs=TOL)
            assert poly.acceleration(T) == pytest.approx(bounds["aT"], abs=TOL)

    def test_rejects_nonpositive_T(self):
        with pytest.raises(ValueError):
            QuinticPolynomial.solve(0, 0, 0, 1, 1, 1, T=0.0)
        with pytest.raises(ValueError):
            QuinticPolynomial.solve(0, 0, 0, 1, 1, 1, T=-1.0)

    def test_array_evaluation(self):
        poly = QuinticPolynomial.solve(0, 1, 0, 10, 1, 0, T=5.0)
        t = np.linspace(0, 5, 11)
        pos = poly.position(t)
        assert pos.shape == t.shape
        assert np.all(np.isfinite(pos))

    def test_jerk_cost_matches_numerical_integration(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            bounds = _random_quintic_bounds(rng)
            poly = QuinticPolynomial.solve(**bounds)
            analytical = poly.jerk_cost()
            numerical, _ = quad(lambda t: poly.jerk(t) ** 2, 0.0, bounds["T"])
            assert analytical == pytest.approx(
                numerical, rel=JERK_COST_RTOL, abs=JERK_COST_ATOL
            )


class TestQuarticPolynomial:
    def test_boundary_conditions_exact(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            bounds = _random_quartic_bounds(rng)
            poly = QuarticPolynomial.solve(**bounds)
            T = bounds["T"]

            assert poly.position(0.0) == pytest.approx(bounds["p0"], abs=TOL)
            assert poly.velocity(0.0) == pytest.approx(bounds["v0"], abs=TOL)
            assert poly.acceleration(0.0) == pytest.approx(bounds["a0_acc"], abs=TOL)

            # Terminal position is free / not constrained.
            assert poly.velocity(T) == pytest.approx(bounds["vT"], abs=TOL)
            assert poly.acceleration(T) == pytest.approx(bounds["aT"], abs=TOL)

    def test_rejects_nonpositive_T(self):
        with pytest.raises(ValueError):
            QuarticPolynomial.solve(0, 0, 0, 1, 1, T=0.0)
        with pytest.raises(ValueError):
            QuarticPolynomial.solve(0, 0, 0, 1, 1, T=-2.0)

    def test_jerk_cost_matches_numerical_integration(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            bounds = _random_quartic_bounds(rng)
            poly = QuarticPolynomial.solve(**bounds)
            analytical = poly.jerk_cost()
            numerical, _ = quad(lambda t: poly.jerk(t) ** 2, 0.0, bounds["T"])
            assert analytical == pytest.approx(
                numerical, rel=JERK_COST_RTOL, abs=JERK_COST_ATOL
            )
