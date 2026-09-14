"""Stage 3-B tests for src/planning/frenet_types.py.

These are simple dataclasses; tests just confirm construction,
immutability, and the documented default values for the Stage 3-C
"not yet evaluated" fields.
"""

import numpy as np
import pytest

from src.planning.frenet_types import CartesianTrajectory, FrenetPath, FrenetState


class TestFrenetState:
    def test_construction_and_fields(self):
        state = FrenetState(s=1.0, s_d=2.0, s_dd=3.0, d=4.0, d_d=5.0, d_dd=6.0)
        assert state.s == 1.0
        assert state.s_d == 2.0
        assert state.s_dd == 3.0
        assert state.d == 4.0
        assert state.d_d == 5.0
        assert state.d_dd == 6.0

    def test_frozen(self):
        state = FrenetState(s=1.0, s_d=0.0, s_dd=0.0, d=0.0, d_d=0.0, d_dd=0.0)
        with pytest.raises(Exception):
            state.s = 2.0


class TestFrenetPath:
    def test_default_valid_true_reason_none(self):
        n = 5
        arr = np.zeros(n)
        path = FrenetPath(t=arr, s=arr, s_d=arr, s_dd=arr, d=arr, d_d=arr, d_dd=arr)
        assert path.valid is True
        assert path.rejection_reason is None
        assert path.s_ddd is None

    def test_explicit_invalid(self):
        n = 5
        arr = np.zeros(n)
        path = FrenetPath(
            t=arr, s=arr, s_d=arr, s_dd=arr, d=arr, d_d=arr, d_dd=arr,
            valid=False, rejection_reason="test reason",
        )
        assert path.valid is False
        assert path.rejection_reason == "test reason"

    def test_optional_jerk_array(self):
        n = 5
        arr = np.zeros(n)
        jerk = np.ones(n)
        path = FrenetPath(
            t=arr, s=arr, s_d=arr, s_dd=arr, d=arr, d_d=arr, d_dd=arr, s_ddd=jerk,
        )
        assert np.array_equal(path.s_ddd, jerk)


class TestCartesianTrajectory:
    def test_construction(self):
        n = 4
        arr = np.arange(n, dtype=np.float64)
        traj = CartesianTrajectory(
            t=arr, x=arr, y=arr, yaw=arr, curvature=arr, velocity=arr,
            acceleration=arr,
        )
        assert traj.t.shape == (n,)
        assert np.array_equal(traj.x, arr)
