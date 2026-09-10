"""Phase 2 common minimal low-level controller (Stage B-1 Section 4).

Converts one ``BehaviorObjective`` (reference speed + reference lane,
from ``behavior_action.BehaviorExecutor`` -- Stage B-0) plus the
CURRENT simulated ego pose into a physical
``[acceleration_m_s2, steering_curvature]`` command for Waymax's
``InvertibleBicycleModel`` (``normalize_actions=False``, the class's
own default -- confirmed by reading the installed source at revision
a64dfec9be8576b60d9cecc94f406d9812d4a7d0: the action space is
UNNORMALIZED physical units unless ``normalize_actions=True`` is
explicitly passed, contrary to the model docstring's more general
"[-1,1] for RL" framing).

Deliberately minimal (Stage B-1 explicit instruction: no Frenet frame
planner, no LTV-MPC -- confirmed absent from this repository by grep
in Stage B-0). This is a plain proportional (P) speed tracker for
acceleration and a proportional heading/lateral-error tracker for
steering, shared unconditionally by FSM and PPO (this module contains
no policy logic -- it only executes whatever BehaviorObjective it is
given identically regardless of who produced it).

Goal for Stage B-1 (explicitly NOT "best performance"): a stable
common downstream that makes KEEP/FOLLOW/MERGE/STOP produce
physically DIFFERENT ego transitions -- the action-causality property
Stage B-1's tests verify.
"""

import dataclasses

import numpy as np

from src.scenarios.lane_geometry import LanePolyline, project_point_to_polyline

# Acceleration proportional gain: (reference_speed - current_speed) *
# ACCEL_GAIN, clipped to [-MAX_ACCEL, MAX_ACCEL]. Not tuned against
# any performance criterion (Stage B-1 scope) -- chosen only to be
# stable and produce a visibly different acceleration sign/magnitude
# for different reference speeds within one dt=0.1s step.
ACCEL_GAIN = 1.0
MAX_ACCEL_MPS2 = 6.0  # matches InvertibleBicycleModel's default max_accel

# Steering gains. Curvature units (1/m), matching
# InvertibleBicycleModel's default max_steering=0.3.
#
# Real-data smoke testing (Stage B-1) found a fixed HEADING_ERROR_GAIN
# applied directly to a radian heading error routinely saturated
# steering at +-max_steering for several consecutive 0.1s steps,
# producing >20 deg/step yaw swings and a spurious ego-vehicle
# collision -- not a metrics bug (waymax.metrics.overlap confirmed
# genuinely reporting bbox overlap at that point), a real controller
# instability. Waymax's own forward model is
# `delta_yaw = steering * (speed * dt + 0.5*accel*dt**2)`, i.e.
# steering multiplies a SPEED-DEPENDENT term, not heading error
# directly -- so a fixed-gain P controller on heading error is
# systematically too aggressive at typical highway speeds (~15 m/s)
# with dt=0.1s (speed*dt ~= 1.5, so steering=0.3 alone already
# produces ~0.45 rad/step). HEADING_ERROR_GAIN is scaled down
# accordingly so a full max_steering command corresponds to a heading
# error on the order of MAX_STEERING_CURVATURE radians at typical
# speeds, not saturating on much smaller errors. This is a basic
# stability correction, not reward-motivated controller tuning
# (explicitly out of Stage B-1 scope).
HEADING_ERROR_GAIN = 0.3
LATERAL_ERROR_GAIN = 0.02
MAX_STEERING_CURVATURE = 0.3


def _wrap_angle_rad(angle_rad: float) -> float:
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


@dataclasses.dataclass(frozen=True)
class ControllerCommand:
    acceleration_mps2: float
    steering_curvature: float


class LowLevelController:
    """Shared, policy-independent longitudinal + lateral controller.

    One instance is stateless and reusable across steps/episodes --
    it is a pure function of (objective, ego pose/speed, reference
    lane polyline), not an object with per-episode internal state.
    """

    def compute_command(
        self,
        reference_speed_mps: float,
        reference_polyline: LanePolyline,
        ego_x: float,
        ego_y: float,
        ego_yaw: float,
        ego_speed_mps: float,
    ) -> ControllerCommand:
        """Args:
            reference_speed_mps: from BehaviorObjective.
            reference_polyline: the ACTIVE lane to track -- source lane
                for KEEP/FOLLOW/STOP, target lane for MERGE (Stage B-1
                Section 5: this is exactly the "which lane is being
                tracked right now" decision, resolved by the caller
                per BehaviorObjective.reference_lane before this call
                -- this controller has no notion of source/target
                itself, only "the polyline to track").
            ego_x, ego_y, ego_yaw, ego_speed_mps: current SIMULATED ego
                state (never logged/future state).

        Returns:
            ControllerCommand, already clipped to
            InvertibleBicycleModel's default bounds.
        """

        projection = project_point_to_polyline(
            reference_polyline, ego_x, ego_y
        )
        lateral_error_m = projection["lateral_distance_m"]
        heading_error_rad = _wrap_angle_rad(
            projection["heading_rad"] - ego_yaw
        )

        acceleration = np.clip(
            (reference_speed_mps - ego_speed_mps) * ACCEL_GAIN,
            -MAX_ACCEL_MPS2,
            MAX_ACCEL_MPS2,
        )

        # Positive lateral_distance_m = left of the lane's direction
        # (lane_geometry.py convention). A positive lateral error
        # (ego left of centerline) should steer right (negative
        # curvature, by Waymax's convention: new_yaw = yaw + steering
        # * ...) to correct back -- hence the minus sign on the
        # lateral term, added to (not overriding) the heading term.
        steering = np.clip(
            heading_error_rad * HEADING_ERROR_GAIN
            - lateral_error_m * LATERAL_ERROR_GAIN,
            -MAX_STEERING_CURVATURE,
            MAX_STEERING_CURVATURE,
        )

        return ControllerCommand(
            acceleration_mps2=float(acceleration),
            steering_curvature=float(steering),
        )
