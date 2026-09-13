"""Phase 3 Stage 3-B real-WOMD spot-check (read-only diagnostic, not a
test): builds ``ReferenceLine``s from a representative sample of REAL
lane polylines (same loading pattern as
``scripts/audit_phase3_reference_geometry.py``) and, for each, runs a
handful of sample Frenet<->Cartesian conversions, checking for finite
output and zero crashes. This is intentionally NOT as exhaustive as
Stage 3-A's full audit -- the Stage 3-B gate only requires a
representative spot-check (10-20 reference lines, a few points each).

Run with:
    PYTHONPATH=. python3 scripts/audit_phase3_frenet_transform.py
"""

import numpy as np

from src.environment.full_split_evaluator import load_maneuver_specs
from src.environment.merge_environment import MergeEnvironment
from src.planning.frenet_transform import cartesian_to_frenet, frenet_to_cartesian
from src.planning.frenet_types import FrenetState
from src.planning.reference import InvalidReferenceGeometryError, ReferenceLine

DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"

NUM_REFERENCE_LINES = 15
SPEEDS_MPS = [0.0, 0.01, 2.0, 8.0, 15.0]
LATERAL_OFFSETS_M = [-2.0, 0.0, 2.0]


def main():
    train_specs = load_maneuver_specs("train")
    print(f"Loaded {len(train_specs)} TRAIN maneuvers.")

    env = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)

    references = []
    seen_lane_ids = set()
    for spec in train_specs:
        if len(references) >= NUM_REFERENCE_LINES:
            break
        try:
            env.reset(spec)
        except Exception as exc:  # noqa: BLE001
            print(f"  RESET FAILURE {spec.maneuver_id}: {type(exc).__name__}: {exc}")
            continue
        for lane_id in spec.lane_chain:
            if lane_id in seen_lane_ids:
                continue
            polyline = env._polylines_by_id.get(lane_id)
            if polyline is None:
                continue
            try:
                reference = ReferenceLine.from_lane_polyline(polyline)
            except InvalidReferenceGeometryError as exc:
                print(f"  BUILD FAILURE lane_id={lane_id}: {exc}")
                continue
            references.append((spec.maneuver_id, lane_id, reference))
            seen_lane_ids.add(lane_id)
            if len(references) >= NUM_REFERENCE_LINES:
                break

    print(f"Spot-checking {len(references)} real ReferenceLines.")

    total_checks = 0
    non_finite_count = 0
    crash_count = 0

    for maneuver_id, lane_id, reference in references:
        sample_s_values = np.linspace(
            0.0, reference.length_m, num=4
        ).tolist()

        for s in sample_s_values:
            for v in SPEEDS_MPS:
                for d in LATERAL_OFFSETS_M:
                    total_checks += 1
                    try:
                        state = FrenetState(
                            s=s, s_d=v, s_dd=0.0, d=d, d_d=0.0, d_dd=0.0
                        )
                        cart = frenet_to_cartesian(state, reference)
                        finite = (
                            np.isfinite(cart.x) and np.isfinite(cart.y)
                            and np.isfinite(cart.yaw) and np.isfinite(cart.curvature)
                            and np.isfinite(cart.velocity)
                            and np.isfinite(cart.acceleration)
                        )
                        if not finite:
                            non_finite_count += 1
                            print(
                                f"  NON-FINITE frenet_to_cartesian lane={lane_id} "
                                f"maneuver={maneuver_id} s={s:.2f} v={v} d={d}: {cart}"
                            )
                            continue

                        frenet_back = cartesian_to_frenet(
                            cart.x, cart.y, cart.yaw, cart.velocity,
                            cart.acceleration, reference,
                        )
                        finite_back = (
                            np.isfinite(frenet_back.s) and np.isfinite(frenet_back.s_d)
                            and np.isfinite(frenet_back.s_dd)
                            and np.isfinite(frenet_back.d)
                            and np.isfinite(frenet_back.d_d)
                            and np.isfinite(frenet_back.d_dd)
                        )
                        if not finite_back:
                            non_finite_count += 1
                            print(
                                f"  NON-FINITE cartesian_to_frenet lane={lane_id} "
                                f"maneuver={maneuver_id} s={s:.2f} v={v} d={d}: "
                                f"{frenet_back}"
                            )
                    except Exception as exc:  # noqa: BLE001
                        crash_count += 1
                        print(
                            f"  CRASH lane={lane_id} maneuver={maneuver_id} "
                            f"s={s:.2f} v={v} d={d}: {type(exc).__name__}: {exc}"
                        )

    print("\n=== Stage 3-B real-WOMD spot-check results ===")
    print(f"Reference lines checked: {len(references)}")
    print(f"Total conversion checks: {total_checks}")
    print(f"Non-finite outputs: {non_finite_count}")
    print(f"Crashes: {crash_count}")
    gate_pass = len(references) >= 10 and non_finite_count == 0 and crash_count == 0
    print(f"SPOT-CHECK GATE: {'PASS' if gate_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
