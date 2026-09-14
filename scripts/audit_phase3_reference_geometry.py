"""Phase 3 Stage 3-A real-WOMD audit (read-only diagnostic, not a
test): builds ``ReferenceLine``s from REAL lane polylines across a
representative sample of the 168 canonical maneuvers (both TRAIN and
VALIDATION splits), including every (source, target) pair in every
CHAINED maneuver's lane_chain, and reports the measurements the Stage
3-A gate requires:

  - raw point spacing (p50/p95/max)
  - processed (post-cleanup) point spacing (p50/p95/max)
  - reference length distribution
  - max/p95 curvature magnitude
  - max curvature-derivative magnitude
  - duplicate/near-duplicate points removed (total + per-lane)
  - lane polylines that failed to build a valid reference (if any)
  - references with non-finite output (should be zero)
  - construction time p50/p95 (wall clock)
  - projection query time p50/p95 (wall clock)

Run with:
    PYTHONPATH=. python3 scripts/audit_phase3_reference_geometry.py

Uses the same real local WOMD shard data and maneuver-spec loading
already relied on by tests/environment/test_merge_environment.py and
tests/environment/test_common_state_freeze.py -- no synthetic
substitute data.
"""

import time

import numpy as np

from src.environment.full_split_evaluator import load_maneuver_specs
from src.environment.merge_environment import MergeEnvironment
from src.planning.reference import InvalidReferenceGeometryError, ReferenceLine

DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"

# Representative sample size per split (Stage 3-A explicitly does not
# require exhaustive full-split coverage the way Stage B-2.5 did;
# "representative sample" per the task brief). All chained maneuvers
# are included regardless of sample size, from both splits.
TRAIN_SAMPLE_SIZE = 40
VALIDATION_SAMPLE_SIZE = 40


def _percentile(values, p):
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def main():
    train_specs = load_maneuver_specs("train")
    validation_specs = load_maneuver_specs("validation")

    print(f"Loaded {len(train_specs)} TRAIN maneuvers, "
          f"{len(validation_specs)} VALIDATION maneuvers.")

    chained_train = [s for s in train_specs if len(s.lane_chain) > 2]
    chained_validation = [s for s in validation_specs if len(s.lane_chain) > 2]
    print(f"Chained maneuvers: {len(chained_train)} TRAIN, "
          f"{len(chained_validation)} VALIDATION.")

    sample_train = train_specs[:TRAIN_SAMPLE_SIZE]
    sample_validation = validation_specs[:VALIDATION_SAMPLE_SIZE]

    # Union (by maneuver_id) of the sample + all chained maneuvers
    # from both splits, so every chained (source,target) pair is
    # covered even if the plain slice sample didn't already include
    # it.
    seen_ids = set()
    all_specs = []
    for spec in sample_train + chained_train + sample_validation + chained_validation:
        if spec.maneuver_id in seen_ids:
            continue
        seen_ids.add(spec.maneuver_id)
        all_specs.append(spec)

    print(f"Auditing {len(all_specs)} distinct maneuvers "
          f"({len(sample_train)} TRAIN sample + {len(chained_train)} "
          f"TRAIN chained + {len(sample_validation)} VALIDATION sample "
          f"+ {len(chained_validation)} VALIDATION chained, deduped).")

    env = MergeEnvironment(dataset_config_path=DATASET_CONFIG_PATH)

    raw_spacings = []
    processed_spacings = []
    reference_lengths = []
    max_curvatures = []
    max_curvature_derivatives = []
    construction_times = []
    projection_times = []

    duplicates_removed_total = 0
    duplicates_removed_per_lane = []

    failed_lanes = []  # (lane_id, maneuver_id, reason)
    non_finite_lanes = []
    reset_failures = []

    audited_lane_ids = set()  # avoid re-auditing the same lane_id twice

    for spec in all_specs:
        try:
            env.reset(spec)
        except Exception as exc:  # noqa: BLE001 -- record and continue
            reset_failures.append((spec.maneuver_id, f"{type(exc).__name__}: {exc}"))
            continue

        for lane_id in spec.lane_chain:
            key = (spec.maneuver_id, lane_id)
            if lane_id in audited_lane_ids:
                # Same physical lane can recur across maneuvers; still
                # worth timing/measuring once per (maneuver, lane) to
                # capture chained-transition pairs faithfully, but
                # avoid double counting distinct-lane duplicate stats.
                pass
            polyline = env._polylines_by_id.get(lane_id)
            if polyline is None:
                failed_lanes.append((lane_id, spec.maneuver_id, "lane_id not found in scene"))
                continue

            raw_xy = np.asarray(polyline.xy, dtype=np.float64)
            if raw_xy.shape[0] >= 2:
                raw_seg = np.hypot(
                    np.diff(raw_xy[:, 0]), np.diff(raw_xy[:, 1])
                )
                raw_spacings.append(raw_seg)

            try:
                t0 = time.perf_counter()
                reference = ReferenceLine.from_lane_polyline(polyline)
                construction_times.append(time.perf_counter() - t0)
            except InvalidReferenceGeometryError as exc:
                failed_lanes.append((lane_id, spec.maneuver_id, str(exc)))
                continue

            n_removed = raw_xy.shape[0] - reference.x.shape[0]
            duplicates_removed_total += n_removed
            duplicates_removed_per_lane.append((lane_id, spec.maneuver_id, n_removed))

            if reference.x.shape[0] >= 2:
                processed_seg = np.hypot(
                    np.diff(reference.x), np.diff(reference.y)
                )
                processed_spacings.append(processed_seg)

            reference_lengths.append(reference.length_m)

            all_finite = (
                np.all(np.isfinite(reference.x))
                and np.all(np.isfinite(reference.y))
                and np.all(np.isfinite(reference.s))
                and np.all(np.isfinite(reference.yaw))
                and np.all(np.isfinite(reference.curvature))
                and np.all(np.isfinite(reference.curvature_derivative))
            )
            if not all_finite:
                non_finite_lanes.append((lane_id, spec.maneuver_id))
                continue

            max_curvatures.append(float(np.max(np.abs(reference.curvature))))
            max_curvature_derivatives.append(
                float(np.max(np.abs(reference.curvature_derivative)))
            )

            # Projection query timing: project the lane's own midpoint
            # (a representative, always-valid query for this lane).
            mid_idx = reference.x.shape[0] // 2
            qx, qy = float(reference.x[mid_idx]), float(reference.y[mid_idx])
            t0 = time.perf_counter()
            reference.project(qx, qy)
            projection_times.append(time.perf_counter() - t0)

            audited_lane_ids.add(lane_id)

    raw_spacings_flat = (
        np.concatenate(raw_spacings) if raw_spacings else np.zeros(0)
    )
    processed_spacings_flat = (
        np.concatenate(processed_spacings) if processed_spacings else np.zeros(0)
    )

    print("\n=== Stage 3-A real-WOMD audit results ===")
    print(f"Scene resets attempted: {len(all_specs)}, "
          f"failed: {len(reset_failures)}")
    if reset_failures:
        for maneuver_id, reason in reset_failures:
            print(f"  RESET FAILURE {maneuver_id}: {reason}")

    print(f"\nLane polylines audited (successful ReferenceLine builds): "
          f"{len(reference_lengths)}")
    print(f"Lane polylines FAILED to build a valid reference: {len(failed_lanes)}")
    for lane_id, maneuver_id, reason in failed_lanes:
        print(f"  FAILED lane_id={lane_id} maneuver={maneuver_id}: {reason}")

    print(f"Lane references with NON-FINITE output: {len(non_finite_lanes)}")
    for lane_id, maneuver_id in non_finite_lanes:
        print(f"  NON-FINITE lane_id={lane_id} maneuver={maneuver_id}")

    print(f"\nRaw point spacing (m): "
          f"p50={_percentile(raw_spacings_flat, 50):.4f} "
          f"p95={_percentile(raw_spacings_flat, 95):.4f} "
          f"max={float(np.max(raw_spacings_flat)) if len(raw_spacings_flat) else float('nan'):.4f} "
          f"min={float(np.min(raw_spacings_flat)) if len(raw_spacings_flat) else float('nan'):.4f} "
          f"n={len(raw_spacings_flat)}")

    print(f"Processed point spacing (m): "
          f"p50={_percentile(processed_spacings_flat, 50):.4f} "
          f"p95={_percentile(processed_spacings_flat, 95):.4f} "
          f"max={float(np.max(processed_spacings_flat)) if len(processed_spacings_flat) else float('nan'):.4f} "
          f"n={len(processed_spacings_flat)}")

    print(f"\nReference length (m): "
          f"p50={_percentile(reference_lengths, 50):.2f} "
          f"p95={_percentile(reference_lengths, 95):.2f} "
          f"max={max(reference_lengths) if reference_lengths else float('nan'):.2f} "
          f"min={min(reference_lengths) if reference_lengths else float('nan'):.2f} "
          f"n={len(reference_lengths)}")

    print(f"\nCurvature magnitude (1/m): "
          f"p95={_percentile(max_curvatures, 95):.6f} "
          f"max={max(max_curvatures) if max_curvatures else float('nan'):.6f}")

    print(f"Curvature derivative magnitude (1/m^2): "
          f"p95={_percentile(max_curvature_derivatives, 95):.6f} "
          f"max={max(max_curvature_derivatives) if max_curvature_derivatives else float('nan'):.6f}")

    print(f"\nDuplicate/near-duplicate points removed: "
          f"total={duplicates_removed_total}, "
          f"lanes_with_removals={sum(1 for _, _, n in duplicates_removed_per_lane if n > 0)}"
          f"/{len(duplicates_removed_per_lane)}")

    construction_times_ms = [t * 1000.0 for t in construction_times]
    projection_times_ms = [t * 1000.0 for t in projection_times]
    print(f"\nConstruction time (ms): "
          f"p50={_percentile(construction_times_ms, 50):.4f} "
          f"p95={_percentile(construction_times_ms, 95):.4f}")
    print(f"Projection query time (ms): "
          f"p50={_percentile(projection_times_ms, 50):.4f} "
          f"p95={_percentile(projection_times_ms, 95):.4f}")

    print("\n=== Gate check ===")
    gate_pass = (
        len(reset_failures) == 0
        and len(failed_lanes) == 0
        and len(non_finite_lanes) == 0
    )
    print(f"Zero reset failures: {len(reset_failures) == 0}")
    print(f"Zero lane build failures: {len(failed_lanes) == 0}")
    print(f"Zero non-finite references: {len(non_finite_lanes) == 0}")
    print(f"AUDIT GATE: {'PASS' if gate_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
