# Phase 3 Overnight Implementation — Progress / Recovery Log

This file is the recovery source of truth for the autonomous Stage 3-A → 3-G
implementation run. If context is lost, re-read this file and continue from
the last verified-complete Stage.

## Run metadata

- Base branch: `phase2/ppo-environment-design`
- Base commit (Phase 2 frozen): `6283ebe79591317cf4771bab002e275fd6a4d709`
- Working branch: `phase3/common-downstream`
- ASMC reference snapshot audited: `b6fff3d3c4d14c7946dbeac94d3146f125ca00a7` (read-only, never vendored)
- Waymax installed revision: `a64dfec9be8576b60d9cecc94f406d9812d4a7d0`
- Stage 3-0 audit: completed in prior conversation turn (not committed as a file; see conversation history)

## Locked design decisions (from Stage 3-0, approved by user)

1. MPC dynamics: Waymax-native `(acceleration_mps2, steering_curvature)`, no ASMC absolute-steering-angle model, no artificial wheelbase.
2. MPC dt = 0.1s, matches Waymax simulation dt 1:1. No inner 0.05s loop.
3. Optimizer: `scipy.optimize`, method chosen based on constraint shape, documented. No JAX unless SciPy proves incapable.
4. STOP target: `s_stop = s_current + v_current^2 / (2 * comfortable_deceleration)`, `comfortable_deceleration = 2.0 m/s^2` in Phase 3 config. Clamp to reference domain if needed, expose `stop_target_clamped`.
5. MERGE fairness: planner/controller never decide *whether* to merge, only *how*. No gap search, no ASMC `FindRoundaboutGap`/`FindHighwayMergeGap`/`merge_gap_safe` porting. Explicit `PLANNER_INFEASIBLE`/`COLLISION_BLOCKED` statuses instead of silent downgrade to another action.

## Stage status

| Stage | Status | Commit SHA |
|---|---|---|
| 3-A Reference/Geometry | COMPLETE | `95ab85c92ab63f74cac036bc46f660826061369a` |
| 3-B Frenet Core | NOT STARTED | — |
| 3-C BehaviorAction Execution Mapping | NOT STARTED | — |
| 3-D LTV-MPC | NOT STARTED | — |
| 3-E Waymax Adapter / Common Downstream | NOT STARTED | — |
| 3-F MergeEnvironment Integration | NOT STARTED | — |
| 3-G Robustness/Regression | NOT STARTED | — |

## Log

### Setup (pre-3-A)
- Verified `git status` clean, branch/HEAD matched expected Phase 2 frozen state exactly.
- Created `phase3/common-downstream` from `6283ebe7`.
- Created this progress file.
- Next: Stage 3-A.

### Stage 3-A: Frenet reference-line geometry layer

**Files added:**
- `src/planning/reference.py` (465 lines) — `ReferenceLine` dataclass + `ReferenceLine.from_lane_polyline(...)` builder, built as a thin layer on top of `src.scenarios.lane_geometry.LanePolyline`. Adds yaw, curvature, curvature-derivative (finite differences over x/y, matching the existing finite-difference convention already used elsewhere in this repo), linear (x, y)/curvature interpolation, circular (angle-aware) yaw interpolation, unclamped open-path extrapolation at the ends (matching `project_point_to_polyline_signed`'s existing precedent), and defensive near-duplicate point cleanup (`DUPLICATE_POINT_EPS_M = 1e-3` m). Does NOT resample WOMD's roadgraph points — real spacing is already near-uniform (measured below), so resampling would manufacture geometry not present in the source data.
- `tests/planning/test_reference.py` (265 lines, 21 tests) — unit tests for construction, duplicate cleanup, yaw/curvature correctness on synthetic straight/circular-arc geometry, interpolation (linear + circular), open-path extrapolation, and error handling on degenerate/non-finite input.
- `scripts/audit_phase3_reference_geometry.py` (256 lines) — read-only diagnostic script that builds `ReferenceLine`s from real WOMD lane polylines across a representative sample of both TRAIN/VALIDATION splits plus every chained-maneuver lane pair, and reports the measurements the Stage 3-A gate requires.

**Key design decisions:**
1. No resampling of WOMD roadgraph points — measured real spacing is already ~1.0 m and near-uniform (WOMD's own fixed sampling rate), unlike the ASMC reference stack's dense/irregular global-planner input that motivated its 10 m window / 0.5 m resampling / 300-iteration curvature relaxation. That stack was tuned for a different data source and isn't warranted here absent a measured problem.
2. Near-duplicate cleanup (`DUPLICATE_POINT_EPS_M = 1e-3` m) is kept purely as defensive input validation (future data sources, synthetic/hand-built polylines, degenerate lanes) — real WOMD data has not been observed to need it (min segment length measured 0.394 m, ~400x above the epsilon).
3. Yaw/curvature via simple finite differences over (x, y), matching `lane_geometry.py`'s / `merge_reference.py`'s existing approach in this repo, for consistency. Well-conditioned at ~1 m spacing; zero non-finite output observed across the full real-WOMD audit.
4. Open path (no wraparound): out-of-domain queries linearly extrapolate along the endpoint segment's own tangent, matching `project_point_to_polyline_signed`'s existing "unclamped, preserve relative ordering" precedent in `lane_geometry.py`.

**Gate verification (real measurements, both jobs run to completion in the `its-merge` conda env):**

Command: `PYTHONPATH=. python3 scripts/audit_phase3_reference_geometry.py`

```
Scene resets attempted: 85, failed: 0

Lane polylines audited (successful ReferenceLine builds): 178
Lane polylines FAILED to build a valid reference: 0
Lane references with NON-FINITE output: 0

Raw point spacing (m): p50=0.9923 p95=0.9990 max=1.0001 min=0.3940 n=6076
Processed point spacing (m): p50=0.9923 p95=0.9990 max=1.0001 n=6076

Reference length (m): p50=22.73 p95=85.22 max=208.07 min=4.38 n=178

Curvature magnitude (1/m): p95=0.187281 max=0.299406
Curvature derivative magnitude (1/m^2): p95=0.041965 max=0.124524

Duplicate/near-duplicate points removed: total=0, lanes_with_removals=0/178

Construction time (ms): p50=0.1444 p95=0.2753
Projection query time (ms): p50=0.0778 p95=0.1558

AUDIT GATE: PASS
```

Sample composition: 85 distinct maneuvers audited = 40 TRAIN sample + 40 VALIDATION sample + 7 TRAIN chained + 1 VALIDATION chained (deduped union), out of 110 TRAIN / 58 VALIDATION maneuvers loaded total. All 8 chained maneuvers across both splits (every lane in every chained maneuver's `lane_chain`) are included, per the Stage 3-A gate requirement, not just the sampled subset.

Full suite: `python3 -m pytest tests/ -q` → **298 passed, 0 failed, 0 errors** in 2773.38s (46m13s), run in the same `its-merge` conda environment against the unmodified Phase 2 test tree plus the new `tests/planning/test_reference.py` (21 tests, included in the 298).

**Gate status — all four criteria met:**

| Criterion | Result |
|---|---|
| (a) New geometry tests pass | 21/21 pass (`tests/planning/test_reference.py`) |
| (b) Real-WOMD audit: representative sample of both splits + all chained-maneuver lane pairs, zero non-finite, zero crashes | PASS — 85 maneuvers, 178 lane references, 0 reset failures, 0 build failures, 0 non-finite outputs |
| (c) All existing Phase 2 tests still pass unchanged | 298/298 passed, 0 failed (full `tests/`) |
| (d) No Phase 2 production file modified | Confirmed — `git diff --stat HEAD` is empty; only new untracked files added (`src/planning/`, `tests/planning/`, `scripts/audit_phase3_reference_geometry.py`, `docs/`) |

**Note on measurement provenance:** the module docstring in `reference.py` was first written against a smaller preliminary sample (38 maneuvers / 2957 segments) before the full 85-maneuver gate audit ran; it has been updated in this commit to cite the full gate-run numbers above (min segment length 0.394 m, not the earlier preliminary 0.487 m — both are consistent with "near-uniform ~1 m spacing, no real near-duplicates," the correction is one of measurement precision, not a change in conclusion).

**Commit:** `95ab85c92ab63f74cac036bc46f660826061369a`
