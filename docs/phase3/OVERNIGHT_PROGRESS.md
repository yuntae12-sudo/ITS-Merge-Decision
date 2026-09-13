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
| 3-B Frenet Core | COMPLETE | `d4b77312639ecf87908a703cfa5db588258ea2e0` |
| 3-C BehaviorAction Execution Mapping | COMPLETE | `PENDING` |
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

### Stage 3-B: Frenet trajectory core (quintic/quartic polynomials + Frenet<->Cartesian transform)

**Files added:**
- `src/planning/frenet_types.py` — `FrenetState` (frozen dataclass: s, s_d, s_dd, d, d_d, d_dd), `FrenetPath` (time-sampled trajectory arrays plus optional `s_ddd` jerk array and `valid`/`rejection_reason` fields reserved for Stage 3-C's candidate evaluator, not implemented here), `CartesianTrajectory` (time-sampled x, y, yaw, curvature, velocity, acceleration). Pure dataclasses, no BehaviorAction dependency.
- `src/planning/polynomial.py` — `QuinticPolynomial` (fully-constrained: position/velocity/acceleration boundary conditions at both t=0 and t=T) and `QuarticPolynomial` (velocity-keeping: position/velocity/acceleration at t=0, only velocity/acceleration at t=T, terminal position free). Both solve the boundary-value problem by setting up and solving the small (3x3 / 2x2) linear system implied by the boundary conditions via `numpy.linalg.solve`, rather than transcribing a hand-expanded closed-form matrix-inverse formula from memory — mathematically identical but eliminates transcription-error risk. Both expose `position/velocity/acceleration/jerk` evaluation and an exact closed-form `jerk_cost()` (`integral_0^T jerk(t)^2 dt`, computed by symbolically integrating the squared jerk polynomial term-by-term).
- `src/planning/frenet_transform.py` — `frenet_to_cartesian`/`cartesian_to_frenet`, consuming Stage 3-A's `ReferenceLine`. Position uses the standard Werling et al. offset formula (`x = rx - d*sin(theta_r)`, `y = ry + d*cos(theta_r)`); heading/curvature/velocity/acceleration are recovered via a first-principles route (Frenet tangential/normal velocity-acceleration decomposition, rotated into world frame via full product-rule differentiation to pick up frame-rotation cross terms, then yaw/curvature/acceleration read off the resulting Cartesian velocity/acceleration vectors via the exact planar identities `yaw=atan2(vy,vx)`, `kappa=(vx*ay-vy*ax)/v^3`, `a=(vx*ax+vy*ay)/v`) rather than transcribing Werling's higher-order closed-form heading/curvature composition formula from memory. Explicit low-speed fallback (see design decisions below).
- `tests/planning/test_frenet_types.py`, `test_polynomial.py`, `test_frenet_transform.py` — 30 new tests (6 + 7 + 17 respectively).
- `scripts/audit_phase3_frenet_transform.py` — read-only real-WOMD spot-check script (15 reference lines x 4 s-samples x 5 speeds x 3 lateral offsets = 900 conversion checks).

**Key design decisions:**
1. **Polynomial boundary-value solve via `np.linalg.solve` on the small residual linear system**, not a memorized closed-form inverse — the boundary conditions are already the exact problem statement; solving them numerically is exact (up to float64 rounding) and removes any risk of a mis-transcribed textbook formula. Verified in tests to reproduce boundary conditions to <1e-8 absolute error.
2. **Jerk cost computed by exact term-by-term polynomial integration** (`integral_0^T t^k dt = T^(k+1)/(k+1)` applied to every cross term of the squared jerk polynomial), not numerically — this is exact for any polynomial. Cross-checked against `scipy.integrate.quad` numerical integration across 20 random boundary-condition sets each for quintic and quartic, requiring `rel=1e-6, abs=1e-9` agreement (chosen because both the closed-form sum and `quad`'s adaptive quadrature on a smooth low-degree polynomial converge to close to machine precision, so a fairly tight relative tolerance is appropriate and both methods agreed far inside it in practice).
3. **Frenet transform uses TIME derivatives** throughout (`s_d`, `d_d`, ...) to match `FrenetState`'s field convention, rather than Werling's original arc-length-derivative (`d' = dd/ds`) notation — internally related by the chain rule wherever needed, never exposed externally.
4. **Low-speed singularity threshold: `LOW_SPEED_THRESHOLD_MPS = 0.5` m/s.** Reasoned from this repo's own numbers, not copied from another codebase: Phase 3's MPC dt is locked at 0.1s, and this repo's own `comfortable_deceleration = 2.0 m/s^2` (locked STOP-target design decision) means one control step changes speed by 0.2 m/s — 0.5 m/s is ~2.5 control-steps' margin (avoids branch-flapping across consecutive planner ticks near a STOP) while remaining far below `NOMINAL_CRUISE_SPEED_MPS = 15.0` (3.3%), so the fallback never fires during normal-speed driving. Below threshold, both directions use a purely geometric fallback (assume lane-aligned heading `theta = theta_r`, `kappa = kappa_r`, no division by `s_d`) — deliberately exercised by the STOP maneuver's own terminal state (`STOP_TARGET_SPEED_MPS = 0.0`), confirmed via a dedicated test (`test_stop_maneuver_terminal_state_finite`). A secondary guard also checks the reconstructed Cartesian speed `v` itself (can be near-zero even when `s_d` wasn't, at a lateral-motion cusp) before the curvature/acceleration divide. The geometric `(1 - kappa_r*d)` singularity is also defensively clamped (floor `1e-3`) in `cartesian_to_frenet`, per the task brief's note that it's unlikely to matter for typical highway-merge `d` ranges but should not silently blow up if ever hit.
5. **`acceleration` is a scalar, tangential-to-heading quantity** in both transform directions (`(ax,ay) = acceleration*(cos(yaw),sin(yaw))` on the way in; `(vx*ax+vy*ay)/v` on the way out) — matches Waymax's own `(acceleration_mps2, steering_curvature)` action convention (locked design decision #1) and makes the two transform directions round-trip-consistent by construction, not coincidentally. Documented explicitly in `cartesian_to_frenet`'s docstring so a future caller does not mistakenly pass a full 2D (including centripetal) acceleration vector.
6. **Sign convention (`d>0` = left of travel) re-verified, not assumed**, against Stage 3-A's own `ReferenceLine.project` (`test_consistent_with_reference_projection_sign`) and directly on a synthetic straight reference (`test_positive_d_is_left_of_travel`).
7. During test-tolerance tuning, discovered (and documented in `test_frenet_transform.py`) that curved-reference round-trip position error is dominated by `ReferenceLine`'s own finite-difference curvature-estimation error (not a transform bug) — verified by re-running the same round trip at 400/2000/8000/32000-point arc resolutions and observing the residual shrink well faster than linearly, consistent with the central-difference scheme's second-order accuracy. The curved-reference test fixture uses 4000 points (denser than the 400-point default used for other synthetic geometry in this file) with a documented `3e-3` m position tolerance to comfortably clear this known, understood, non-bug error source.

**Test results:**

`PYTHONPATH=. pytest tests/planning/ -q` → **51 passed** (21 unchanged Stage 3-A `test_reference.py` + 30 new: 6 `test_frenet_types.py` + 7 `test_polynomial.py` + 17 `test_frenet_transform.py`, run to completion in the `its-merge` conda env), 0 failed.

Full suite: `pytest tests/ -q` → **328 passed, 0 failed, 0 errors** in 1588.40s (26m28s), confirming exactly 298 (Stage 3-A baseline) + 30 (new Stage 3-B) = 328, i.e. zero Phase 2/Stage 3-A regressions and all new tests accounted for. No warnings beyond the usual TF/oneDNN/GPU startup notices already seen in Stage 3-A's run.

**Real-WOMD spot-check** (`PYTHONPATH=. python3 scripts/audit_phase3_frenet_transform.py`):

```
Loaded 110 TRAIN maneuvers.
Spot-checking 15 real ReferenceLines.

=== Stage 3-B real-WOMD spot-check results ===
Reference lines checked: 15
Total conversion checks: 900
Non-finite outputs: 0
Crashes: 0
SPOT-CHECK GATE: PASS
```

**Gate status — all criteria met:**

| Criterion | Result |
|---|---|
| (1) New Frenet-math tests pass (boundary conditions, jerk cost cross-check, round-trips, sign convention, low-speed, yaw wrap, curvature finiteness, boundary handling) | 51/51 pass in `tests/planning/` (30 new) |
| (2) Stage 3-A's geometry tests still pass unchanged | Confirmed — same 21 tests in `test_reference.py`, untouched file, still passing |
| (3) ALL existing Phase 2 tests still pass | 328/328 passed, 0 failed, 0 errors, full `tests/` (1588.40s) |
| (4) Real-WOMD spot-check (10-20 reference lines) finite, zero crashes | PASS — 15 reference lines, 900 checks, 0 non-finite, 0 crashes |
| (5) No Phase 2 production file modified, no Stage 3-A file modified | Confirmed — `git diff --stat HEAD` empty; only new files under `src/planning/`, `tests/planning/`, `scripts/`, `docs/` |

**Known issues / follow-ups for Stage 3-C:**
- The curved-reference round-trip discretization-error finding above (item 7) is worth keeping in mind if Stage 3-C's candidate evaluator does tight numerical comparisons against real (not synthetic) WOMD reference lines with sparser point density than the 4000-point synthetic fixture used here — real WOMD spacing is ~1m (per Stage 3-A's own measurement), so this is not expected to be a practical problem, but has not been exhaustively quantified against real curvature ranges the way the synthetic arc was.
- `frenet_types.FrenetPath.s_ddd` and `valid`/`rejection_reason` fields are defined but unused placeholders, intentionally deferred to Stage 3-C per the task brief.

**Commit:** `d4b77312639ecf87908a703cfa5db588258ea2e0`

**Next stage: 3-C** (BehaviorAction execution mapping / candidate generation).

### Stage 3-C: Common Frenet behavior-execution planner

**Files added:**
- `src/planning/candidate_generator.py` -- generates exactly ONE candidate `FrenetPath` per already-decided `BehaviorAction`/`BehaviorObjective`: KEEP (lateral quintic to d=0 in source frame; longitudinal quartic velocity-keeping toward `objective.reference_speed_mps`), FOLLOW (identical shape to KEEP, executing whatever follow-speed `BehaviorExecutor` already computed against the causal source-lane lead), MERGE (lateral quintic generated DIRECTLY in the TARGET reference's own Frenet frame -- ego's current Cartesian state is projected into the target frame via `cartesian_to_frenet`, approach (b) from the task brief -- longitudinal identical shape to FOLLOW/KEEP against the causal target-lane lead or cruise speed), STOP (lateral quintic to d=0 in source frame; longitudinal QUINTIC to `s_stop = s_current + v_current^2/(2*comfortable_deceleration)`, using its own physically-natural stopping-time horizon rather than the generic cross-action horizon -- see design decision 2 below -- with explicit `stop_target_clamped` diagnostic if `s_stop` exceeds the reference's domain).
- `src/planning/candidate_evaluator.py` -- explicit feasibility/collision evaluation of one generated candidate: longitudinal acceleration bound (+-6.0 m/s^2), non-negative forward progress, curvature bound (0.3 1/m, computed via Stage 3-B's `frenet_to_cartesian`), longitudinal jerk bound (20.0 m/s^3), and a constant-velocity/circular-proxy collision check against current surrounding-agent state. Defines `PlannerStatus` (`OK`/`PLANNER_INFEASIBLE`/`COLLISION_BLOCKED`/`INVALID_REFERENCE`) and `EvaluationResult` (status + reason + which checks were evaluated/failed + Cartesian trajectory + collision diagnostics).
- `src/planning/frenet_planner.py` -- top-level `plan(request, config)` entry point wiring generation + evaluation together; `PlanRequest`/`PlanResult`/`PlannerConfig`/`EgoKinematicState`/`FollowInputs` dataclasses; `load_planner_config` reads `configs/phase3_downstream.yaml`. NOT wired into `MergeEnvironment` (Stage 3-F's job).
- `configs/phase3_downstream.yaml` -- planner dt/horizon, `comfortable_deceleration_mps2=2.0` (locked decision #4), feasibility limits (`max_longitudinal_accel_mps2=6.0` and `max_curvature_per_m=0.3`, both reused directly from `low_level_controller.py`'s existing `MAX_ACCEL_MPS2`/`MAX_STEERING_CURVATURE` for consistency with the one dynamics model these trajectories are actually executed on; `max_longitudinal_jerk_mps3=20.0`, reasoned from the other two bounds -- see the config file's own comments), and collision-proxy radii/margin.
- `tests/planning/test_candidate_generator.py` (9 tests), `tests/planning/test_candidate_evaluator.py` (9 tests), `tests/planning/test_frenet_planner.py` (12 tests, including 2 backed by real WOMD TRAIN maneuvers via the same `load_maneuver_specs`/`MergeEnvironment` loading pattern Stage 3-B's audit script established) -- 30 new tests total under `tests/planning/`.

**Key design decisions:**

1. **Causal gap/relative-speed sourcing for FOLLOW/MERGE**: the planner's `PlanRequest`/`FollowInputs` accept `reference_speed_mps` already computed by the frozen `BehaviorExecutor.compute_objective` (which itself derives its follow-speed from the 14D observation's own causally-selected `source_front_gap`/`source_front_relative_speed` or `target_front_gap`/`target_front_relative_speed`, via `src.scenarios.scenario_features`). `candidate_generator.py` never reselects a leader or re-derives gap/TTC -- it consumes `objective.reference_speed_mps` directly, exactly as the task brief required ("if it's simplest to just accept ... as direct inputs to your planner request, do that"). Concretely, both FOLLOW and MERGE call the identical `generate_follow_or_merge_candidate` function, which is literally `generate_keep_candidate` under an alias (same quartic velocity-keeping shape) -- the only difference between KEEP/FOLLOW/MERGE longitudinal profiles is WHICH speed and WHICH frame they are handed, never a different code path.

2. **STOP's longitudinal horizon is the physically natural stopping time** (`v_current / comfortable_deceleration_mps2`, floored at `MIN_STOP_HORIZON_S = 1.0`), not the shared generic `trajectory_horizon_s` (3.0s) used by every other action's longitudinal profile. This was discovered as a genuine bug during test-driven development, not assumed: forcing a quintic's fixed terminal state `[s_stop, v=0, a=0]` into an unrelated fixed 3.0s horizon (when the natural constant-deceleration stopping time was, e.g., 6.0s for a 12 m/s current speed) produces a sharp overshoot-and-correct S-curve that peaks at approximately -15.7 m/s^2 (2.6x the 6.0 m/s^2 limit) -- verified directly via `scripts`-style diagnostic before writing the fix. Using the natural stopping time instead makes that same maneuver peak at exactly the requested 3.0 m/s^2 (well within bounds) because it reduces to the textbook constant-deceleration profile. This is NOT a behavior decision (it does not change the stop TARGET, `s_stop`, at all -- only how much time the already-fixed terminal condition is given to reach, which is a kinematic/comfort consistency fix, matching `comfortable_deceleration_mps2`'s own definition). The lateral quintic still uses the caller's shared `horizon_s` (lateral convergence has no dependency on the longitudinal stopping physics). When the two horizons differ, the returned `FrenetPath` spans the LONGER of the two (so the full stop is always represented), holding whichever polynomial finished first at its own already-converged terminal value for the remaining samples (each terminal condition is an exact fixed point, so holding is exact, not an approximation).

3. **MERGE frame-handling**: approach (b) from the task brief -- ego's current Cartesian state is projected directly into the TARGET reference's own Frenet frame via `cartesian_to_frenet` (`candidate_generator.project_cartesian_to_frame`), and the MERGE candidate is generated entirely within that frame (terminal `d=0` in the target frame). No blended source/target reference is invented. `frenet_planner.plan()` decides which reference is "active" purely from `objective.reference_lane` ("source" vs "target", the frozen `BehaviorObjective` field) -- KEEP/FOLLOW/STOP always resolve to `source_reference`, MERGE always resolves to `target_reference`. This is a pure frame/geometry choice, not a decision about whether to merge.

4. **Collision-check simplification**: both ego and every surrounding agent are modeled as CIRCLES (not oriented rectangles), and future agent position is predicted via CURRENT position + CURRENT velocity * elapsed time only -- no logged/future trajectory is ever read for any agent. This is explicitly permitted by the task brief ("a reasonably conservative circular or axis-aligned approximation is acceptable") and is sufficient for Stage 3-C's correctness bar: the collision-status contract (`OK`/`PLANNER_INFEASIBLE`/`COLLISION_BLOCKED`/`INVALID_REFERENCE`) does not depend on the geometric precision of the overlap test, only on the test firing correctly for a genuinely-colliding case and not firing for a genuinely-clear one -- both verified directly in `test_candidate_evaluator.py`/`test_frenet_planner.py`. A tighter oriented-bounding-box/SAT check can be layered in later (e.g. Stage 3-D/E) without changing this module's status contract.

5. **Curvature/jerk limits**: `max_curvature_per_m = 0.3` and `max_longitudinal_accel_mps2 = 6.0` are reused DIRECTLY from `src/environment/low_level_controller.py`'s existing `MAX_STEERING_CURVATURE`/`MAX_ACCEL_MPS2` (themselves `InvertibleBicycleModel`'s own default bounds) -- not independently re-derived -- specifically so a trajectory this planner marks `OK` is guaranteed physically realizable by the one dynamics model it will actually be executed on later (Stage 3-D/E). `max_longitudinal_jerk_mps3 = 20.0` has no prior repo precedent; it is reasoned from the two borrowed bounds (a full 0 to 6.0 m/s^2 ramp within one 0.1s MPC control step already implies ~60 m/s^3 for that single transition; 20 m/s^3 rejects genuinely violent jerk spikes across the whole 3s trajectory while remaining well above what the four actions' own smooth quintic/quartic profiles produce in normal operation, per the full-suite test run). All four numbers, and their justification, live in `configs/phase3_downstream.yaml`'s own comments, not hardcoded in source.

**Self-audit: no hidden behavior-decision logic (gate criterion 3).**

- `grep -n "gap_safe|FindRoundaboutGap|FindHighwayMergeGap|is_safe|safety_gate|should_merge|merge_allowed" src/planning/candidate_generator.py src/planning/candidate_evaluator.py src/planning/frenet_planner.py` returns NO matches (confirmed during this stage's own completion check).
- `frenet_planner.plan()`'s action dispatch (lines with `if action == BehaviorAction.KEEP / elif ... FOLLOW / elif ... MERGE / elif ... STOP`) is a flat, one-way dispatch to that action's OWN dedicated generator call -- there is no code path anywhere in `frenet_planner.py`, `candidate_generator.py`, or `candidate_evaluator.py` that reassigns `action` to a different `BehaviorAction`, or that falls through from one action's branch into another's geometry.
- When `evaluate_candidate` returns anything other than `OK` (`PLANNER_INFEASIBLE`/`COLLISION_BLOCKED`/`INVALID_REFERENCE`), `plan()` returns that exact status immediately, still carrying the ORIGINALLY-generated candidate's own `frenet_path`/`cartesian_trajectory` (see `frenet_planner.py`'s final `if evaluation.status != PlannerStatus.OK: return PlanResult(status=evaluation.status, ...)` branch) -- it never substitutes a different action's trajectory and never silently reports success. Verified directly by `test_frenet_planner.py::test_collision_blocked_on_merge_returns_explicit_status_not_silent_fallback` (confirms MERGE's own terminal `d≈0` geometry is still present on a `COLLISION_BLOCKED` result, not replaced by KEEP/FOLLOW/STOP source-lane-bound geometry) and `test_candidate_evaluator.py::test_collision_blocked_status_is_not_silently_reclassified_as_ok`.
- MERGE is generated with the exact same code path shape as FOLLOW (`generate_follow_or_merge_candidate`) and evaluated with the exact same `evaluate_candidate` function as every other action -- there is no MERGE-specific gap-selection, gap-midpoint, or "is this gap safe" computation anywhere (ASMC's `FindRoundaboutGap`/`FindHighwayMergeGap`-equivalent logic, flagged out-of-scope by the Stage 3-0 audit, was never ported).

**Test results:**

`PYTHONPATH=. pytest tests/planning/ -q` -> **80 passed** (51 unchanged Stage 3-A/3-B tests + 29 new Stage 3-C: 9 `test_candidate_generator.py` + 9 `test_candidate_evaluator.py` + 12 (incl. 2 real-WOMD) `test_frenet_planner.py`), 0 failed, run in the `its-merge` conda env, ~4.2s.

Full suite: `PYTHONPATH=. pytest tests/ -q` -> **357 passed, 0 failed, 0 errors** in 1497.10s (24m57s), confirming exactly 328 (Stage 3-A+3-B baseline) + 29 (new Stage 3-C) = 357, i.e. zero regressions.

**Gate status -- all criteria met:**

| Criterion | Result |
|---|---|
| (1) All four actions generate valid, physically-distinct trajectories on synthetic feasible cases | Confirmed -- `test_candidate_generator.py`/`test_frenet_planner.py`'s KEEP-vs-STOP, MERGE-vs-KEEP-vs-STOP tests |
| (2) Explicit infeasibility/collision statuses work correctly, never silently converted to a different action's geometry | Confirmed -- see self-audit above and the dedicated `COLLISION_BLOCKED`/`PLANNER_INFEASIBLE`/`INVALID_REFERENCE` tests in all three new test files |
| (3) No hidden behavior-decision logic | Confirmed -- see self-audit above (grep + code-path citation) |
| (4) Phase 2 frozen contracts unchanged | Confirmed -- `behavior_action.py`, `observation_builder.py` never modified; only imported/read |
| (5) Full existing test suite still passes | 357/357 passed (1497.10s) |
| (6) No Phase 2/3-A/3-B file modified | Confirmed -- `git diff --stat HEAD` empty; only new untracked files under `src/planning/`, `tests/planning/`, `configs/` |

**Known issues / follow-ups for Stage 3-D:**
- `FrenetPath`'s `valid`/`rejection_reason` fields (defined in Stage 3-B, unused placeholders) remain unused -- this stage's own `EvaluationResult`/`PlannerStatus` carry the equivalent information at the evaluator level instead; Stage 3-D/E may choose to also populate the `FrenetPath` fields directly if that proves convenient for the MPC layer, but nothing in Stage 3-C required it.
- The collision check's circular-proxy simplification (design decision 4 above) is a documented, deliberate simplification for THIS stage's correctness bar; if Stage 3-D/E's MPC layer needs tighter collision geometry, that is a natural place to introduce an oriented-bounding-box check without changing `candidate_evaluator.py`'s external status contract.
- `frenet_planner.py` is standalone and NOT wired into `MergeEnvironment` yet (by design -- Stage 3-F's job).

**Commit:** `PENDING`
