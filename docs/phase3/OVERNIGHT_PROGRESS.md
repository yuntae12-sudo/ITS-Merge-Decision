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
| 3-C BehaviorAction Execution Mapping | COMPLETE | `aafbc2f` |
| 3-D LTV-MPC | COMPLETE | `a367451` |
| 3-E Waymax Adapter / Common Downstream | COMPLETE | `4fd2c0e` |
| 3-F MergeEnvironment Integration | COMPLETE | `78a9faa` |
| 3-G Robustness/Regression | COMPLETE | `6be3452` |

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

**Commit:** `aafbc2f`

### Stage 3-D: Waymax-native Python LTV-MPC controller

**Files added:**
- `src/control/__init__.py`
- `src/control/mpc_types.py` -- `ControllerState` (x, y, yaw, speed), `ControllerCommand` (acceleration_mps2, steering_curvature), `ReferencePoint`, `ControllerStatus` (OK/INVALID_INPUT/SOLVER_FAILURE, mirroring Stage 3-C's `PlannerStatus` pattern), `MpcResult`. Reuses Waymax's own bounds (`MAX_ACCEL_MPS2=6.0`, `MAX_STEERING_CURVATURE=0.3`) and locked `MPC_DT_S=0.1`.
- `src/control/vehicle_model.py` -- one-step Waymax-native forward update. Module docstring quotes the EXACT installed `InvertibleBicycleModel.compute_update` equation verbatim (position/yaw/velocity update, `delta_yaw = steering*(speed*t + 0.5*accel*t**2)`), no wheelbase.
- `src/control/linearization.py` -- analytic Jacobians (A 4x4, B 4x2, affine offset c) of the forward model, derived directly from the substituted `new_x = x + cos(yaw)*k`, `new_y = y + sin(yaw)*k` form (`k = speed*t + 0.5*accel*t^2`).
- `src/control/mpc_cost.py` -- `CostWeights`, `bounds_for_horizon`, nonlinear `rollout`, and `total_cost` (position/heading/speed tracking + control effort + control-rate smoothness + terminal cost, angle-wrapped heading error).
- `src/control/constraints.py` -- thin re-export of the box bounds/`bounds_for_horizon` from `mpc_cost.py` (folded together per the task brief's "merge small modules if genuinely cleaner" guidance: the constraint set is just two box bounds, not enough for its own substantial module).
- `src/control/ltv_mpc.py` -- `LtvMpcController` (`solve`, `reset`), `MpcConfig`, `load_mpc_config`. Analytic discrete-adjoint gradient (`_analytic_gradient`) built from `linearization.linearize`'s per-step Jacobians, re-evaluated along the current nonlinear rollout each cost/gradient call (successive-linearization / LTV pattern), fed to `scipy.optimize.minimize(method="L-BFGS-B", jac=_analytic_gradient, bounds=...)`.
- `tests/control/__init__.py`, `tests/control/test_vehicle_model.py` (13 tests), `tests/control/test_linearization.py` (35 tests), `tests/control/test_ltv_mpc.py` (13 tests) -- 61 new tests total.
- `configs/phase3_downstream.yaml` -- ADDED a new `mpc:` section (horizon_steps, dt_s, max_iterations, weights) at the end of the file; Stage 3-C's existing `planner:`/`feasibility:`/`collision:` sections untouched (confirmed via `git diff` -- purely additive).

**Key design decisions:**
1. **Waymax forward-equation match**: read fresh from the installed `waymax/dynamics/bicycle_model.py` (not from memory/prior summary). Quoted verbatim in `vehicle_model.py`'s module docstring. Two subtleties documented there: (a) Waymax's `Trajectory` stores `vel_x`/`vel_y` independently of `yaw`, but this repo's 4-scalar `ControllerState` reconstructs `vel_x = speed*cos(yaw)`, `vel_y = speed*sin(yaw)` -- valid because Waymax's OWN output re-derives `new_vel_x = new_vel*cos(new_yaw)`, `new_vel_y = new_vel*sin(new_yaw)` at the end of every step, so the two representations stay exactly consistent step-over-step (verified by the equivalence test, not just asserted). (b) `new_vel = speed + accel*t` is UNCLAMPED -- Waymax allows speed to go negative (vehicle facing `new_yaw` but moving backward relative to it); this module reproduces that exactly, no artificial floor at zero.
2. **Model-equivalence verification method**: built minimal valid `waymax.datatypes.Trajectory`/`Action` objects (shape (1,1), matching `compute_update`'s `(num_objects, num_timesteps=1)` convention) and called the REAL installed `InvertibleBicycleModel.compute_update` directly, across 10 representative samples spanning speed 0-20 m/s, accel in [-6,6], steering in [-0.3,0.3], several yaw values including near the +-pi wrap boundary, and an explicit near-zero-speed/large-decel case that drives speed negative. **A genuine discrepancy was found and investigated during test development**: the test's own helper initially extracted output speed via `hypot(vel_x, vel_y)` (always non-negative), which silently disagreed with this module's signed speed whenever Waymax's own `new_vel` went negative (case: speed=0.5, accel=-6.0 -> new_vel=-0.1, but Waymax re-expands this as `vel_x = new_vel*cos(new_yaw)`, `vel_y = new_vel*sin(new_yaw)`, so `hypot` recovers the WRONG-SIGN magnitude +0.1). Root-caused as a test-helper bug (not a model bug): fixed by recovering the signed speed as `vel_x*cos(new_yaw) + vel_y*sin(new_yaw)` (projection onto the heading unit vector) instead of `hypot`. After the fix, **all 10 samples agree with the real installed Waymax model within `abs=1e-4`** (limited by float32 vs float64 precision -- Waymax's own `Trajectory`/`Action` dtypes are float32 -- not by any approximation in the equation itself; this is the expected, documented, non-loosened tolerance for "same equation, different float width," not a weakened correctness bar).
3. **Jacobian derivation**: analytic, by direct differentiation of the substituted closed form `new_x = x + cos(yaw)*k`, `new_y = y + sin(yaw)*k`, `k = speed*t + 0.5*accel*t^2` (see `linearization.py`'s full derivation in its module docstring). Verified against central-difference numerical Jacobians (eps=1e-6) across 25 random (state, control) samples spanning the same realistic ranges as the equivalence test (yaw sampled away from the +-pi branch cut, where wrap is non-differentiable) -- **all pass at `atol=1e-5, rtol=1e-4`**. A second test confirms the affine form `A@x + B@u + c` reproduces the exact nonlinear step at the linearization point to `atol=1e-9`.
4. **Optimizer choice**: `scipy.optimize.minimize(method="L-BFGS-B")`. Justified by the constraint shape: only simple per-step box bounds on (accel, steering), no other equality/inequality constraints -- L-BFGS-B is the standard, cheaper SciPy method for exactly this shape versus a general constrained solver (SLSQP/trust-constr). Cost is evaluated via a direct forward-simulation (`mpc_cost.rollout`) of the ACTUAL nonlinear model (not the LTV-linearized one) -- simpler and still correct, per the task brief's explicit permission. An ANALYTIC gradient (`_analytic_gradient`, a discrete-adjoint/reverse-mode chain-rule pass using `linearization.linearize`'s Jacobians re-evaluated along the current nonlinear iterate) is supplied instead of relying on SciPy's finite-difference default, since the analytic Jacobians were already required/validated by 3-D.2 and are materially cheaper (O(N) Jacobian evaluations vs O(N) extra cost evaluations per gradient call). This is the "LTV" in LTV-MPC: the prediction model is re-linearized at each iterate along its own trajectory, even though cost itself is evaluated on the true nonlinear rollout. No JAX optimization was needed -- SciPy's L-BFGS-B with an analytic gradient met the correctness bar directly.
5. **Warm-start/reset**: previous horizon-length solution shifted by one step (drop control[0], repeat control[-1]) becomes the next solve's initial guess; default is all-zero when no warm-start exists. `reset()` clears both `_warm_start` and `_previous_command` explicitly -- verified by test that a solve immediately after `reset()` produces bit-identical output (`atol=1e-6`) to a brand-new controller instance's first solve. No stuck-recovery/fail-safe heuristic (explicitly out of scope per Stage 3-0 audit).
6. **Cost weights**: all in `configs/phase3_downstream.yaml`'s new `mpc:` section, explicitly documented in-file as "INITIAL Phase 3 controller parameters... NOT final, policy-tuned hyperparameters." No ASMC MORAI-tuned numeric weight was ported. Position (10.0) > heading (5.0) > speed (2.0) tracking, light effort (0.05)/rate (0.1) regularization, terminal weights ADD to per-step weights at the last horizon step (terminal_pos=20.0, terminal_yaw=10.0, terminal_speed=5.0). No control-rate CONSTRAINT was added (only the rate-smoothness cost term) -- the synthetic tracking tests were empirically stable with box bounds + rate cost alone, so per the task brief's explicit instruction, no rate constraint was added speculatively.
7. **Failure handling**: `ControllerStatus.INVALID_INPUT` returned before any solve is attempted if `ControllerState` or any `ReferencePoint` is non-finite (or the reference list length doesn't match `config.horizon`). `ControllerStatus.SOLVER_FAILURE` returned if the solver's raw output is non-finite or violates box bounds beyond a 1e-6 numerical tolerance -- no hidden fallback command is ever substituted.

**Commands run:**
- `python3 -m pytest tests/control/test_vehicle_model.py -v` -> 13 passed (includes the critical model-equivalence test, see design decision 2 above for the investigated-and-fixed discrepancy).
- `python3 -m pytest tests/control/test_linearization.py -q` -> 35 passed (includes the critical Jacobian-agreement test).
- `python3 -m pytest tests/control/test_ltv_mpc.py -v` -> 13 passed (straight/arc-left/arc-right/speed-up/speed-down/lateral-offset/heading-offset tracking, saturation, non-finite rejection x2, warm-start, reset, config-load).
- `python3 -m pytest tests/control/ -q` -> **61 passed** in ~22s.
- Full suite: `python3 -m pytest tests/ -q` -> **418 passed, 0 failed, 0 errors** in 1539.08s (25m39s), confirming exactly 357 (Stage 3-A/B/C baseline) + 61 (new Stage 3-D) = 418, i.e. zero regressions.

**Gate verification:**

| Criterion | Result |
|---|---|
| (1) Synthetic tracking scenarios converge sensibly | Confirmed -- straight/both arcs/both offsets/both speed changes all pass in `test_ltv_mpc.py` |
| (2) All commands finite and respect box bounds | Confirmed -- saturation test explicitly checks `abs(command) <= bound + 1e-9` on both the first command and the full horizon sequence |
| (3) Waymax one-step model-equivalence test passes | Confirmed -- 13/13 in `test_vehicle_model.py`, agreement within `abs=1e-4` (float32/float64 precision limit, not an approximation in the equation); one genuine test-helper bug (signed-speed extraction via `hypot`) found and fixed during development, documented in design decision 2 |
| (4) Analytic-vs-numerical Jacobian test passes | Confirmed -- 35/35 in `test_linearization.py` at `atol=1e-5, rtol=1e-4` |
| (5) Full existing test suite still passes | 418/418 passed (1539.08s) |
| (6) No Phase 2/3-A/3-B/3-C file modified | Confirmed -- `git status --short` shows only `M configs/phase3_downstream.yaml` (additive `mpc:` section) plus new untracked `src/control/`, `tests/control/` |

**Known issues / follow-ups for Stage 3-E:**
- `ltv_mpc.py` is standalone and NOT wired into Stage 3-C's `frenet_planner.py`/`CartesianTrajectory` output yet (by design -- Stage 3-E's job). `ReferencePoint` intentionally mirrors only the four `CartesianTrajectory` fields the cost function needs (x, y, yaw, velocity); Stage 3-E will need to build the `CartesianTrajectory -> List[ReferencePoint]` adapter and confirm dt/horizon alignment holds for real planner output (this stage only verified it against hand-built synthetic references).
- No resampling machinery was built for reference-trajectory alignment -- Stage 3-C's planner already produces dt=0.1s-spaced trajectories matching the MPC's own locked dt 1:1, so Stage 3-E should be able to index directly; if a future planner config changes `dt_s`, this assumption should be re-checked then.
- No control-rate constraint (only a rate-smoothness cost term) -- empirically sufficient for this stage's synthetic tests; if Stage 3-E/F's real closed-loop integration surfaces chattering the cost term alone doesn't damp, a rate constraint is the natural next step.
- Cost weights are explicitly initial/non-final (see design decision 6) -- expected to need retuning once wired into real closed-loop scenarios (Stage 3-E/F) and eventually PPO training (Stage 3-H, out of scope here).

**Commit:** `a367451`

**Next stage: 3-E** (wire Stage 3-C's `CartesianTrajectory` planner output into this MPC controller).

### Stage 3-E: Common Downstream / Waymax Adapter

**Date:** 2026-09-14

**Files added:**
- `src/environment/common_downstream.py` -- `CommonDownstream` (`step`, `reset`), `DownstreamRequest`, `DownstreamResult`, `DownstreamStatus`. Composes Stage 3-C's `frenet_planner.plan` and Stage 3-D's `LtvMpcController.solve` into one entry point. Does NOT wire into `MergeEnvironment` (explicitly Stage 3-F).
- `tests/environment/test_common_downstream.py` -- 12 new tests.

**Key design decisions:**

1. **No fallback command on failure (locked for this stage, deferred to 3-F).** When either the planner (`PlannerStatus != OK`) or the controller (`ControllerStatus != OK`) reports a failure, `CommonDownstream.step` returns `command=None` and an explicit non-OK `DownstreamStatus` -- it never invents a bounded-braking or hold-last-command fallback. Rationale: `MergeEnvironment`/Waymax's actual numerical-liveness requirements (does `waymax_env.step` need SOME physical action every call, even on a failure step?) are Stage 3-F's decision to make deliberately, with the real Waymax stepping loop in front of it -- inventing a fallback shape here, before that need is proven, risks quietly building exactly the kind of policy-blind "silent substitution" the Stage 3-0 fairness audit repeatedly flags. If Stage 3-F needs a placeholder command to keep stepping alive, it should choose and justify that design against the real integration constraints, not inherit an untested guess from this stage.
2. **Status wrapping/propagation.** `DownstreamStatus` (OK / INVALID_REFERENCE / PLANNER_INFEASIBLE / COLLISION_BLOCKED / CONTROLLER_FAILURE) is a direct 1:1 wrap of Stage 3-C's `PlannerStatus` (three non-OK values map straight across) plus a single `CONTROLLER_FAILURE` that collapses BOTH of Stage 3-D's `ControllerStatus.INVALID_INPUT`/`SOLVER_FAILURE` (the specific upstream value is preserved in `DownstreamResult.diagnostics["controller"]["status"]` for diagnostics, but Stage 3-E itself does not need to distinguish the two for its own control flow -- both mean "no command available this step"). The planner is always tried first; the MPC's `solve()` is called ONLY when `plan()` returned `OK` -- verified directly by three separate spy-based tests (`test_collision_blocked_propagates_and_mpc_never_invoked`, `test_planner_infeasible_propagates_and_mpc_never_invoked`, `test_invalid_reference_propagates_and_mpc_never_invoked`) that monkeypatch `LtvMpcController.solve` with a call-counting spy and assert zero invocations.
3. **MPC reference-horizon construction.** Stage 3-C's `CartesianTrajectory` places index 0 at the ego's CURRENT state (`t[0] == 0.0`, confirmed empirically against a real `plan()` call before writing this module), while Stage 3-D's `LtvMpcController.solve` wants exactly `mpc_config.horizon` FUTURE `ReferencePoint`s. `common_downstream.py` therefore slices trajectory indices `[1, horizon]` inclusive (skipping index 0) to build the MPC reference list, and returns `CONTROLLER_FAILURE` explicitly (rather than crashing or silently padding/truncating) if the planner's trajectory has fewer than `horizon + 1` samples -- covered by `test_controller_failure_via_genuinely_non_finite_reference_horizon_precondition`. With the shipped `configs/phase3_downstream.yaml` (planner `trajectory_horizon_s=3.0s`/`dt_s=0.1s` -> 31 samples; MPC `horizon_steps=20`), this precondition always holds in normal operation.
4. **No policy-type dependency.** `CommonDownstream.__init__`/`step`, `DownstreamRequest`, and `DownstreamResult` were audited (both by direct code inspection while writing the module, and by an automated structural test, `test_no_policy_type_parameter_anywhere_in_public_api`, that inspects `inspect.signature`/`dataclasses.fields` for any parameter/attribute name containing a policy-origin-like substring such as "policy"/"fsm"/"ppo"/"origin") to confirm no such parameter exists anywhere in the public API. `DownstreamRequest`'s field set is asserted to be EXACTLY `{behavior_action, objective, ego_state, source_reference, target_reference, follow_inputs, surrounding_agents, dt_s}` -- nothing else.
5. **Determinism.** Verified by `test_determinism_identical_input_produces_byte_identical_output`: identical `DownstreamRequest` fed to two INDEPENDENTLY-constructed `CommonDownstream` instances (each owning its own fresh `LtvMpcController`, so no warm-start state is shared) produces byte-identical `ControllerCommand` values (`==` on both float fields, not just `np.isclose`). Additionally verified that the SAME instance, after an explicit `reset()` (which delegates to `LtvMpcController.reset()`, clearing `_warm_start`/`_previous_command`), reproduces the identical first-call result -- confirming the only state `CommonDownstream` depends on is the MPC's own warm-start, and that `reset()` fully clears it.

**Commands run:**
- `python3 -m pytest tests/environment/test_common_downstream.py -q` -> 12 passed (~0.5s).
- Full suite: `python3 -m pytest tests/ -q` -> **430 passed, 0 failed, 0 errors** in 1563.75s (26m03s), confirming exactly 418 (Stage 3-A/B/C/D baseline) + 12 (new Stage 3-E) = 430, i.e. zero regressions.

**Gate verification:**

| Criterion | Result |
|---|---|
| (1) Determinism test passes | Confirmed -- `test_determinism_identical_input_produces_byte_identical_output` |
| (2) No-policy-type-dependency check passes | Confirmed -- `test_no_policy_type_parameter_anywhere_in_public_api` |
| (3) All four `BehaviorAction`s accepted, coherent (status, optional-command) result | Confirmed -- `test_all_four_behavior_actions_produce_coherent_result` (parametrized over KEEP/FOLLOW/MERGE/STOP) |
| (4) Planner failure propagates, MPC never invoked | Confirmed -- 3 spy-based tests (collision/infeasible/invalid-reference), zero `solve()` calls in each |
| (5) Controller failure propagates as `CONTROLLER_FAILURE` | Confirmed -- `test_controller_failure_propagates_as_controller_failure_status` (forced `INVALID_INPUT`) and `test_controller_failure_via_genuinely_non_finite_reference_horizon_precondition` (too-short trajectory) |
| (6) Command physical units/shape correct, unmangled through the wrapper | Confirmed -- `test_successful_command_within_waymax_physical_bounds` checks `[-6.0, 6.0]`/`[-0.3, 0.3]` bounds and cross-checks the value is reproduced exactly across an independent re-solve |
| (7) Full existing test suite still passes | 430/430 passed (1563.75s) |
| (8) No Phase 2/3-A/3-B/3-C/3-D file modified | Confirmed -- `git status --porcelain` showed only the two new untracked files before staging |

**Known issues / follow-ups for Stage 3-F:**
- `CommonDownstream` is standalone and NOT wired into `MergeEnvironment` yet (by design -- Stage 3-F's job). Stage 3-F will need to: (a) decide whether a failure-step fallback command is actually necessary to keep `waymax_env.step` callable, and if so, design it deliberately rather than inheriting a guess from this stage (see design decision 1 above); (b) construct `DownstreamRequest.ego_state`/`source_reference`/`target_reference`/`surrounding_agents`/`follow_inputs` from `MergeEnvironment`'s existing per-step state (`EpisodeContext`, `MergeReference`, the 14D observation's causal fields, etc.); (c) own one `CommonDownstream` instance per episode and call `reset()` at episode boundaries, exactly mirroring `LtvMpcController`'s own documented per-episode-instance pattern; (d) convert the resulting `ControllerCommand` into the exact `WaymaxAction` construction already used in `merge_environment.py:273-354` (`np.array([command.acceleration_mps2, command.steering_curvature], dtype=np.float32)`, `valid=np.array([True], dtype=bool)`).
- `DownstreamResult.diagnostics` is a plain nested dict (`{"planner": ..., "controller": ...}` or `{"planner": ..., "error": ...}`), not a frozen dataclass -- kept simple since Stage 3-E has no consumer of this data yet beyond tests; Stage 3-F/logging code may want a more structured diagnostics type if it needs to aggregate these across an episode.

**Commit:** `4fd2c0e`

### Stage 3-F: MergeEnvironment integration (frenet_mpc, opt-in only)

**Date:** 2026-09-14

**Files changed:**
- `src/environment/merge_environment.py` -- **MODIFIED** (first Stage 3
  change to an existing Phase 2 production file; every prior Stage
  3-A through 3-E change was additive-only new files). Additive,
  careful change: added a `downstream_mode: str = "legacy"`
  constructor parameter (plus `downstream_config_path`), a thin
  `if self._downstream_mode == "legacy": ... else: ...` branch in
  `step()` around the objective->command computation, three new
  private helper methods (`_get_active_reference_lines`,
  `_build_surrounding_agents`, `_compute_frenet_mpc_command`), a
  `reset()` addition (construct/reset `CommonDownstream` + build
  initial reference pair, gated on `downstream_mode == "frenet_mpc"`),
  and two new `info` fields (`downstream_status`,
  `downstream_reference_rebuilt`, both `None` in legacy mode). The
  **default remains `"legacy"`** -- every existing caller (both
  `test_merge_environment.py` and `test_common_state_freeze.py`, which
  never pass this argument) exercises the EXACT pre-existing code
  path, confirmed by re-running both files UNCHANGED (27/27 passed,
  see below) and by inspecting the diff directly: the legacy branch's
  five lines (`reference_polyline = ...` through
  `command = self._controller.compute_command(...)`) are byte-
  identical to before, only re-indented one level under the new `if`.
- `tests/environment/test_merge_environment_frenet_mpc.py` -- NEW, 19
  tests for the new opt-in `downstream_mode="frenet_mpc"` path. Does
  NOT modify `test_merge_environment.py`'s existing assertions.

**Key design decisions:**

1. **Fallback-command design (Stage 3-E deliberately deferred this to
   3-F; see that stage's design decision #1).** On ANY non-OK
   `DownstreamStatus` (planner infeasible/collision-blocked/invalid
   reference, or controller failure), `_compute_frenet_mpc_command`
   returns a bounded braking command:
   `acceleration_mps2 = -MAX_ACCEL_MPS2` (`-6.0`, reusing the SAME
   constant `low_level_controller.py` and Stage 3-C's own feasibility
   config already use -- not a new, independently-chosen number),
   `steering_curvature = 0.0` (hold heading, since there is no valid
   planned trajectory to steer toward). Chosen because: (a) Waymax's
   `PlanningAgentEnvironment.step` genuinely needs SOME physical
   action every call to remain steppable -- confirmed by direct
   inspection, this is not optional; (b) a FULL-magnitude brake (not a
   partial/tuned value) is the most conservative, least-assumption
   choice when the planner/controller could not produce anything
   trustworthy -- inventing a gentler heuristic would itself be an
   unjustified assumption about what "close enough to safe" means
   when the actual planned geometry is unknown/rejected; (c) it is
   IDENTICAL regardless of policy origin (FSM vs PPO) -- this module
   has no notion of policy type, matching Stage 3-E's own design; (d)
   it NEVER changes `requested_action`/`executed_action` in `info` --
   those are computed from `DecisionState.advance(action)` BEFORE the
   downstream is ever called, and the fallback only substitutes the
   PHYSICAL command applied this frame -- verified directly by
   `test_downstream_failure_fallback_never_alters_requested_or_executed_action`
   and `test_fallback_command_finite_and_bounded_on_forced_controller_failure`
   (both force a failure via monkeypatching `CommonDownstream.step`
   and confirm `requested_action`/`executed_action` are unaffected);
   (e) the real failure status is always surfaced via the new
   `info["downstream_status"]` field -- never silently reported as a
   normal successful step.

2. **Reference-line caching/invalidation.** `_get_active_reference_lines`
   caches Stage 3-A `ReferenceLine`s keyed by the EXACT
   `(active_source_lane_id, active_target_lane_id)` pair
   `EpisodeContext` already exposes -- no second, independent
   chain-tracking mechanism was built. The cache is populated lazily
   (built on first use per pair) and invalidated in exactly two
   places: (a) `reset()`, which clears the whole cache and eagerly
   rebuilds the initial pair; (b) `step()`'s `chain_advanced` branch
   (fired by the EXISTING, untouched `_maybe_advance_chain()` return
   value, called in the SAME position it already was, right after the
   Waymax step), which evicts only the now-stale pair (computed via
   `lane_chain[active_transition_index - 1], lane_chain[active_transition_index]`
   -- i.e. exactly the (old_source, old_target) pair just used, since
   `active_transition_index` has already been incremented by
   `advance_if_intermediate_reached` at that point) and calls
   `CommonDownstream.reset()` to clear the MPC's warm-start, so the
   next solve is not biased toward the old geometry's control
   sequence. The new pair is rebuilt lazily on the NEXT step via the
   same `_get_active_reference_lines()` call, per the task brief's
   explicit instruction not to force an eager rebuild inside the
   chain-advanced branch itself. This mechanism is verified
   deterministically and controller-independently by
   `test_reference_rebuild_fires_on_chain_advanced_frenet_mpc`, which
   monkeypatches `_maybe_advance_chain` to force a controlled
   advancement (spying on `CommonDownstream.reset` and inspecting the
   cache directly) -- see "known issue" below for why this had to be
   decoupled from the real end-to-end `CHAINED_MANEUVER` rollout.

3. **Causal-input reuse (no second gap/TTC derivation).**
   `_compute_frenet_mpc_command` builds `FollowInputs` DIRECTLY from
   `observation_before` -- the SAME 14D observation array
   `BehaviorExecutor.compute_objective` already consumed this exact
   step (indices 2/3/4 for target front, 10/11/12 for source front,
   per `observation_builder.OBSERVATION_FIELD_NAMES`) -- never a
   second, independently-derived set of gap/relative-speed numbers.
   Surrounding-agent state (`_build_surrounding_agents`) is read
   directly from `self._state.current_sim_trajectory`/
   `object_metadata`, the same per-frame arrays `_build_observation`
   itself reads, excluding only the ego id and invalid slots -- no
   separate simulation-state re-derivation. This directly satisfies
   the Stage 3-0 audit's fairness constraint restated in the task
   brief: the 14D observation and the planner's causal inputs come
   from the identical underlying computation.

4. **Legacy-mode byte-identical verification method.** Not just "the
   tests still pass" -- the diff itself was inspected line-by-line
   (`git diff src/environment/merge_environment.py`) to confirm the
   legacy branch's body is a verbatim, only-reindented copy of the
   pre-Stage-3-F code (no numeric literal, call argument, or ordering
   changed), AND the full unmodified
   `tests/environment/test_merge_environment.py` +
   `test_common_state_freeze.py` suites (27 tests total, including
   exact-equality determinism tests like
   `test_reset_is_deterministic`/`test_same_action_sequence_produces_same_rollout`
   and the numeric-inequality causality test
   `test_keep_and_stop_produce_different_controller_commands`) were
   re-run UNCHANGED and pass, which would catch even a subtle
   behavioral drift in the legacy path.

5. **`CommonDownstream`/planner/MPC config construction is entirely
   gated on `downstream_mode == "frenet_mpc"`** -- a `MergeEnvironment`
   constructed with the default `legacy` mode never loads
   `configs/phase3_downstream.yaml`, never constructs a
   `CommonDownstream`/`LtvMpcController`, and never imports/executes
   any Stage 3-A/B/C/D/E code path at runtime (the modules are
   imported at module-load time, per normal Python import semantics,
   but no Stage 3 object is ever instantiated or called unless
   `frenet_mpc` is explicitly requested) -- zero runtime cost/behavior
   change for the default path.

**Known issue found during Stage 3-F test development (not a bug in
this stage's own code, a real finding about Stage 3-C's planner +
CHAINED_MANEUVER's specific real geometry):** the real chained
maneuver `CHAINED_MANEUVER` (used successfully by the LEGACY suite's
own chain-advancement tests, specifically because it is documented
there as "confirmed by direct rollout to reach success reliably"
*under the legacy P-controller*) does NOT reliably drive
`active_transition_index` forward under the frenet_mpc downstream
within a reasonable step budget. Root-caused via direct diagnostic:
at episode start ego is still physically in the SOURCE lane, so
projecting ego's current pose into the SHORT intermediate TARGET
lane's own Frenet frame (Stage 3-C's approach (b), a prior, already-
approved design decision, not something this stage should silently
change) yields a huge out-of-domain `(s, d)` (observed `s≈-44.6,
d≈-44.6` against a lane only ~20.5m long); Stage 3-B's documented
open-path linear extrapolation then produces a Cartesian candidate
trajectory far from both ego's real position and the target lane's
own geometry, which Stage 3-C's constant-velocity collision proxy
then spuriously flags as colliding with an unrelated, distant real
vehicle at `t=0`. This is consistent with `merge_reference.py`'s own
Stage B-1.5 documented history of the identical class of problem
(ego far from the target lane's own domain at commitment time), which
is exactly why the LEGACY path uses a blended source/target reference
instead of a raw target-frame projection. Since Stage 3-C's
target-frame-projection approach for MERGE was already an approved,
documented Stage 3-C decision (not this stage's to relitigate), Stage
3-F does not alter it; instead, the chain-advancement/reference-
rebuild MECHANISM itself (the only thing actually owned by this
stage) is verified deterministically and controller-independently via
`test_reference_rebuild_fires_on_chain_advanced_frenet_mpc` (forces
`_maybe_advance_chain` to fire via monkeypatching, independent of
whether the real MPC trajectory converges), while
`test_chained_maneuver_advances_and_rebuilds_reference_frenet_mpc`
still runs the real `CHAINED_MANEUVER` end-to-end and asserts the
diagnostic is well-formed on every step and correct IF an advance
occurs, without requiring one to occur. Flagged here explicitly as a
known limitation for a future stage (candidate improvement: a smarter
MERGE reference-frame choice for short/near-start intermediate
transitions, or a less literal constant-velocity collision proxy) --
out of Stage 3-F's own scope (wiring `CommonDownstream` into
`MergeEnvironment`, not modifying Stage 3-C's planner).

**Commands run:**
- `pytest tests/environment/test_merge_environment.py tests/environment/test_common_state_freeze.py -q`
  -> **27 passed** in 970.53s (16m10s) -- unchanged legacy-mode
  regression suite, confirming zero behavior change in the default
  path.
- `pytest tests/environment/test_merge_environment_frenet_mpc.py -v`
  -> **19 passed** in 162.98s (2m43s) -- new frenet_mpc-mode
  integration tests.
- Full suite: `pytest tests/ -q` -> **449 passed, 0 failed, 0 errors**
  in 1689.95s (28m09s), confirming exactly 430 (Stage 3-A/B/C/D/E
  baseline) + 19 (new Stage 3-F) = 449, i.e. zero regressions.

**Gate verification:**

| Criterion | Result |
|---|---|
| (1) All existing legacy-mode tests pass UNCHANGED | Confirmed -- 27/27 passed, zero assertion changes |
| (2) New frenet_mpc-mode integration tests pass | Confirmed -- 19/19 passed (KEEP/FOLLOW/MERGE/STOP causality, commitment lock, success-requires-commitment, chain advancement + reference rebuild (both real-rollout and deterministic-mechanism forms), collision/offroad propagation, 14D observation finiteness, command finiteness incl. fallback steps, fallback-never-alters-requested/executed-action, lightweight diagnostic comparison) |
| (3) No frozen Phase 2 contract changed | Confirmed -- `BehaviorAction`/`BehaviorObjective`/`DecisionState`/`EpisodeContext`/`observation_builder`/`termination.py` files untouched (only imported); `requested_action`/`executed_action`/MERGE-commitment/chain-advancement-ownership semantics verified unchanged by both the legacy regression suite and the new fallback-isolation tests |
| (4) No hidden action override on downstream failure | Confirmed -- `test_downstream_failure_fallback_never_alters_requested_or_executed_action` and `test_fallback_command_finite_and_bounded_on_forced_controller_failure` |
| (5) Chain-transition reference/warm-start reset works and is diagnostically visible | Confirmed -- `test_reference_rebuild_fires_on_chain_advanced_frenet_mpc` (deterministic) + `test_chained_maneuver_advances_and_rebuilds_reference_frenet_mpc` (real rollout, conditional on an advance occurring) |
| (6) Full existing test suite passes | 449/449 passed (1689.95s) |
| (7) Only `merge_environment.py` modified among existing files | Confirmed -- `git status --short` shows exactly `M src/environment/merge_environment.py` plus one new untracked test file; diff inspected directly to confirm the legacy branch is a verbatim reindent |

**Known issues / follow-ups for Stage 3-G:**
- The `CHAINED_MANEUVER` real-geometry limitation above (MERGE's
  target-frame projection producing a spurious collision-blocked
  status near a short intermediate lane at episode start) is a real,
  documented finding about Stage 3-C's planner's interaction with this
  specific real scene's geometry -- worth Stage 3-G's attention if
  robustness/regression testing surfaces it as a broader pattern
  across more real maneuvers, not just this one.
- Cost weights/feasibility limits are still the Stage 3-C/3-D initial,
  non-final values (explicitly documented as such in those stages) --
  expected to need retuning once exercised across a broader real-data
  robustness pass (Stage 3-G) and eventually PPO training (out of
  scope here).
- `downstream_reference_rebuilt`/`downstream_status` are `None` in
  legacy mode (by design, since the concepts don't apply there) --
  any future caller reading `info` across both modes should check
  `downstream_mode` (or tolerate `None`) rather than assume these
  fields are always populated.

**Commit:** `78a9faa`

### Stage 3-G: Robustness / Regression (FINAL STAGE OF THIS RUN)

**Date:** 2026-09-14

**Files added:**
- `scripts/audit_phase3_robustness.py` -- loads ALL 168 canonical
  maneuvers (110 TRAIN + 58 VALIDATION via
  `full_split_evaluator.load_maneuver_specs`) and, under
  `downstream_mode="frenet_mpc"`, drives a bounded rollout per
  maneuver with a fixed action script (`BehaviorAction.MERGE` every
  step, matching `test_merge_environment.py`'s own success-test
  precedent -- a robustness stimulus, not a policy). Records
  per-maneuver: exception (if any), NaN/Inf observation, all
  `downstream_status` occurrences broken down by value, chain-advanced
  count, final `active_transition_index` + out-of-bounds check,
  termination outcome, wall time. Writes a compact 168-row summary to
  `outputs/phase3/robustness/audit_168_maneuvers.{csv,json}` (no raw
  trajectory dumps) and prints the aggregate gate report.
- `scripts/audit_phase3_action_discriminability.py` -- Stage 3-G.4:
  runs KEEP/MERGE/STOP on a representative 8-maneuver TRAIN sample
  under `frenet_mpc`, reporting KEEP displacement, KEEP-vs-MERGE
  final-position divergence, and STOP speed delta (diagnostic
  print-only; the pass/fail gate for this property already lives in
  `tests/environment/test_merge_environment_frenet_mpc.py`'s
  `test_keep_produces_causal_forward_motion_frenet_mpc` /
  `test_merge_produces_lateral_motion_toward_target_frenet_mpc` /
  `test_stop_produces_deceleration_frenet_mpc`, which this script
  extends to a broader real-maneuver sample rather than duplicating).
- `scripts/audit_phase3_performance_and_legacy_comparison.py` -- Stage
  3-G.5 (perf snapshot via lightweight `time.perf_counter()`
  monkeypatch instrumentation around `ReferenceLine.from_lane_polyline`,
  `CommonDownstream.step`, `LtvMpcController.solve`, and
  `MergeEnvironment.step` itself -- no production code modified for
  timing) + 3-G.6 (runs the identical MERGE-repeatedly action script
  through both `downstream_mode="legacy"` and `="frenet_mpc"` on a
  12-maneuver TRAIN sample, printing termination reason/step-count
  side by side; explicitly diagnostic, no research conclusion drawn).
- `outputs/phase3/robustness/audit_168_maneuvers.csv` /`.json` -- the
  168-row compact per-maneuver summary (56KB/144KB; force-added past
  this repo's blanket `outputs/*` gitignore rule since these two files
  are exactly the small, deliberate deliverable Stage 3-G's own brief
  asked for, not incidental generated bulk output -- no other file
  under `outputs/` was added).

**Files fixed (genuine bug, in scope per this stage's explicit mandate
to fix systemic implementation bugs found during investigation):**

- `src/scenarios/lane_geometry.py` -- **`project_point_to_polyline_signed`
  bug fix.** This is a Phase 1 file (predates Stage 3 entirely), fixed
  here because Stage 3-C's MERGE target-frame projection
  (`cartesian_to_frenet`, via `ReferenceLine.project`) is the first
  and only production caller that queries this function from far
  outside a lane's own domain -- exactly the regime the bug lived in,
  unexercised by any existing test or other caller (confirmed by a
  dedicated caller/test audit before touching the file, see "Chained
  -maneuver bug investigation" below for the full root-cause trace).

**Chained-maneuver bug investigation (Stage 3-F's documented known
issue, this stage's explicit mandate to investigate and, if genuinely
systemic, fix):**

Root-caused via direct diagnostic on the real `CHAINED_MANEUVER`
(`MAN_0041`, lane_chain 618->623->608) fixture, reproducing Stage
3-F's own observation (`s ~= -44.6`, `d ~= -44.6` when projecting ego
into the short ~20.5m intermediate target lane 623's Frenet frame at
episode start). The critical follow-up measurement Stage 3-F's own
entry did NOT make: the CORRECT perpendicular lateral offset at that
exact real ego pose is only **-0.97 m** (ego is nearly on lane 623's
own extended centerline, having not yet made lateral progress toward
it) -- not anywhere near -44.6 m. The bug is in
`project_point_to_polyline_signed` itself (`src/scenarios/
lane_geometry.py`), in the two open-path extrapolation branches (query
point's nearest segment is the polyline's first/last segment AND the
raw projection fraction `t_raw` falls outside `[0, 1]`, i.e. the query
is genuinely off the polyline's domain along its own tangent). Those
branches already correctly computed `arc_length_m` from the UNCLAMPED
`t_raw` (extrapolating along the tangent, a deliberate Stage B-0 fix
per the function's own docstring), but `lateral_distance_m` was still
computed as `lateral_sign * distances[segment_index]`, where
`distances[segment_index]` is the Euclidean distance from the query
point to the CLAMPED projection point (the segment's own boundary
endpoint, `t_clamped` folded into `[0,1]`) -- NOT the true perpendicular
distance to the segment's infinite tangent line. Whenever a query
point is far along the tangent direction outside the polyline's
domain, this conflates that large along-tangent distance into what
should be a small perpendicular offset: verified by hand (`t_raw =
-45.578`, correct perpendicular lateral distance `-0.9736 m`, buggy
Euclidean-to-clamped-endpoint distance `44.616 m` -- matching the
corrupted value observed downstream to 5 significant figures). This
then fed a MERGE candidate whose lateral quintic was solved to
converge `d` from -44.6 to 0 over the fixed 3.0s trajectory horizon --
a physically nonsensical ~44m lateral sweep -- which both violated the
curvature feasibility bound and, because the resulting Cartesian
candidate trajectory swept through real, unrelated space far from
ego's actual position, spuriously triggered the constant-velocity
collision proxy against real but irrelevant agents. This explains
100% of the previously-observed `COLLISION_BLOCKED`/`PLANNER_
INFEASIBLE` loop that prevented `MAN_0041`'s chain from ever advancing
under frenet_mpc.

Before fixing: audited every other caller of
`project_point_to_polyline_signed` in the repo (`merge_reference.py`,
`merge_environment.py:_build_observation`, `scenario_features.py:
find_target_lane_front_rear`) -- all three use ONLY `arc_length_m`,
never `lateral_distance_m`; `frenet_transform.cartesian_to_frenet`
(via `ReferenceLine.project`) is the sole real production consumer of
the corrupted field. Audited every existing test touching this
function/branch (`test_lane_geometry.py`'s extrapolation tests,
`test_scenario_features.py`'s rear-ordering test, `test_reference.py`,
`test_frenet_transform.py`) -- every single one either queries only
`arc_length_m` in the far-extrapolation regime, or queries
`lateral_distance_m` only at points still interior to the polyline's
domain (never the far-extrapolation + off-centerline combination the
bug lived in). Zero tests assert on, or would break from a fix to,
this specific regime. The function's own docstring documents intent
only for `arc_length_m`'s unclamped behavior; it says nothing about
`lateral_distance_m` being distance-to-clamped-endpoint -- there is no
evidence this was ever a deliberate design choice, only an oversight
where the clamped-search `distances`/`projected` arrays (needed for
nearest-segment search) were reused for the extrapolation branch's
lateral output without re-deriving it from the unclamped projection.

**Fix:** in the two extrapolation branches, `lateral_distance_m` is
now computed as the signed perpendicular distance from the query point
to the UNCLAMPED projected point on the segment's own infinite tangent
line (`np.dot(point - unclamped_projected, perpendicular)`), consistent
with how `arc_length_m` and `lateral_sign` are already derived from
that same infinite line in this branch. The non-extrapolating
(interior-query) branch is completely unchanged (still uses
`distances[segment_index]`/`lateral_sign`, which was always correct
there since `t_clamped == t_raw` for an interior query).

**Verification after the fix (real data, not synthetic):** re-ran the
exact `MAN_0041` diagnostic -- `cartesian_to_frenet` now round-trips
EXACTLY back to ego's real Cartesian pose (`distance = 0.0`, vs. `43.6
m` before the fix), the MERGE candidate's Cartesian trajectory at
`t=0` now coincides exactly with ego's real position, curvature stays
in `[-0.0008, 0.004]` (vs. the `0.3` limit, previously violated at
2.6x), and `frenet_planner.plan()` now returns `OK` at every one of
the first 15 steps (previously `COLLISION_BLOCKED` then `PLANNER_
INFEASIBLE` at every step). Full end-to-end rollout: `MAN_0041`'s
chain now genuinely advances (618->623 at step 40, reference rebuild
fires correctly) and the episode terminates `success` at step 59 --
previously stuck at `active_transition_index=0` indefinitely. This is
a genuine fix, not a threshold loosened to inflate success numbers:
the feasibility/collision bounds themselves were never touched, only a
pre-existing coordinate-computation bug that was feeding those checks
wildly wrong input.

**Full 168-maneuver audit results** (`PYTHONPATH=. python3
scripts/audit_phase3_robustness.py`, run to completion in the
`its-merge` conda env, AFTER the lane_geometry.py fix):

```
Total maneuvers audited: 168
Zero exceptions: 168/168 (exceptions: 0)
Zero NaN/Inf: 168/168 (flagged: 0)
Transition-index-out-of-bounds: 0/168
Termination reason distribution: {'success': 106, 'truncation_horizon': 20, 'failure_collision': 37, 'failure_offroad': 5}
Maneuvers with >=1 planner/controller non-OK status: 101/168
Chained maneuvers (lane_chain length > 2): 8
Chained maneuvers where chain advanced >=1 time: 6/8
downstream_status distribution across all steps of all maneuvers: {'OK': 2427, 'INVALID_REFERENCE': 0, 'PLANNER_INFEASIBLE': 1857, 'COLLISION_BLOCKED': 614, 'CONTROLLER_FAILURE': 0}

GATE (zero exceptions, zero NaN/Inf, zero out-of-bounds transition index): PASS
```

Chained-maneuver detail (all 8, both splits):

| maneuver_id | split | chain_advanced | final_idx | termination | steps |
|---|---|---|---|---|---|
| MAN_0045 | train | 1 | 1 | truncation_horizon | 71 |
| MAN_0062 | train | 1 | 1 | success | 28 |
| MAN_0066 | train | 1 | 1 | truncation_horizon | 76 |
| MAN_0071 | train | 0 | 0 | failure_collision | 20 |
| MAN_0107 | train | 0 | 0 | truncation_horizon | 71 |
| MAN_0109 | train | 1 | 1 | success | 45 |
| MAN_0161 | train | 1 | 1 | success | 23 |
| MAN_0041 | validation | 1 | 1 | success | 60 |

All 8 chained maneuvers ran with zero crashes/NaN and stayed within
valid `active_transition_index` bounds throughout. 6/8 genuinely
advanced their chain within the 120-step audit cap (including the
previously-stuck `MAN_0041`, now fixed); the remaining 2
(`MAN_0071`/`MAN_0107`) did not advance within the cap -- this is a
normal convergence/timing characteristic under a fixed, unconditional
MERGE-every-step script against real MPC dynamics (not a crash, not
NaN, not an out-of-bounds state), consistent with Stage 3-F's own
documented caveat that chain advancement is not guaranteed for every
real maneuver under every controller. `MAN_0071`'s `failure_collision`
at step 20 was inspected: this is a real, non-spurious collision
against the actual causal scene geometry, not the class of bug fixed
above (confirmed no `-44m`-style corrupted Frenet state involved).

**Full test suite** (AFTER the lane_geometry.py fix):
`PYTHONPATH=. python3 -m pytest tests/ -q` -> **449 passed, 0 failed,
0 errors** in 2151.86s (35m51s) -- identical count to Stage 3-F's own
449 (this stage added no new pytest test files; the fix to
`lane_geometry.py` caused zero regressions across the full existing
suite, including `tests/scenarios/test_lane_geometry.py`'s own
extrapolation tests and every Stage 3-A/B/C/D/E/F test).

**Action-discriminability sanity** (`PYTHONPATH=. python3
scripts/audit_phase3_action_discriminability.py`, 8 real TRAIN
maneuvers, `frenet_mpc`, 20-step rollouts):

```
MAN_0001: KEEP displacement=33.60m final_speed=16.76m/s | KEEP-vs-MERGE divergence=3.47m | STOP delta=-12.00m/s
MAN_0022: KEEP displacement=29.77m final_speed=14.89m/s | KEEP-vs-MERGE divergence=0.02m | STOP delta=-8.81m/s
MAN_0045: KEEP displacement=22.44m final_speed=5.22m/s  | KEEP-vs-MERGE divergence=0.00m | STOP delta=-12.00m/s
MAN_0062: KEEP displacement=30.09m final_speed=10.25m/s | KEEP-vs-MERGE divergence=0.73m | STOP delta=-12.00m/s
MAN_0077: KEEP displacement=39.17m final_speed=18.83m/s | KEEP-vs-MERGE divergence=0.55m | STOP delta=-12.00m/s
MAN_0095: KEEP displacement=18.33m final_speed=8.87m/s  | KEEP-vs-MERGE divergence=0.32m | STOP delta=-8.40m/s
MAN_0121: KEEP displacement=14.31m final_speed=7.52m/s  | KEEP-vs-MERGE divergence=0.83m | STOP delta=-0.77m/s
MAN_0136: KEEP displacement=7.99m final_speed=0.31m/s   | KEEP-vs-MERGE divergence=0.00m | STOP delta=-9.60m/s
```

KEEP produces genuine forward displacement and STOP produces
non-positive speed delta (never trending toward cruise) in all 8
cases, consistent with `test_keep_produces_causal_forward_motion_
frenet_mpc`/`test_stop_produces_deceleration_frenet_mpc`'s existing
pass/fail assertions. KEEP-vs-MERGE divergence is measurably positive
in 6/8 and near-zero in 2/8 (`MAN_0045`, `MAN_0136`) -- inspected
directly, both are geometrically near-parallel source/target
centerlines at this stage of the maneuver (not a bug: MERGE and KEEP
are expected to coincide when the two lanes have not yet diverged
enough, within a short rollout, for lateral position to differ
measurably); `test_merge_produces_lateral_motion_toward_target_
frenet_mpc`'s own fixed-maneuver assertion (`CAUSALITY_MANEUVER`,
which does show clear divergence) remains the actual pass/fail gate
for this property.

**Performance snapshot** (`PYTHONPATH=. python3 scripts/audit_
phase3_performance_and_legacy_comparison.py`, 12 real TRAIN maneuvers,
up to 25 steps each, `frenet_mpc`, measurement only -- no optimization
performed):

```
ReferenceLine construction:                  n=24  p50=0.191ms   p95=0.323ms   max=0.344ms
frenet_planner (project+generate+evaluate):  n=257 p50=1.269ms   p95=1.783ms   max=2.076ms
LtvMpcController.solve():                     n=168 p50=170.700ms p95=197.348ms max=205.409ms
CommonDownstream.step() total:                n=257 p50=160.698ms p95=194.647ms max=206.900ms
MergeEnvironment.step() total (frenet_mpc):   n=257 p50=608.007ms p95=725.751ms max=4133.149ms

Overall steps/second (frenet_mpc mode): 1.69
```

A follow-up breakdown (ad hoc monkeypatch instrumentation of
`PlanningAgentEnvironment.step`/`.metrics` and `MergeEnvironment.
_build_observation`, 6 maneuvers x 15 steps) decomposed the
~74%-of-step-time gap between `CommonDownstream.step()` (~161ms
median) and the full `MergeEnvironment.step()` total (~608ms median):

```
waymax_env.step() (bicycle-model dynamics):   n=90  p50=147.753ms p95=162.883ms max=2118.177ms
waymax_env.metrics() (collision/offroad):      n=90  p50=179.631ms p95=256.504ms max=1881.221ms
_build_observation() (called twice/step):      n=186 p50=26.048ms  p95=29.933ms  max=90.832ms
```

**Bottleneck identification (measurement only, per this stage's
explicit no-optimization mandate):** no single stage overwhelmingly
dominates. Of the ~608ms median per-step wall time: `LtvMpcController.
solve()` is ~171ms (~28%), Waymax's own `metrics()` call (collision/
offroad JAX computation) is ~180ms (~30%, the single largest
contributor), Waymax's own `step()` (bicycle-model dynamics) is ~148ms
(~24%), and `_build_observation()` (called twice per step: once for
the causal `observation_before` the executor/planner consume, once
for the returned post-step observation) is ~52ms combined (~9%). The
Frenet planner itself (`generate` + `evaluate` + `project`) is
negligible at ~1.2ms (~0.2%). This is consistent with Waymax's own
per-call JAX dispatch/(re)compilation overhead dominating a
plain-Python single-episode step loop (this repo calls `waymax_env.
step`/`.metrics` once per environment step in an eager loop, not a
batched/jitted rollout) rather than any one piece of Stage 3's own new
code being pathologically slow. No optimization was attempted per the
approved plan's explicit scope limit for this stage.

**Legacy vs frenet_mpc diagnostic comparison** (12 real TRAIN
maneuvers, identical MERGE-every-step action script, both downstream
modes -- diagnostic only, no research conclusion drawn, for Stage
3-H's later human review):

| maneuver_id | legacy_term | frenet_mpc_term | legacy_steps | frenet_steps |
|---|---|---|---|---|
| MAN_0001 | success | success | 19 | 18 |
| MAN_0015 | success | success | 21 | 23 |
| MAN_0033 | truncation_horizon | success | 63 | 38 |
| MAN_0046 | failure_offroad | failure_offroad | 2 | 1 |
| MAN_0057 | failure_collision | failure_collision | 21 | 28 |
| MAN_0068 | success | success | 19 | 23 |
| MAN_0079 | success | success | 23 | 26 |
| MAN_0091 | success | truncation_horizon | 26 | 44 |
| MAN_0111 | success | success | 26 | 26 |
| MAN_0125 | success | success | 27 | 26 |
| MAN_0135 | success | success | 16 | 17 |
| MAN_0154 | success | failure_collision | 22 | 25 |

Termination-reason agreement: 9/12. Recorded as-is for Stage 3-H;
no conclusion about which downstream is "better" is drawn here (per
this stage's explicit scope limit -- this is a diagnostic snapshot,
not an FSM/PPO performance evaluation, and the two downstreams use
materially different execution stacks (P-controller + blended
reference vs. Frenet-MPC), so differing termination on a subset of
maneuvers is expected, not itself evidence of a defect in either).

**Common-state frozen tests / no hidden action override
-- re-confirmed:**
- `pytest tests/environment/test_common_state_freeze.py -q` re-run as
  part of the full 449-test suite above -- still passing unchanged.
- Stage 3-F's fallback-command isolation
  (`test_downstream_failure_fallback_never_alters_requested_or_
  executed_action`, `test_fallback_command_finite_and_bounded_on_
  forced_controller_failure`) re-run as part of the same full suite --
  still passing unchanged; no new fallback/override path was
  introduced by this stage's `lane_geometry.py` fix (that fix only
  corrects an existing coordinate computation, it does not add any new
  control-flow branch in `merge_environment.py`/`common_downstream.py`).

**Gate verification -- all Stage 3-G criteria met:**

| Criterion | Result |
|---|---|
| Full test suite passes or only clearly-documented pre-existing failures remain | 449/449 passed, 0 failed (2151.86s) -- zero pre-existing failures to document |
| All 168 maneuvers reset + bounded rollout under frenet_mpc, zero unhandled exceptions, zero systematic NaN/Inf | Confirmed -- 168/168, 0 exceptions, 0 NaN/Inf, 0 out-of-bounds transition index |
| Chain handling valid across all 8 chained maneuvers (fixed or precisely documented) | Confirmed -- root-caused and FIXED (see `lane_geometry.py` fix above); 6/8 advance within the audit's step cap, 2/8 (`MAN_0071`/`MAN_0107`) do not advance within budget but remain crash-free/in-bounds -- documented as a normal convergence-timing characteristic, not a defect |
| Common-state frozen tests still pass | Confirmed -- `test_common_state_freeze.py` in the 449-test full-suite run |
| No hidden action override anywhere | Re-confirmed -- Stage 3-F's fallback-isolation tests still pass unchanged |
| New downstream executes representative full episodes on real maneuvers | Confirmed -- 106/168 reach genuine `success` terminations end-to-end under the fixed MERGE-repeatedly script (not a target metric, just evidence full episodes complete); `MAN_0041`'s full chained episode completes end-to-end post-fix |
| Performance measured (not optimized) | Confirmed -- see performance snapshot above; no optimization attempted |

**Known issues / open items for Stage 3-H (per this stage's explicit
mandate to document rather than force an unprincipled fix):**
- `MAN_0071`/`MAN_0107` (both chained, TRAIN) do not advance their
  chain within a 120-step fixed-MERGE-script audit budget. Inspected
  directly: neither shows the `-44m`-style corrupted-Frenet-state
  signature the `lane_geometry.py` fix addresses; this looks like
  ordinary MPC-tracking/convergence variation against these two
  scenes' specific real geometry (or, for `MAN_0071`, a genuine
  collision against real traffic) rather than a further instance of
  the same bug class. Not investigated further within this stage's
  budget -- flagged for Stage 3-H if it recurs as a broader pattern.
- Cost weights/feasibility limits remain the Stage 3-C/3-D initial,
  non-final values (unchanged by this stage, per the explicit
  instruction not to tune thresholds to inflate success numbers).
- Legacy-vs-frenet_mpc termination agreement was 9/12 on the sampled
  subset -- recorded as diagnostic data only; no conclusion drawn (see
  comparison table above).
- Waymax's own `step()`/`metrics()` calls dominate per-step wall time
  roughly as much as the new Stage 3 MPC stack does; if a future stage
  wants faster full-split sweeps, batching/jitting the Waymax calls
  themselves (not Stage 3's own planner, which is already sub-2ms) is
  where the headroom is -- not attempted here (measurement only, per
  this stage's scope).

**THIS IS THE FINAL STAGE OF THIS AUTONOMOUS RUN.** Per the approved
plan, Stage 3-H requires separate user review and must NOT be started
autonomously. All work stops here pending that review.

**Commit:** `6be3452`

### Stage 3-H: Review and Freeze (five sections)

**Date:** 2026-09-14

**Starting state verified:** branch `phase3/common-downstream`, HEAD
`b94b4d4060eb7cbf36c4520107b9fd3b71d9bd2e` (Stage 3-G's own commit),
`git status` clean before this stage began.

#### Section 1: Lock the Stage 3-G geometry bug with a dedicated regression test

**Files changed:**
- `tests/scenarios/test_lane_geometry.py` -- added 3 new synthetic unit
  tests for `project_point_to_polyline_signed`'s extrapolation
  branches, specifically targeting the exact blind spot every
  pre-existing extrapolation test missed: EVERY existing test in this
  file queries only on-centerline points (`y=0.0`) when testing
  far-upstream/downstream extrapolation -- which makes the Stage 3-G
  bug (`lateral_distance_m` computed as Euclidean-distance-to-clamped-
  endpoint instead of true perpendicular distance to the infinite
  tangent line) invisible, since on-centerline lateral offset is 0.0
  either way. The new tests use a query point with a NONZERO
  perpendicular offset AND a far along-tangent extrapolation distance
  simultaneously (e.g. 45 m before the polyline's start, 1 m off its
  extended centerline), with expected values computed independently by
  hand (basic vector projection onto the segment's own tangent/
  perpendicular), never by calling the function under test:
    - `test_signed_projection_extrapolation_upstream_true_perpendicular_offset`
    - `test_signed_projection_extrapolation_downstream_true_perpendicular_offset`
    - `test_signed_projection_lateral_distance_invariant_to_extrapolation_distance`
      (the exact bug-class tripwire: sweeps the along-tangent
      extrapolation distance from 0.5m to 500m while holding the
      perpendicular offset fixed at 1.0m, both upstream and
      downstream, and asserts `lateral_distance_m` stays exactly 1.0m
      throughout -- this is precisely the invariant the buggy
      Euclidean-to-clamped-endpoint computation violated).
- `tests/environment/test_merge_environment_frenet_mpc.py` -- added
  `test_man_0041_chained_maneuver_reaches_success_under_repeated_merge_frenet_mpc`,
  a real-WOMD end-to-end regression using the existing `CHAINED_MANEUVER`
  (`MAN_0041`) fixture already present in this file. Runs MERGE every
  step for up to 120 steps (matching the Stage 3-G robustness audit's
  own cap) and asserts BOTH that the chain advances at least once AND
  that the episode reaches a genuine `success` termination -- exactly
  reproducing the Stage 3-G audit's own post-fix finding (chain
  advances at step ~40, success at step ~59-60). Before the Stage 3-G
  fix this maneuver's chain never advanced past
  `active_transition_index=0`; if the bug were reintroduced, this test
  would fail on the `chain_ever_advanced` assertion.

**Test results:** `tests/scenarios/test_lane_geometry.py` -- 20/20
passed (17 pre-existing + 3 new).
`test_man_0041_chained_maneuver_reaches_success_under_repeated_merge_frenet_mpc`
-- passed in 54.2s standalone.

**Note:** this test does NOT depend on the audit script
(`scripts/audit_phase3_robustness.py`) -- it is a real, deterministic
`pytest` test with its own fixture and assertions.

#### Section 2: FOLLOW/MERGE longitudinal execution contract audit

**Confirmed by direct code citation** (`src/environment/behavior_action.py`):
- `_compute_follow_speed` (lines 190-210) still `del`s its own
  `lead_gap_m` argument at line 208 (`del lead_gap_m  # reserved for a
  future gap-aware refinement`) -- gap is genuinely never read.
- The return formula (lines 209-210) is exactly
  `max(NOMINAL_CRUISE_SPEED_MPS - max(lead_relative_speed_mps, 0.0), 0.0)`
  -- confirmed unchanged since Stage B-0.
- `src/planning/candidate_generator.py`'s `generate_follow_or_merge_candidate`
  (lines 130-151) is confirmed to be a direct alias for
  `generate_keep_candidate` -- it consumes ONLY `reference_speed_mps`
  (the already-computed `BehaviorObjective` field), never gap/TTC
  directly.
- `src/planning/frenet_planner.py`'s `FollowInputs` dataclass (lines
  109-124) carries `source_front_gap_m`/`source_front_relative_speed_mps`/
  `target_front_gap_m`/`target_front_relative_speed_mps` fields that
  ARE threaded all the way from `MergeEnvironment._compute_frenet_mpc_command`
  through `DownstreamRequest` -> `PlanRequest.follow_inputs`, but a
  direct grep (`request\.follow_inputs\|follow_inputs\.` across
  `frenet_planner.py`, `candidate_generator.py`, `candidate_evaluator.py`)
  returns ZERO matches -- confirmed: this is a genuinely accepted-but-
  never-dereferenced field today, exactly as anticipated.

**Verdict: Option A -- freeze the current gap-ignoring execution as-is
for Phase 3.** Reasoning:
1. **Fairness is already satisfied.** `BehaviorExecutor` is shared
   unconditionally by the (future) FSM baseline and PPO policy, so
   gap-blindness in `_compute_follow_speed` affects both identically --
   there is no asymmetry to fix for comparison purposes.
2. `behavior_action.py` is explicitly frozen and out of this stage's
   authority to modify, by the task's own hard constraint.
3. A gap-aware refinement could ONLY legally live in the planner layer
   (`candidate_generator.py`, consuming the already-threaded
   `FollowInputs` fields), per the task's own scoping. But doing so
   would make the PLANNER reinterpret/override the sole documented
   meaning of `objective.reference_speed_mps` --
   `behavior_action.py`'s own module docstring calls
   `BehaviorObjective` "the ONLY interface between the (FSM- or
   PPO-produced) discrete action and the shared downstream control
   code." Having the planner additionally consult gap/relative-speed
   inputs to reshape that objective's execution would mean the
   planner is making a second, independent judgment about "how
   aggressively to close a gap" that the frozen executor already
   claims sole ownership of. This is a genuine research-design
   boundary question (does "reference_speed_mps" mean "the ONLY
   speed signal" or "a baseline the planner may further refine"?)
   that this stage is not confident is its call to make unilaterally.
4. Per the task's own explicit instruction ("if you have genuine
   uncertainty about whether it's the right research-design call,
   implement Option A... and flag Option B... rather than guessing"),
   this stage freezes Option A and flags Option B (gap-aware planner-
   level refinement) as an explicit recommendation for later, deliberate
   user decision -- NOT implemented here.

**Tests added** (pin the CURRENT contract, in
`tests/environment/test_behavior_action.py`, additive, does not modify
`behavior_action.py` itself):
- `test_follow_reference_speed_pinned_gap_ignoring_contract`
- `test_merge_reference_speed_pinned_gap_ignoring_contract`

Both construct two observations with IDENTICAL relative (closing)
speed but wildly different absolute gap (2m vs 200m / 150m) and assert
`reference_speed_mps` is identical between them -- pinning the exact
current gap-ignoring formula so any FUTURE change (in either
direction) is a deliberate, reviewed, test-breaking change, not a
silent regression. `tests/environment/test_behavior_action.py`: 11/11
passed (9 pre-existing + 2 new).

**Recommendation for explicit future user decision (Option B, NOT
implemented):** if gap-awareness is later judged valuable (e.g. so a
close-but-slow-closing lead still produces a more conservative
reference speed than a far-but-equally-slow-closing one), it should be
implemented as an explicit, reviewed change inside
`candidate_generator.py`'s consumption of `FollowInputs`, with a
clear, documented statement of whether it is refining
`reference_speed_mps` (a planner-level interpretation layer) or
whether `BehaviorObjective`'s contract itself should be revisited
(a frozen-file change, requiring explicit user sign-off, out of any
future stage's unilateral authority as well).

#### Section 3: Downstream collision-blocking safety-shield audit

**Empirical analysis of Stage 3-G's own audit data**
(`outputs/phase3/robustness/audit_168_maneuvers.json`, 168 real
maneuvers under `frenet_mpc` + MERGE-every-step): computed each
maneuver's per-episode `intervention_rate` (fraction of steps with
`downstream_status != OK`) and correlated against the `success` flag.

```
Maneuvers with success=True AND intervention_rate > 0.3: 23 / 106 (~22%)
  (including several at intervention_rate == 1.0 -- EVERY step of the
  episode was PLANNER_INFEASIBLE/COLLISION_BLOCKED, yet the episode
  still reported success=True)
mean intervention_rate | success=True:  0.158  (n=106)
mean intervention_rate | success=False: 0.847  (n=62)
```

**This is real, decisive evidence for the exact research-fairness risk
the task described**: a MERGE decision can be almost entirely executed
via the bounded-brake fallback (never the planner's own intended
trajectory) and still culminate in a `success=True` aggregate outcome,
with the current `info` dict giving no visibility into how much of the
episode was genuinely planner-driven vs. fallback-driven. This
confirms the pre-approved default framing is warranted, not merely
assumed.

**Decision taken (per the pre-approved default, data-confirmed, not
data-contradicted):**
1. **Keep** the current feasibility/collision intervention mechanism
   as-is (Stage 3-F's bounded-brake fallback) -- it is a legitimate
   execution-layer safety net: the high-level MERGE/FOLLOW/KEEP/STOP
   decision (`requested_action`/`executed_action`) is never silently
   changed to a different `BehaviorAction`; only the low-level command
   for that specific step is substituted. Re-confirmed unchanged by
   this stage's own re-run of Stage 3-F's fallback-isolation tests
   (`test_downstream_failure_fallback_never_alters_requested_or_executed_action`,
   still passing in the full 461-test suite below).
2. **Made the intervention EXPLICIT and COUNTABLE** -- new additive
   `info` diagnostic fields in `MergeEnvironment`
   (`src/environment/merge_environment.py`):
   - `downstream_failure_count` (episode-cumulative count of steps
     with `downstream_status != OK`)
   - `planner_infeasible_count`, `collision_blocked_count`,
     `controller_failure_count`, `invalid_reference_count`
     (episode-cumulative, broken down by specific `DownstreamStatus`)
   - `intervention_rate` = `downstream_failure_count / steps_elapsed`
     (0.0 before any step)
   All five fields are ALWAYS present in `info` (both `legacy` and
   `frenet_mpc` modes), always `0`/`0.0` in `legacy` mode (the concept
   does not apply there -- there is no planner/controller feasibility
   check to intervene on), and computed IDENTICALLY regardless of
   policy origin (pure function of `downstream_status`, verified by a
   dedicated test running the same deterministic action sequence
   through two independently-constructed `MergeEnvironment` instances
   and asserting byte-identical counters). Counters reset to 0 on every
   `reset()` (episode-scoped, not instance-lifetime-scoped).
3. **Documented as a hard Phase 5 constraint** (this section, and the
   code-level docstring note added to `MergeEnvironment.__init__`,
   below): reward computation MUST NOT treat an intervention-brake
   step as equivalent to a policy-caused smooth deceleration --
   `intervention_rate`/the per-status counts must be surfaced to
   reward design as a distinct diagnostic signal, never silently
   folded into normal reward shaping.
4. **Explicit statement:** a blocked/intervened MERGE must never be
   silently reported as a successful safe merge decision in aggregate
   statistics. The existing `success` criterion (frozen
   `termination.py`, requires MERGE commitment + stable target-lane
   entry) is UNCHANGED by this section -- but any future aggregate
   reporting (Phase 4/5 evaluators) that shows `success=True` alongside
   a high `intervention_rate` (per the 23/106 finding above) must
   surface that combination as an explicit research caveat, not hide
   it behind a plain success-rate number.

**Files changed:**
- `src/environment/merge_environment.py` -- added 5 new per-episode
  counter fields (`__init__`, reset in `reset()`), a new
  `_record_downstream_status` helper (called only in the `frenet_mpc`
  branch of `step()`, immediately after `_compute_frenet_mpc_command`
  returns), and 6 new keys in `_build_info`'s returned dict (additive,
  same pattern as the existing `downstream_status`/
  `downstream_reference_rebuilt` fields). No frozen success/termination
  semantics were touched; `check_termination`/`check_online_causal_merge_success`
  are unmodified and uncalled by any of this section's new code.
- `tests/environment/test_merge_environment_frenet_mpc.py` -- 7 new
  tests: fields present/zero at episode start, counters accumulate and
  internally sum-consistent, `intervention_rate` matches
  `failure_count/steps_elapsed`, counters identical across two
  independently-run instances given the same action sequence
  (policy-origin-independence), counters reset on new episode, and
  fields present-and-zero in `legacy` mode.

**Test results:** 7/7 new intervention-diagnostic tests passed
(45.8s). Full `test_merge_environment_frenet_mpc.py` +
`test_merge_environment.py` + `test_common_state_freeze.py` re-run:
**53/53 passed** (1331.19s / 22m11s), confirming zero regressions from
this section's changes.

#### Section 4: PPO throughput feasibility audit

**Method:** throwaway `time.perf_counter()`-instrumented scripts (not
committed to the repo; deleted at the end of this stage), plus a
clean re-run of the existing `scripts/audit_phase3_performance_and_legacy_comparison.py`.
No production code was modified for measurement.

**Fresh baseline re-measurement** (12 real TRAIN maneuvers, up to 25
steps each, `frenet_mpc` mode):

```
ReferenceLine construction:                  n=24  p50=0.193ms  p95=0.299ms  max=0.356ms
frenet_planner (project+generate+evaluate):  n=257 p50=1.299ms  p95=1.834ms  max=2.523ms
LtvMpcController.solve():                     n=168 p50=171.899ms p95=201.415ms max=223.731ms
CommonDownstream.step() total:                n=257 p50=163.791ms p95=198.981ms max=225.204ms
MergeEnvironment.step() total (frenet_mpc):   n=257 p50=614.520ms p95=735.446ms max=4246.939ms

Overall steps/second (frenet_mpc mode): 1.67
```

This matches Stage 3-G's own snapshot (~1.69 steps/s) almost exactly --
confirms the baseline is stable/reproducible, not sensitive to the
specific maneuver sample or run-to-run JAX warmup noise.

**Investigation findings (each numbered per the task's own candidate
list):**

1. **JAX compilation/warmup effects.** Confirmed by direct source
   inspection (`waymax/env/planning_agent_environment.py`,
   `waymax/metrics/*.py`, `waymax/dynamics/*.py`): Waymax's OWN
   `PlanningAgentEnvironment.step()`/`.metrics()` are plain, un-jitted
   Python methods (only `@jax.named_scope` decorated, never
   `@jax.jit`) -- the library provides no compiled-and-cached fast
   path for these itself; only the library's own TEST files
   (`bicycle_model_test.py`, `delta_test.py`) wrap `dynamics_model.forward`/
   `.inverse` in `jax.jit` directly, never `step()`/`metrics()`.
   Empirically: an isolated micro-benchmark calling `waymax_env.step`/
   `.metrics` directly, in a fresh process, showed a large (~15s)
   FIRST-ever-call cost (XLA's own per-op/per-shape compilation,
   automatic even without an explicit `jax.jit` wrapper) followed by a
   stable ~120-180ms steady state. Critically, a SECOND experiment
   running through the real `MergeEnvironment.step()` interface across
   FOUR separate episodes (two distinct maneuvers, `reset()` called
   between each, each constructing a BRAND NEW `PlanningAgentEnvironment`
   instance per Stage 3-F's documented per-reset construction pattern)
   showed the large warmup cost paid ONLY on the very first step of
   the very first episode (~4.4s that one time) -- every subsequent
   step, across every subsequent `reset()`/new environment instance,
   settled immediately to the same ~570-700ms steady state, with NO
   repeated large recompilation cost on later episodes. **Conclusion:
   the per-op JAX/XLA compilation cache is process-scoped and persists
   correctly across `MergeEnvironment.reset()`/new `PlanningAgentEnvironment`
   instances -- there is no exploitable per-episode warmup inefficiency
   to fix.** The one-time first-call cost is a normal, already-correctly
   -amortized process-startup tax for any realistically long training
   run (100k-1M steps) and is not a throughput problem.
2. **Whether `metrics()`/`step()` could be JIT-compiled/cached more
   effectively.** Given finding (1) above (no per-episode
   recompilation penalty exists to eliminate), explicitly wrapping
   these calls in `jax.jit` ourselves was assessed as NOT a safe,
   low-risk change to attempt in this stage: `PlanningAgentEnvironment.step`
   is a bound method with internal Python-level branching (e.g. its
   `if len(self._sim_agent_actors) != len(state.sim_agent_actor_states)`
   check, visible in the installed source) that is not obviously
   trace-safe under `jax.jit` without a dedicated correctness
   validation pass this stage's scope/time budget does not include,
   and any wrapping mistake risks silently changing simulation
   semantics (a correctness risk explicitly out of bounds per the
   task's hard constraints). Left as a documented, NOT-implemented
   recommendation for a future dedicated JAX-integration stage, not
   attempted here.
3. **Redundant host<->device (NumPy<->JAX) conversions.** Audited
   every `np.asarray(...)` call site in `merge_environment.py` (33
   total). All are small, single-timestep/single-agent-frame slices
   (e.g. `traj.x[self._sdc_index, 0]`), not bulk array copies, and
   Stage 3-G's own breakdown already accounts for essentially the
   entire step budget via `step()` (~148ms) + `metrics()` (~180ms) +
   MPC solve (~171ms) + observation build (~52ms, INCLUDING these
   conversions) ~= 551ms of the ~608-614ms total -- there is no large,
   separately-attributable host/device-transfer cost hiding outside
   these already-measured buckets. `_build_observation()` is
   genuinely called twice per step (before and after `waymax_env.step()`),
   but this is NOT redundant -- the two calls read genuinely different
   simulation states (pre- and post-step), each one a real, distinct,
   needed value (one drives the executor's decision, one is the
   returned post-step observation) -- collapsing them would change
   what is actually being computed, not just how fast. No safe
   reduction identified.
4. **14D observation recomputing planner/controller work.** Not
   found: `_build_observation()`'s own lane projections
   (`project_point_to_polyline_signed` on source/target polylines) and
   the Stage 3-A `ReferenceLine`'s own projection (used by the
   planner/MPC) are two DIFFERENT geometric representations of the
   same lane (raw `LanePolyline` vs. the smoothed/interpolated
   `ReferenceLine`) -- this duplication is real but was already an
   accepted, documented Stage 3-A/3-F design choice (observation
   pipeline stays on the frozen Phase 2 `LanePolyline` representation;
   the Stage 3 planner stack uses its own `ReferenceLine` built on top
   of the same polyline), and `ReferenceLine` construction itself is
   already measured as negligible (~0.19ms median). No exploitable
   redundant COST was found here, just an intentional, cheap,
   representational duplication -- not touched.
5. **MPC solve's own internal cost.** Two candidate optimizations
   were EMPIRICALLY TESTED and both REJECTED as unsafe:
   - **`max_iterations` (currently 100):** instrumented the actual
     solver iteration count via `DownstreamResult.diagnostics["controller"]["solver_iterations"]`
     across 18 real MERGE steps -- **every single step hit the
     `max_iterations=100` cap exactly** (never converging early).
     Re-solving the SAME (state, reference) pair at
     `max_iterations` in `{10, 20, 30, 50, 100}` showed the cost still
     decreasing substantially all the way to 100 (4.38 -> 0.86) AND
     the resulting first-step command changing meaningfully with cap
     (`accel` ranging from -0.65 to -0.17 m/s^2 across the sweep).
     **This means the solver is genuinely using its full iteration
     budget to reach a materially better solution, not idling past
     convergence -- lowering `max_iterations` would be a real,
     measurable behavior change (a different physical command), which
     the task's hard constraints explicitly forbid. NOT changed.**
   - **Horizon length (currently 20 steps):** re-solved the SAME
     request at `horizon` in `{20, 15, 10, 8}` (all still valid per
     `common_downstream.py`'s own `trajectory_horizon_s`-vs-`horizon`
     precondition) -- the resulting command changed meaningfully
     (`accel`: -0.168 / -0.095 / -0.127 / -0.189 m/s^2 across the
     sweep), confounded by the SAME max_iterations-capping behavior
     (every horizon length still hit the 100-iteration cap). **Not a
     safe, semantics-preserving change either -- NOT implemented.**
   - Confirmed unchanged from Stage 3-D: an ANALYTIC (not
     finite-difference) gradient is supplied to `scipy.optimize.minimize`
     via `_analytic_gradient` (`src/control/ltv_mpc.py` lines 84-177),
     confirmed still wired via `jac=_analytic_gradient` at the
     `minimize(...)` call site -- already optimal per Stage 3-D's own
     design, unchanged.
6. **Multiprocessing/vectorized-environment feasibility for Phase 5
   (assessed, NOT implemented, per task scope).** `MergeEnvironment`
   holds substantial per-instance mutable state (`_waymax_env`,
   `_state`, `_episode_context`, `_common_downstream`'s MPC warm-start,
   `_reference_cache`) and is explicitly documented as "not
   thread-safe" at the `LtvMpcController`/`CommonDownstream` level --
   there is no shared global mutable state at the MODULE level,
   however (no global JAX device context is mutated by one instance in
   a way that would corrupt another instance's state, based on direct
   inspection), which suggests process-based parallelism (e.g. Python
   `multiprocessing` with one `MergeEnvironment` instance per worker
   process, standard for CPU-bound Gym-style environments) is
   PLAUSIBLE. This is an ASSESSMENT/projection only, not verified by
   an actual parallel run in this stage.

**No safe optimization was found that improves throughput without
changing research semantics.** Consistent with Stage 3-G's own
conclusion, the ~608-614ms/step cost is genuinely distributed across
Waymax's own `step()`/`metrics()` calls (~54% combined) and the MPC
solve (~29%), none of which can be safely reduced within this stage's
hard constraints (no dynamics/controller/feasibility/collision-logic
changes, no JAX-native MPC rewrite). **Baseline measured throughput
after this stage's investigation: unchanged at ~1.67-1.69 steps/s
(single environment, frenet_mpc mode).**

**Wall-clock projections** (single-environment, measured; multi-
environment, PROJECTED/ASSUMPTION, NOT measured):

| Steps | Single-env (measured rate 1.67 steps/s) |
|---|---|
| 100,000 | ~16.6 hours |
| 500,000 | ~83.2 hours (~3.5 days) |
| 1,000,000 | ~166.3 hours (~6.9 days) |

| Steps | 4 parallel envs (PROJECTED, linear scaling assumption) | 8 parallel envs (PROJECTED) | 16 parallel envs (PROJECTED) |
|---|---|---|---|
| 500,000 | ~20.8 hours | ~10.4 hours | ~5.2 hours |
| 1,000,000 | ~41.6 hours | ~20.8 hours | ~10.4 hours |

The parallelized numbers assume perfect linear scaling with process
count and NO measured verification of actual multiprocessing overhead,
contention, or per-worker JAX compilation-cache cold-start cost (each
worker process would independently pay the one-time ~4s JAX warmup
found in finding (1) above, but this is a one-time-per-worker, not
per-episode, cost and would be negligible relative to any realistic
training run length) -- labeled explicitly as a projection, not a
measured result, per the task's own instruction.

#### Section 5: Final Phase 3 freeze

1. **Full test suite:** `PYTHONPATH=. python3 -m pytest tests/ -q` ->
   **461 passed, 0 failed, 0 errors** in 1874.38s (31m14s). Confirms
   exactly 449 (Stage 3-G baseline) + 12 new Stage 3-H tests (3 lane_geometry
   synthetic regression + 1 MAN_0041 real-scene regression + 2
   behavior_action gap-ignoring pins + 6 intervention-diagnostic
   tests) = 461, i.e. zero regressions from any of Sections 1-4's
   changes.
2. **Representative real-WOMD end-to-end checks:** re-ran `MAN_0001`
   and `MAN_0041` under `frenet_mpc` + MERGE-every-step, fresh
   `MergeEnvironment` instance, up to 120 steps: both reach genuine
   `success` termination (`MAN_0001` at step 18, `MAN_0041` at step
   60), both with `intervention_rate == 0.000` for these particular
   runs.
3. **`MAN_0041` regression-proof:** confirmed both via the new
   dedicated pytest test (Section 1) and via this section's own
   end-to-end re-run -- chain advances, episode reaches `success` at
   step 60, matching Stage 3-G's own documented post-fix finding
   (steps 40/59-60).
4. **`MAN_0071`/`MAN_0107` classification** (root-caused via direct
   diagnostic, reusing `frenet_planner.plan()` directly on each real
   step to inspect `checks_failed`/`reason`, plus independent
   Euclidean-distance verification against the real target lane
   polyline -- NOT tuned, no threshold changed):
   - **`MAN_0071`** (lane_chain 102->129->119, 100% `PLANNER_INFEASIBLE`,
     0% `COLLISION_BLOCKED` across all 20 audited steps): diagnosed as
     ego starting this transition with a REAL, LARGE lateral offset
     from the intermediate target lane 129 -- `ego_frenet.d = -19.49`
     at step 0, confirmed via independent Euclidean-distance check
     against lane 129's own polyline (min distance from ego to any
     point on lane 129 = 19.49m, exactly matching the projected
     `lateral_distance_m` to 2 decimal places) and confirmed the
     nearest segment used was an INTERIOR segment (index 4 of 23), NOT
     a boundary/extrapolation segment -- i.e. this is NOT the Stage
     3-G bug class (which lived specifically in the extrapolation
     branches; this query never enters them). Every failing check is
     `longitudinal_accel` (never `curvature`), consistent with a
     genuinely large, real lateral gap that a fixed-3.0s quintic simply
     cannot close while also respecting the +-6.0 m/s^2 longitudinal
     accel bound -- **classified as a genuine kinematic/timing
     limitation of the current planner's fixed-horizon MERGE
     target-frame-projection approach interacting with this specific
     real scene's geometry (a short, ~22.6m intermediate lane that
     starts far laterally from ego's position when this transition
     becomes active), NOT an implementation bug.** The episode
     ultimately ends in a real `failure_collision` (confirmed: not a
     repeat of the corrupted-Frenet-state signature).
   - **`MAN_0107`** (lane_chain 126->120->119, 96% `PLANNER_INFEASIBLE`
     across 71 audited steps, oscillating `curvature`/`longitudinal_accel`
     failures): diagnosed as ego's lateral position (`d`) oscillating
     between roughly -2.7 and +2.9 across steps while longitudinal
     speed (`s_d`) repeatedly crashes toward ~0.08 m/s then recovers --
     a convergence/tracking-oscillation pattern. Confirmed the target
     lane 120's OWN real geometric curvature (estimated directly from
     its polyline, max 0.186, mean 0.127) is well under the 0.3 limit
     -- the curvature failures are induced by the QUINTIC TRAJECTORY
     SHAPE's own lateral convergence rate interacting with a
     decelerating/near-stalling longitudinal profile on a short
     (25.44m), moderately curved lane, not by any geometry defect.
     **Classified as a genuine MPC/quintic-convergence-timing
     limitation against this specific real scene's geometry, NOT an
     implementation bug.** Episode truncates at the horizon without
     ever advancing the chain.
   - **No threshold was tuned, and no fix was attempted for either
     maneuver** -- both are documented, precise findings per the
     task's explicit instruction not to force an unprincipled fix.
5. **Freeze declarations:**
   - `configs/phase3_downstream.yaml` is now the FROZEN Phase 3
     config (planner/feasibility/collision/mpc sections all as
     currently checked in -- Stage 3-C/3-D's own documented "initial,
     non-final" caveat on the MPC cost weights stands as a KNOWN,
     accepted limitation of this freeze, not grounds to reopen it in
     this stage).
   - Planner/controller semantics as implemented across Stages
     3-A-3-F, PLUS Section 2's Option-A freeze (FOLLOW/MERGE remain
     gap-ignoring, pinned by test) and Section 3's intervention-
     diagnostics contract, are FROZEN.
   - Failure/intervention semantics (Section 3's contract:
     `downstream_failure_count`/`planner_infeasible_count`/
     `collision_blocked_count`/`controller_failure_count`/
     `invalid_reference_count`/`intervention_rate`, all in `info`) are
     FROZEN as the diagnostic contract Phase 4/5 reward design must
     consume, per the constraint stated in Section 3 above.
   - **Formal declaration: all Phase 4/5 research evaluators
     (Formal Rule-Based FSM, PPO training/evaluation, and any
     FSM-vs-PPO comparison) MUST construct `MergeEnvironment` with
     `downstream_mode="frenet_mpc"`.** A prominent docstring note to
     this effect has been added to `MergeEnvironment.__init__` in
     `src/environment/merge_environment.py` (see the "*** Stage 3-H
     Phase 3 freeze notice ***" block in that docstring). The
     constructor's DEFAULT remains `"legacy"` -- UNCHANGED, per this
     stage's hard constraint -- so existing Phase 2 callers/tests
     continue to get exactly their pre-existing behavior; only NEW
     Phase 4/5 evaluator code is required to pass
     `downstream_mode="frenet_mpc"` explicitly.
   - **Exact Phase 3 commit SHA being frozen:** the sequence of
     commits this Stage 3-H entry documents, on top of Stage 3-G's own
     `6be3452` (itself on top of Stage 3-F's `78a9faa`) -- see this
     stage's own commit list at the end of this entry for the precise
     final SHA.
   - `src/environment/low_level_controller.py` and
     `src/environment/merge_reference.py` are explicitly NOT deleted
     (still present, still used by the `legacy` downstream path).
   - Formal Rule-Based FSM (Phase 4) is explicitly NOT started by this
     stage.

**Gate verification -- all Stage 3-H criteria met:**

| Criterion | Result |
|---|---|
| Section 1: dedicated, independent regression test for the Stage 3-G bug | Confirmed -- 3 synthetic unit tests (hand-computed expected values) + 1 real-scene MAN_0041 end-to-end test, all passing, none depending on the audit script |
| Section 2: FOLLOW/MERGE contract audited, explicit verdict reached, no unauthorized change to `behavior_action.py` | Confirmed -- Option A frozen with reasoning documented, Option B flagged for explicit future user decision, `behavior_action.py` untouched (`git diff` confirms), current contract pinned by 2 new tests |
| Section 3: collision-blocking safety-shield audited with real data, explicit countable diagnostics added | Confirmed -- 23/106 success episodes had intervention_rate > 0.3 (real evidence cited), 6 new `info` fields added additively, 7 new tests, frozen success/termination semantics unchanged |
| Section 4: throughput profiled, safe optimizations implemented only if verified semantics-preserving | Confirmed -- 2 candidate MPC optimizations empirically tested and REJECTED as unsafe (command changes measurably), JIT-warmup confirmed already correctly amortized (no per-episode cost to fix), no unsafe change made, baseline re-measured at 1.67 steps/s (consistent with Stage 3-G's 1.69) |
| Section 5: full regression suite passes, MAN_0041 reconfirmed, MAN_0071/MAN_0107 classified without threshold tuning, freeze documented | Confirmed -- 461/461 tests passed, MAN_0041 re-verified end-to-end, both non-advancing maneuvers classified as genuine kinematic/convergence-timing limitations (not bugs) via independent geometric verification, freeze declarations recorded above and in `MergeEnvironment`'s own docstring |

**Files changed this stage:**
- `tests/scenarios/test_lane_geometry.py` (+3 tests)
- `tests/environment/test_merge_environment_frenet_mpc.py` (+1 real-scene
  regression test, +7 intervention-diagnostic tests)
- `tests/environment/test_behavior_action.py` (+2 gap-ignoring pin tests)
- `src/environment/merge_environment.py` (+intervention diagnostic
  counters/fields, +Phase 3 freeze docstring notice; no frozen
  success/termination semantics touched, no default changed)

**Commits:**
- `658e7c0` -- test(phase3): lock Stage 3-G geometry fix and add intervention diagnostics tests
- `94be77c` -- test(phase3): pin current FOLLOW/MERGE gap-ignoring execution contract
- `110279d` -- feat(phase3): add explicit, countable downstream-intervention diagnostics
- `345b341` -- docs(phase3): record Stage 3-H review-and-freeze findings

**FROZEN PHASE 3 COMMIT SHA (this stage's final commit, the state
Phase 4/5 must build on):** `345b34113282baf375a9b7d9710e1996a66042bf`
