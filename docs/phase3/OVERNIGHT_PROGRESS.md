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
| 3-F MergeEnvironment Integration | COMPLETE | (pending, see below) |
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

**Commit:** (recorded in a follow-up commit after this doc is committed, per prior stages' pattern)

**Next stage: 3-G** (robustness/regression).
