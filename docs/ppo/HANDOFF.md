# PPO Session Handoff

Read this whenever resuming PPO work after a context loss. See
[PROGRESS.md](PROGRESS.md) for the detailed live-state tracker and
[PPO_PLAN.md](PPO_PLAN.md) for the durable scope/architecture reference.

## Resume protocol

When the user says **"계속 시작해"** (or equivalent), do this first,
without asking questions:

1. `git status`
2. `git branch`
3. `git log -n 10`
4. Read [docs/ppo/PROGRESS.md](PROGRESS.md)
5. Read this file (`docs/ppo/HANDOFF.md`)

Then resume exactly from the point recorded in "Next Owner Action" below.
Do **not** restart a Phase that PROGRESS.md already marks complete.

Unified branch-creation order (see also
[PPO_PLAN.md § 9](PPO_PLAN.md#9-git-conventions)):

1. Check `git status` / `git branch` / `git log`
2. Create `feat/ppo-phase0-5` (if absent)
3. Begin P0 Baseline Audit
4. All further PPO changes/doc commits happen on that branch, never on
   `main`

---

## What's completed so far

**P0 — Baseline Audit is COMPLETE.** Working on branch
`feat/ppo-phase0-5` (already created and checked out at session start,
carrying the approved plan docs from commit `aa8cf5b`).

P0 confirmed, by direct inspection (not assumption), that every frozen
Phase 1–3 invariant PPO_PLAN.md §2 lists is intact:

- `OBSERVATION_DIM == 14`, exact field order
  (`src/environment/observation_builder.py:68-85`)
- `BehaviorAction`: `KEEP=0, FOLLOW=1, MERGE=2, STOP=3`
  (`src/environment/behavior_action.py:59-70`)
- `TerminationReason` has exactly the 5 values PPO_PLAN.md expects
  (`src/environment/termination.py:48-53`)
- MERGE commitment (`decision_state.py`): the step where the policy
  selects MERGE is the step where `DecisionState.advance` still
  returns MERGE as the effective action for THAT step, before flipping
  `phase` to `MERGE_COMMITTED` — confirming §7.1's pre-step
  `info_before["merge_committed"]` pattern is directly supported by
  the existing `reset()`/`step()` API with no code change. This exact
  pattern (`was_committed_before = info.get("merge_committed", False)`
  read before calling the policy/`env.step`) is already used in
  `src/environment/full_split_evaluator.py::run_episode` — good
  precedent to mirror in the P4 rollout loop.
- `downstream_mode="frenet_mpc"` exists and is exercised by
  `tests/environment/test_merge_environment_frenet_mpc.py`
- Intervention diagnostics (`intervention_rate`,
  `downstream_failure_count`, `planner_infeasible_count`,
  `collision_blocked_count`, `controller_failure_count`,
  `invalid_reference_count`) are all present in `_build_info`

Full existing regression suite: **461 passed**, 0 failed, in 1798.33s
(0:29:58) (`pytest tests/ -q`, inside the `its-merge` conda env — note:
system `python3` has no jax/numpy at all; the project's env must
always be activated first via
`source /home/autonav/miniconda3/etc/profile.d/conda.sh && conda activate its-merge`).
This was run in the background and the actual process exit was waited
for (`[exited with code 0]`), not inferred from a partial read — an
intermediate `tail`-based snapshot taken mid-run had briefly shown `F`
marks around the 31% mark (a module still in progress at that moment);
the completed run's own final summary is 100% dots with `461 passed`
and zero failures, so that mid-run snapshot was stale/truncated, not a
real failure. There are no pre-existing or newly-introduced regression
failures to report — and since P0 made zero source changes, any
failure would necessarily have been pre-existing/environmental rather
than something this session introduced; none were found.

A reproducibility sanity check (fixed KEEP-action script on maneuver
`MAN_0001`, `frenet_mpc` mode, 40 steps) produced byte-identical
observation traces across 2 independent runs
(`max_abs_obs_diff_between_two_runs = 0.0000000000`); the episode did
not terminate within 40 KEEP-only steps
(`termination_reason='none'`, `steps_elapsed=40`), with
`downstream_status='COLLISION_BLOCKED'`, `collision_blocked_count=2`,
`intervention_rate=0.05` — confirms the intervention-diagnostics
fields are live and populated as expected (a diagnostic signal firing
correctly under a deliberately non-merging fixed script, not a test
failure). A throughput measurement on a separate timed rollout (5-step
warmup + 60 timed steps): ~1.64 steps/sec (60 steps in 36.69s). That
specific script invocation logged `Cannot dlopen some GPU libraries
... Skipping registering GPU devices` and ran on CPU fallback — an
env/library-path issue local to that one subprocess, not a code
regression (the interactive `jax.devices()` check and the full
GPU-backed pytest run above both confirm the GPU path itself works).
Per-step cost is dominated by the LTV-MPC's `scipy.optimize.minimize`
solve regardless of backend
(`configs/phase3_downstream.yaml`'s `mpc.max_iterations: 100`), which
is expected/by-design, not a bug. **This is why P5 Smoke Training must
use very small step/maneuver budgets** — already required by the
plan, now confirmed necessary for wall-clock reasons too. A future
phase should re-run this specific throughput check with GPU dlopen
confirmed working if a tighter number is needed.

Dependency versions recorded (Python 3.10.21, JAX/jaxlib 0.6.2, NumPy
2.2.6, TensorFlow 2.21.0, waymo-waymax 0.1.0, GPU: RTX 4060 via
`jax.devices() == [CudaDevice(id=0)]`). `flax==0.10.7`,
`optax==0.2.8`, and `orbax-checkpoint==0.11.39` were **already present**
in the conda env, compatible with the installed JAX — no install
needed. Only `wandb` was missing; installed via plain
`pip install wandb` (resulted in `wandb==0.30.0`, no `--upgrade` flag
used, no jax/jaxlib/numpy version touched). Re-verified
`jax.devices()` and did NOT need to re-run the full suite again after
this install since wandb has no jax dependency overlap — but
`jax.devices()` was explicitly re-checked post-install and still
reports the GPU.

Full details: [PROGRESS.md](PROGRESS.md).

## Last successful test

`pytest tests/ -q` → **461 passed**, 0 failed, 1798.33s (0:29:58),
exit code 0 — waited for real process completion.

## Failed tests

None.

## Problems found

None. No contradiction found between PPO_PLAN.md and the current
codebase in P0.

## Recent commits (PPO-related)

- `aa8cf5b` — `docs(ppo): add approved PPO P0-P5 plan and context-resume docs`
  (pre-existing at session start)
- P0 completion commit: see `git log` on `feat/ppo-phase0-5` for the
  exact SHA (`chore(ppo): audit frozen training baseline`) — committed
  immediately after this HANDOFF.md update.

## Running processes

None.

## Latest checkpoint

None yet (P1 skeleton / P5 full save/load contract).

## Next command to run

Begin P1 per [PROGRESS.md § Next Exact Action](PROGRESS.md#next-exact-action):
create the `src/rewards/`, `src/policies/ppo/`, `src/training/`,
`src/tracking/`, `configs/reward/`, `configs/ppo/` directories and the
`scripts/train_ppo.py`/`scripts/smoke_train_ppo.py` skeletons, then a
config loader + seed handling + checkpoint skeleton.

## Next file to modify

New files only (P1 creates new PPO scaffolding — no existing Phase 1–3
file under `src/environment/`, `src/planning/`, `src/control/`,
`src/scenarios/` is modified or relocated).

---

## NEXT OWNER ACTION

**Begin P1 PPO Foundation.** Per
[PPO_PLAN.md §0.1/P1](PPO_PLAN.md#p1--ppo-foundation) and
[PPO_PLAN.md §4](PPO_PLAN.md#4-file-structure-convention) (file
structure convention). Do not skip ahead to P2–P5. Do not perform any
reward/hyperparameter tuning, W&B sweeps, full training, or FSM-vs-PPO
comparisons — those remain out of scope through P5 and are reserved for
the user to do manually afterward. Remember: no PPO-FIT/PPO-TUNE split,
ever, through P5.

**Fixed constraint for the rest of this entire P0–P5 effort (applies
to every remaining Phase): exactly ONE commit per Phase.** No
intermediate "-A"/"-B", sub-stage, or WIP commits within a Phase. Do
all of a Phase's implementation, its required tests, and its doc
updates first, and only commit once, at the very end of that Phase,
with everything for that Phase staged together in a single commit.
P0 already followed this rule (one commit,
`chore(ppo): audit frozen training baseline`). P1 through P5 must each
get exactly one commit the same way — six new commits total across
the whole effort, on top of the pre-existing `aa8cf5b` docs commit.
