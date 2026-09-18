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

**P0 — Baseline Audit, P1 — PPO Foundation, and P2 — Reward V0 + W&B
Foundation are all COMPLETE.**
Working on branch
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

**P1 — PPO Foundation** then added, as NEW files only (no existing
Phase 1-3 file touched): package directories with `__init__.py`
(`src/rewards/`, `src/policies/` + `src/policies/ppo/`,
`src/training/`, `src/tracking/`, `tests/rewards/`, `tests/policies/`,
`tests/training/`); `configs/reward/merge_reward_v0.yaml` and
`configs/ppo/ppo_base.yaml`/`ppo_smoke.yaml` (values only, matching
PPO_PLAN.md §5/§6 verbatim — no reward/PPO CODE yet, that's P2/P3);
`src/training/config.py` (`load_ppo_config`/`load_reward_config`,
matching this repo's existing `yaml.safe_load` + frozen-dataclass
convention); `src/training/seeding.py` (`make_seed_state`/`split_key`);
`src/training/checkpoint.py` (`CheckpointPayload` carrying every field
PPO_PLAN.md §10 requires, plus `get_git_sha`/`save_checkpoint`/
`load_checkpoint` as a P1 pickle-based skeleton — P5 verifies the real
JAX-pytree save→load→resume→update cycle); `scripts/train_ppo.py`/
`scripts/smoke_train_ppo.py` (config+seed resolution only, no rollout
loop yet — smoke script also deterministically selects a small
canonical-TRAIN maneuver_id subset via
`--max-maneuvers`/`--maneuver-ids`/`--seed`, explicitly NOT a
PPO-FIT/TUNE split); `tests/training/test_config.py` (10 tests).

Both scripts were run directly and printed the expected resolved
config/seed/maneuver-subset with exit code 0. New P1-specific tests
(`tests/policies/test_imports.py` [7], `tests/rewards/test_imports.py`
[4], `tests/training/test_imports.py` [7],
`tests/training/test_config.py` [10] — 28 tests total) all pass when
run alone (28 passed in 1.08s). Re-ran the FULL existing regression
suite (Phase 1-3 tests + new P1 tests together), waited for real
process completion (not a partial/estimated read): **489 passed**, 0
failed, 0 errors, in 1867.61s (0:31:07), exit code 0 — confirms P1
broke nothing (461 pre-existing Phase 1-3 tests + 28 new P1 tests =
489). Note: a first attempt at this full-suite rerun showed 62
failures, but investigation confirmed that run was contaminated by a
second, stale `pytest tests/ -q` process left running concurrently
from an earlier session (both processes competing for the same GPU,
causing `FailedPreconditionError: Failed to allocate scratch buffer
for device 0` / TF dataset-loading `TypeSpec` errors) — not a real
regression. No source file was touched during that investigation. The
stale process was allowed to finish and exit on its own, then the full
suite was re-run solo; that clean, uncontended run is the **489
passed, 0 failed** result above and is the one being reported.

**P2 — Reward V0 + W&B Foundation** then implemented the real logic
behind the P1 structural skeletons in `src/rewards/merge_reward.py`,
`src/rewards/reward_wrapper.py`, and `src/tracking/wandb_logger.py`
(no existing Phase 1-3 file touched; `configs/reward/merge_reward_v0.yaml`
was already correct from P1 and needed no change).
`merge_reward.compute_reward(reward_config, termination_reason,
is_policy_step, info=None)` is a pure fixed-table lookup: terminal
component keyed by the frozen `MergeEnvironment`'s own
`info["termination_reason"]` string (`None` treated as `"none"`),
decision-cost component keyed purely by the caller-supplied pre-step
`is_policy_step` flag — no import of `src.environment` or `waymax`
anywhere in the module (verified by a code-level test), satisfying
§5.1's source-of-truth rule. `MergeRewardWrapper` wraps this into
rollout-loop-friendly per-episode `reward/terminal`/
`reward/decision_cost`/`reward/total` accumulation
(`episode_sums()`/`reset()`), cross-checking its own component sum
against `compute_reward`'s total. `WandbLogger` now does real
`wandb.init`/`wandb.config.update`/`wandb.log` calls, supporting both
online and `WANDB_MODE=offline`; `log_config` validates all of §8's
`REQUIRED_CONFIG_KEYS` are present (raises `ValueError` otherwise);
`log_metrics` accepts any metric name by design so P3-P5 can add more
without touching this module.

Added 35 new tests over the P1 baseline:
`tests/rewards/test_merge_reward.py` (new file, 32 collected items —
mostly from parametrization: the 5-way terminal-outcome table incl.
`None`->`"none"`, real-decision-step=-0.01/auto-step=0.0 decision
cost, 8 total-reward sum combinations, a finite/no-NaN sweep across
every reason x `is_policy_step` combo, an unrecognized-reason
`ValueError` guard, a code-level no-`src.environment`/no-`waymax`
-import check, wrapper-vs-`compute_reward` agreement, and wrapper
episode-sum accumulation/reset), `tests/tracking/test_wandb_logger.py`
(new package, 4 tests — module constants, an offline smoke run that
logs a config dict + a metric point and asserts real files exist on
disk, `log_config` rejecting a config missing a required key,
`log_metrics` accepting an arbitrary non-`MINIMUM_METRICS` name).
`tests/rewards/test_imports.py` (updated in place, still 4 collected)
and `tests/training/test_imports.py` (updated in place, 6 collected,
was 7 — the redundant `WandbLogger.__init__`-raises-`NotImplementedError`
stub check was removed since it is now real, folded into
`test_wandb_logger_module_imports`'s constant checks) had their
now-obsolete P1 stub-`NotImplementedError` assertions replaced with
real-behavior assertions; `rollout.py`/`gae.py`/`trainer.py` (P4/P5
scope) remain untouched `NotImplementedError` stubs and their tests
still assert that.

Ran the new P2 tests together with all pre-existing P1 tests
(`tests/rewards/ tests/tracking/ tests/training/ tests/policies/`):
**63 passed** in 1.88s. Ran a standalone offline W&B smoke run
directly (outside pytest, `WANDB_MODE=offline`): logged the full §8
config-key set plus the three reward metrics, exit code 0, real files
written under a local `wandb/offline-run-<timestamp>-<id>/` directory
(`run-<id>.wandb`, `logs/debug.log`, `logs/debug-internal.log`,
`files/requirements.txt`, etc.). Confirmed no other `pytest` process
was running (`ps aux | grep pytest`) before launching the full
regression suite (learning P1's lesson — this run needed no
mid-course correction). Ran the FULL existing regression suite (Phase
1-3 + P1 + P2 tests together) and waited for the real process exit:
**524 passed**, 0 failed, 0 skipped, 1769.33s (0:29:29), independently
confirmed via `pytest tests/ --collect-only -q` -> "524 tests
collected" (489 pre-existing + 35 new P2 = 524, matching exactly).

## Last successful test

`pytest tests/ -q` (after P2, clean solo run, first attempt — no
concurrent-pytest contention this time) -> **524 passed**, 0 failed, 0
skipped, 1769.33s (0:29:29) — waited for real process completion,
verified directly against the log file's final summary line
(`524 passed in 1769.33s (0:29:29)`, 100% dots, no `F`/`E` marks).

## Failed tests

None.

## Problems found

None. No contradiction found between PPO_PLAN.md and the current
codebase in P0, P1, or P2.

## Recent commits (PPO-related)

- `aa8cf5b` — `docs(ppo): add approved PPO P0-P5 plan and context-resume docs`
  (pre-existing at session start)
- `bcf2b4c` — `chore(ppo): audit frozen training baseline` (P0 completion)
- `2f865b6` — `feat(ppo): add PPO foundation scaffolding (P1)` (P1
  completion)
- P2 completion commit: see `git log` on `feat/ppo-phase0-5` for the
  exact SHA (`feat(ppo): implement Reward V0 and W&B logging
  foundation (P2)`) — committed immediately after this HANDOFF.md
  update.

## Running processes

None.

## Latest checkpoint

None yet (P1 only defines the `CheckpointPayload` contract + a pickle
-based save/load skeleton with no real PPO train state to save; P5
exercises the full save→load→resume→update cycle against real
policy/value params and optimizer state). P2 did not touch checkpoint
code.

## Next command to run

Begin P3 per [PROGRESS.md § Next Exact Action](PROGRESS.md#next-exact-action):
implement `src/policies/ppo/{networks,distribution,policy,loss,state}.py`
(discrete PPO core, algorithm-only, no real environment yet), add
`tests/policies/` behavioral tests per §0.1/P3, re-run the full
regression suite, then commit.

## Next file to modify

`src/policies/ppo/networks.py` (P3's first task — policy/value
network definitions).

---

## NEXT OWNER ACTION

**P0, P1, and P2 are all complete.** Begin **Phase P3 — Discrete PPO
Core** per
[PPO_PLAN.md §0.1/P3](PPO_PLAN.md#p3--discrete-ppo-core). Do not skip
ahead to P4/P5. P3 validates the PPO algorithm in isolation only — no
real `MergeEnvironment` rollout yet (that's P4). Preserve the fixed
§11 action-index -> `BehaviorAction` mapping
(`0->KEEP, 1->FOLLOW, 2->MERGE, 3->STOP`) exactly as already
established in P1's constant + regression test.

**Fixed constraint for the rest of this entire P0–P5 effort (applies
to every remaining Phase): exactly ONE commit per Phase.** No
intermediate "-A"/"-B", sub-stage, or WIP commits within a Phase. Do
all of a Phase's implementation, its required tests, and its doc
updates first, and only commit once, at the very end of that Phase,
with everything for that Phase staged together in a single commit. P0,
P1, and P2 already followed this rule (`chore(ppo): audit frozen
training baseline`, `feat(ppo): add PPO foundation scaffolding (P1)`,
`feat(ppo): implement Reward V0 and W&B logging foundation (P2)`). P3
through P5 must each get exactly one commit the same way — three more
commits total across the rest of the effort, on top of the
pre-existing `aa8cf5b` docs commit and the three already-landed
P0/P1/P2 commits.
