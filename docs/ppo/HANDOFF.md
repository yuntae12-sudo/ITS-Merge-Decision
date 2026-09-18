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

**P0 — Baseline Audit, P1 — PPO Foundation, P2 — Reward V0 + W&B
Foundation, P3 — Discrete PPO Core, and P4 — Rollout + GAE Integration
are all COMPLETE.**
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

**P3 — Discrete PPO Core** then implemented the real algorithm behind
the P1 structural skeletons in `src/policies/ppo/{networks,distribution,
loss,state,policy}.py` (no existing Phase 1-3 file touched, no
`src/environment/` file touched). `networks.py` now defines real
`flax.linen.Module`s: `PolicyNetwork` (14 -> 256 -> 64 -> 32 -> 4
logits, tanh) and `ValueNetwork` (14 -> 256 -> 64 -> 32 -> 1, tanh,
squeezed to scalar/`(batch,)`). `distribution.py` implements
`sample_action` (`jax.random.categorical`), `deterministic_action`
(`jnp.argmax`), `log_prob` (`log_softmax` + `take_along_axis`),
`entropy` (`-sum(p*log p)`), and `probs`; `ACTION_INDEX_TO_BEHAVIOR`
(SS11's fixed mapping, still importing the real `BehaviorAction` enum)
is unchanged from P1. `loss.py` implements `ppo_ratio`,
`ppo_clipped_surrogate_loss` (with `clip_fraction`/`approx_kl`
diagnostics), `value_loss` (MSE), `entropy_bonus`, and `ppo_total_loss`
-- math ported from the V-Max reference pattern
(`ppo_factory.py::_make_loss_fn`) adapted to the discrete 4-way
categorical head, with no continuous Gaussian/Beta action head,
observation extractor, reward design, or vectorized-env structure
ported (PPO_PLAN.md SS3). `state.py::create_train_state` builds two
independent `flax.training.train_state`-based `PPOTrainState`s
(policy, value -- no shared parameters, SS7.2), each with its own
`optax.chain(clip_by_global_norm, adam)` optimizer. `policy.py`'s
`PPOPolicy` wraps a network + params with `act`/`act_deterministic`,
consuming only the frozen `BehaviorAction` enum from
`src.environment.behavior_action` -- confirmed
(`grep -rn "policies\|ppo" src/environment/`) the PPO -> Environment
dependency direction stays strictly one-directional.

Added 34 new behavioral tests in `tests/policies/test_ppo_core.py`
covering every item in PPO_PLAN.md SS0.1/P3's required-tests list (14D
input handling, logits shape, finite logits, softmax sums to 1, action
range 0..3, deterministic == argmax, stochastic variation across 200
keys, finite log_prob/entropy, 6 hand-checked PPO-ratio pairs, 3
clipping-behavior tests, scalar value loss, finite full PPO
loss/gradients on a synthetic minibatch, non-all-zero gradients,
optimizer step changing parameters for both policy and value states,
same-seed reproducibility, different-seeds-can-differ, and the SS11
action-mapping regression against the real `BehaviorAction` enum).
Updated `tests/policies/test_imports.py` in place (stub-`NotImplementedError`
checks replaced with real-attribute assertions; still 6 collected
items). Ran the new P3 tests together with the updated import tests
(`tests/policies/`): **40 passed** in 17.55s. Re-confirmed
`jax.devices() == [CudaDevice(id=0)]` before and after. Confirmed no
other `pytest` process was running before the full-suite run. Ran the
FULL existing regression suite (Phase 1-3 + P1 + P2 + P3 tests
together) and waited for the real process exit: **557 passed**, 0
failed, 0 skipped, 1802.23s (0:30:02), exit code 0, independently
confirmed via `pytest tests/ --collect-only -q` -> "557 tests
collected" (524 pre-existing + 33 net new P3 = 557, matching exactly).

**P4 — Rollout + GAE Integration** then connected the real P3 PPO core
to the real `MergeEnvironment` (`downstream_mode="frenet_mpc"`, SS2).
`src/training/rollout.py::collect_episode_rollout` mirrors the exact
pre-step `info_before["merge_committed"]` pattern P0 confirmed is
already used by `full_split_evaluator.py::run_episode`:
`is_policy_step = not info_before.get("merge_committed", False)` is
computed from the PREVIOUS `reset()`/`step()` call's info, never the
current step's own post-step info. On an auto-execution step, the PPO
policy network is not called for action selection at all --
`BehaviorAction.MERGE` is submitted directly. Every physical frame
(regardless of `policy_mask`) becomes one `Transition` covering the
fixed field set; reward comes from P2's `MergeRewardWrapper.compute`,
fed the same pre-step `is_policy_step` flag plus the environment's own
post-step `info["termination_reason"]` (SS5.1, never re-derived).
`src/training/gae.py::compute_gae` implements standard backward-
recursion GAE over the FULL physical trajectory
(`gamma=0.99`/`gae_lambda=0.95`), correctly distinguishing true
termination (no bootstrap) from truncation (does bootstrap from
`next_value`); `normalize_advantages_masked`/`masked_mean_std` compute
the Actor-side normalization mean/std using ONLY `policy_mask == 1`
positions via boolean-mask indexing before the reduction (SS7.2) --
structurally excluded, not merely down-weighted.
`src/training/trainer.py::build_training_batch` wires rollout -> GAE
-> masked normalization into one flat PPO-ready batch dict end to end;
`run_training` (the actual multi-epoch update loop) correctly remains
a `NotImplementedError` stub, per P4's explicit non-goal (P5 scope).

Added 38 new P4 tests: `tests/training/test_gae.py` (11),
`tests/training/test_rollout.py` (8), `tests/training/test_trainer.py`
(3), plus `tests/training/test_imports.py` updated in place (still 6
collected). **The single most important test in this Phase**,
`test_prestep_merge_policy_mask_regression`, runs a scripted-MERGE
policy against the real `MergeEnvironment` on a single-transition
maneuver (`MAN_CAUSALITY`) and confirms the VERY FIRST transition (the
decision frame where MERGE is selected) has `policy_mask == 1`, while
every transition strictly after it (auto-execution) has
`policy_mask == 0` -- directly guarding against SS7.1's reversed
post-step bug. **PASSED.** The SS7.2 test D
(`test_terminal_reward_propagates_to_merge_decision_frame`) confirms a
real terminal SUCCESS/COLLISION reward's GAE credit propagates back
through the intervening auto-execution frames to the MERGE decision
frame's return. **PASSED.** `git diff --stat` against
`src/environment/`, `src/planning/`, `src/control/`, `src/scenarios/`
confirmed empty -- zero frozen files touched.

Ran the new tests file-by-file, then `tests/training/ tests/policies/
tests/rewards/` together (114 passed, 148.94s) as an intermediate
check. Confirmed no other `pytest` process was running before the full
suite. Ran the FULL existing regression suite and waited for the real
process exit: **579 passed**, 0 failed, in 1920.04s (0:32:00), exit
code 0, independently confirmed via `pytest tests/ --collect-only -q`
-> "579 tests collected" (557 pre-existing + 22 net new P4 = 579,
matching exactly: 11 gae + 8 rollout + 3 trainer = 22).

## Last successful test

`pytest tests/ -q` (after P4, clean solo run — no concurrent-pytest
contention) -> **579 passed**, 0 failed, 1920.04s (0:32:00) — waited
for real process completion, verified directly against the log file's
final summary line (`579 passed in 1920.04s (0:32:00)`, 100% dots, no
`F`/`E` marks).

## Failed tests

None.

## Problems found

None. No contradiction found between PPO_PLAN.md and the current
codebase in P0, P1, P2, P3, or P4.

## Recent commits (PPO-related)

- `aa8cf5b` — `docs(ppo): add approved PPO P0-P5 plan and context-resume docs`
  (pre-existing at session start)
- `bcf2b4c` — `chore(ppo): audit frozen training baseline` (P0 completion)
- `2f865b6` — `feat(ppo): add PPO foundation scaffolding (P1)` (P1
  completion)
- `565a7fe` — `feat(ppo): implement Reward V0 and W&B logging
  foundation (P2)` (P2 completion)
- `334e4c6` — `feat(ppo): implement discrete PPO core (P3)` (P3
  completion)
- P4 completion commit: see `git log` on `feat/ppo-phase0-5` for the
  exact SHA (`feat(ppo): implement rollout and GAE integration (P4)`)
  — committed immediately after this HANDOFF.md update.

## Running processes

None.

## Latest checkpoint

None yet (P1 only defines the `CheckpointPayload` contract + a pickle
-based save/load skeleton with no real PPO train state to save; P5
exercises the full save→load→resume→update cycle against real
policy/value params and optimizer state). P3 built the real
`PPOTrainingState`/`PPOTrainState` structures that will populate that
contract's `policy_params`/`value_params`/`optimizer_state` fields;
P4 built the real rollout/GAE/batch-building pipeline that will feed
the P5 update loop, but did not itself touch checkpoint code.

## Next command to run

Begin P5 per [PROGRESS.md § Next Exact Action](PROGRESS.md#next-exact-action):
wire the actual multi-epoch parameter-update loop into
`src/training/trainer.py::run_training` (currently a
`NotImplementedError` stub), consuming P4's `build_training_batch`
output. Verify env reset, action sampling/mapping, rollout, reward,
GAE, PPO loss (filtering to `policy_mask == 1` for every Actor-side
quantity per §7.2 -- `build_training_batch` does not do this
filtering itself), parameter update/backprop, finite loss/gradients,
checkpoint save/load/resume (full §10 contract), and W&B logging, all
on a small deterministic TRAIN-subset "smoke" run (Stage 1: 1-3
maneuvers, minimal updates; Stage 2 only if Stage 1 succeeds). No
long/full-TRAIN run, no TUNE split, no sweeps, no VAL evaluation, no
FSM-vs-PPO comparison. Re-run the full regression suite, update
PROGRESS.md/HANDOFF.md, append an EXPERIMENT_LOG.md entry, then make
exactly one P5 commit -- the final commit of this entire P0-P5 effort.

## Next file to modify

`src/training/trainer.py::run_training` (P5's core task — the real
multi-epoch update loop, replacing its current `NotImplementedError`
stub; consumes `build_training_batch`'s P4 output directly).

---

## NEXT OWNER ACTION

**P0, P1, P2, P3, and P4 are all complete.** Begin **Phase P5 — Smoke
Training** per
[PPO_PLAN.md §0.1/P5](PPO_PLAN.md#p5--smoke-training). This is the
FINAL Phase of the entire P0-P5 effort. P5 wires the actual PPO
parameter-update loop (`trainer.py::run_training`) on top of P4's
real rollout -> GAE -> masked-normalized batch pipeline
(`build_training_batch`), verified against the real `MergeEnvironment`
on a small deterministic TRAIN subset -- pipeline correctness, not
performance, is the goal (§0.1/P5's explicit non-goals: no long
training, no millions of steps, no reward-curve tuning, no
hyperparameter tuning, no TRAIN/TUNE split, no W&B sweeps, no VAL
evaluation, no FSM-vs-PPO comparison). Exercise the full checkpoint
save→load→resume→additional-update cycle per §10 (full state, not
just weights). When P5 succeeds, the state becomes **P5 COMPLETE —
WAITING FOR USER TUNING** and this entire P0-P5 effort is done; P6+ is
reserved for the user to run manually.

**Fixed constraint (already applied to every prior Phase): exactly
ONE commit per Phase.** No intermediate "-A"/"-B", sub-stage, or WIP
commits within a Phase. Do all of P5's implementation, its required
tests, and its doc updates first, and only commit once, at the very
end, with everything staged together in a single commit. P0, P1, P2,
P3, and P4 already followed this rule (`chore(ppo): audit frozen
training baseline`, `feat(ppo): add PPO foundation scaffolding (P1)`,
`feat(ppo): implement Reward V0 and W&B logging foundation (P2)`,
`feat(ppo): implement discrete PPO core (P3)`,
`feat(ppo): implement rollout and GAE integration (P4)`). P5 must get
exactly one commit the same way — the final commit of this effort, on
top of the pre-existing `aa8cf5b` docs commit and the five
already-landed P0/P1/P2/P3/P4 commits.
