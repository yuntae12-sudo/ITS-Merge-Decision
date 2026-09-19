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
Foundation, P3 — Discrete PPO Core, P4 — Rollout + GAE Integration, and
P5 — Smoke Training are all COMPLETE. The entire P0-P5 PPO effort is
DONE.** State: **P5 COMPLETE — WAITING FOR USER TUNING**. See
[SMOKE_TRAINING_REPORT.md](SMOKE_TRAINING_REPORT.md) for the full P5
report (architecture, exact smoke-run conditions/results, checkpoint
location, verified resume command, regression numbers, known
limitations, reproduction commands). Working on branch
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

**P5 — Smoke Training** then implemented the real multi-update PPO
training loop (`src/training/trainer.py::run_update`/`run_training`,
both previously `NotImplementedError` stubs), wired both
`scripts/train_ppo.py` and `scripts/smoke_train_ppo.py` into real
working entry points (real config/env/train-state setup, real
`--resume` handling, real checkpoint save at the end), and ran two
real smoke-training stages against the real `MergeEnvironment`
(`downstream_mode="frenet_mpc"`, `WANDB_MODE=offline`, seed 0):

- **Stage 1** (fresh init): 2 maneuvers (`MAN_0001`, `MAN_0002`), 2
  updates, 60 max steps/episode. Finished in 61.1s;
  `global_env_step=70`, `ppo_update_step=2`. Checkpoint saved to
  `outputs/ppo_checkpoints/smoke_stage1_final.pkl`.
- **Stage 2** (`--resume` from Stage 1's checkpoint, a **separate**
  `python` process invocation): 4 maneuvers, 3 more updates, 60 max
  steps/episode. Printed `Resumed: global_env_step=70,
  ppo_update_step=2`, then finished in 211.0s with
  `global_env_step=403`, `ppo_update_step=5` — counters genuinely
  CONTINUED (never reset to 0) across the resume boundary. Checkpoint
  saved to `outputs/ppo_checkpoints/smoke_stage2_final.pkl` (md5
  differs from Stage 1's checkpoint, confirming genuinely different
  parameters after the additional updates).

`run_update` filters every Actor-side quantity (`observation`,
`action`, `log_prob`, `advantages`) to `policy_mask == 1` rows before
computing policy loss/entropy/gradients (§7.2 -- `build_training_batch`
deliberately does not do this filtering itself); the value-loss pass
uses the full, unfiltered trajectory. Both policy and value network
parameters were confirmed to provably differ (leaf-by-leaf numpy
comparison) before vs. after `run_update`/`run_training`, in both a
dedicated unit test
(`tests/training/test_run_training.py::test_run_update_changes_both_policy_and_value_params`)
and via the real Stage1->Stage2 checkpoint md5 difference above. The
full save -> load -> resume -> additional-update cycle (§10) was
verified both via the real two-process CLI run above and independently
via `tests/training/test_checkpoint.py::test_full_save_load_resume_additional_update_cycle`,
which round-trips real Flax/optax pytrees through `save_checkpoint`/
`load_checkpoint` and confirms an additional real update happens on
top of the loaded state with step counters continuing.

Added net **7 new P5 tests** over the P4 baseline:
`tests/training/test_checkpoint.py` (new file, 3 -- real-JAX-pytree
round-trip, params usable for a real forward pass after reload, full
save->load->resume->additional-update cycle), `tests/training/
test_run_training.py` (new file, 5 -- `run_update` changes both
policy/value params + finite metrics, `run_update` raises on an
all-masked-out batch, `run_training` end-to-end changes parameters,
`run_training` continues step counters from a nonzero starting point,
action-ratio metrics reflect only `policy_mask==1` rows), plus
`tests/training/test_trainer.py` (net 0 -- the old
`run_training`-still-raises-`NotImplementedError` test was replaced
with `test_run_training_is_real_not_a_stub`, and the SS7.2 test-B-at-
the-real-PPO-loss-level test `test_masked_frames_do_not_affect_real_ppo_loss_statistics`
was already present from P4's own commit) and `tests/training/
test_imports.py` (-1 -- the now-obsolete `run_training`-stub check was
removed). See [SMOKE_TRAINING_REPORT.md](SMOKE_TRAINING_REPORT.md) §10
for the exact reconciliation (580 P4-baseline + 7 = 587).

Confirmed no other `pytest` process was running before the full-suite
run. Ran the FULL existing regression suite (Phase 1-3 + P1 + P2 + P3
+ P4 + P5 tests together) and waited for the real process exit (PID
495630, watched via a `Monitor` until-loop polling the actual PID, not
a fixed-duration sleep): **587 passed**, 0 failed, 80 warnings (all
`optax.global_norm`-deprecation warnings, not failures), in 1948.37s
(0:32:28), exit code 0, independently confirmed via `pytest tests/
--collect-only -q` -> "587 tests collected". `git diff --stat --
src/environment/ src/planning/ src/control/ src/scenarios/` confirmed
empty -- zero frozen Phase 1-3 files touched across the ENTIRE P0-P5
effort, including P5.

## Last successful test

Pre-P6 hardening pass, full regression (`pytest tests/ -q`, clean solo
run, no concurrent-pytest contention, verified by the orchestrating
session): **629 passed**, 0 failed, in 2169.30s (0:36:09).
`pytest tests/ --collect-only -q` independently confirms 629 tests
collected. Targeted run over the 5 changed/new test files: **65
passed**, 0 failed, 238 warnings (all pre-existing
`optax.global_norm`-deprecation warnings, not failures), in 233.97s.

(Prior: P5's `pytest tests/ -q`, clean solo run -> **587 passed**,
0 failed, 80 warnings, 1948.37s (0:32:28).)

## Failed tests

None.

## Problems found

Seven correctness/instrumentation issues found and fixed in the
Pre-P6 hardening pass (episode-aware GAE cross-episode advantage
leakage; PPO update metric aggregation silently using only the last
minibatch/epoch's value instead of the full sweep; missing exact-KL
diagnostic; `value_coef` documentation inconsistency risk; missing
NumPy RNG checkpointing causing resumed minibatch order to silently
diverge; several W&B `MINIMUM_METRICS` fields listed in the module but
never actually computed/logged by `run_training`; `reward/terminal`
undercounting `TRUNCATION_HORIZON` episodes) — see
[PRE_P6_REPORT.md](PRE_P6_REPORT.md) for the full itemized fix list
and verification evidence. All seven are now fixed and covered by
tests. No contradiction found between PPO_PLAN.md and the current
codebase in P0, P1, P2, P3, P4, or P5 themselves — these were bugs in
the P0-P5 implementation discovered by a dedicated hardening pass, not
plan/implementation contradictions.

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
- `fe8edc9` — `feat(ppo): implement rollout and GAE integration (P4)`
  (P4 completion)
- `755a498` — `test(ppo): validate smoke training pipeline (P5)` — the
  sixth and FINAL commit of the P0-P5 effort (`feat/ppo-phase0-5`,
  merged to `main` at `c00743a` via PR #1).
- Pre-P6 hardening completion commit (`fix(ppo): harden pre-P6
  training correctness and diagnostics`): see `git log` on
  `feat/ppo-pre-p6` for the exact SHA — the single commit for this
  entire follow-up phase, based on `main` at `c00743a`.

## Running processes

None.

## Latest checkpoint

Real checkpoints from the Pre-P6 hardening pass's fresh + resume
smoke-training re-run (against the current, post-fix code):
`outputs/ppo_checkpoints/pre_p6_smoke.pkl` (fresh run) and
`outputs/ppo_checkpoints/pre_p6_smoke_resumed.pkl` (resumed from the
fresh run, 1 more update). Both carry the full §10 contract PLUS the
new `numpy_rng_state` field (Fix 5) as a plain-`pickle` dump of
`CheckpointPayload`. Round-trip and resume verified both by real
script runs (the resumed run's printed `global_env_step`/
`ppo_update_step` correctly continued from the fresh run's final
values, 70/2 -> 103/3) and by `tests/training/test_pre_p6_hardening.py`/
`test_checkpoint.py`. See [PRE_P6_REPORT.md](PRE_P6_REPORT.md) §5 for
the exact commands and full numeric results.

(Prior: P5's `outputs/ppo_checkpoints/smoke_stage1_final.pkl`/
`smoke_stage2_final.pkl` — still present on disk, predate the Pre-P6
fixes, superseded by the above as the "latest" evidence but not
deleted.)

## Next command to run

None queued — both the P0-P5 effort and the Pre-P6 hardening pass are
complete. If resuming this repo later for P6+ work
(reward/hyperparameter tuning, a real TRAIN/TUNE split, longer
training, VAL evaluation, FSM-vs-PPO comparison), start by reading
[PRE_P6_REPORT.md](PRE_P6_REPORT.md) in full (and
[SMOKE_TRAINING_REPORT.md](SMOKE_TRAINING_REPORT.md) for the earlier
P0-P5 context it builds on), then follow the NEXT OWNER ACTION below.

## Next file to modify

None queued for either effort. The Pre-P6 hardening pass already
closed out the previously-noted gap (the remaining §8
`MINIMUM_METRICS` W&B fields are now actually computed and logged by
`src/training/trainer.py::run_training` — success/collision/offroad/
timeout rates, downstream intervention diagnostics, exact-KL,
policy_decision_count/physical_step_count — see
[PRE_P6_REPORT.md](PRE_P6_REPORT.md) §3/§6). Any further change is a
P6+ user decision, not a queued action.

---

## Small follow-up correction (post Pre-P6 hardening)

A narrow, additive bug fix landed on this branch immediately after the
Pre-P6 hardening pass's final commit: the `reward/terminal`/
`reward/decision_cost` W&B breakdown (Fix 7) still mis-split a
terminal/truncated outcome that landed on a real policy-decision step
(e.g. SUCCESS `+1.0` plus `-0.01` decision cost was fully counted as
`reward/terminal`). Fixed via Transition-level propagation of
`MergeRewardWrapper`'s own already-correct per-step components — no
change to Reward V0's values, PPO hyperparameters, or the end-state
below. See [PRE_P6_REPORT.md §10](PRE_P6_REPORT.md#10-follow-up-fix-reward-component-logging-correctness-post-report)
for the full writeup (12 new tests, full regression 641 passed/0
failed).

## NEXT OWNER ACTION

**State: P6 WORKSPACE READY — WAITING FOR USER BASELINE RUN.**

The entire P0-P5 PPO effort, the follow-up Pre-P6
correctness/instrumentation hardening pass, and the P6 workspace setup
described below are all COMPLETE. Every item in
[PPO_PLAN.md §12](PPO_PLAN.md#12-completion-checklist) holds — see
[SMOKE_TRAINING_REPORT.md](SMOKE_TRAINING_REPORT.md) for the P0-P5
itemized evidence and [PRE_P6_REPORT.md](PRE_P6_REPORT.md) for the
Pre-P6 hardening evidence (7 fixes, real re-run smoke-training numbers,
full regression result, scope-compliance verification).

**Git state:** `feat/ppo-pre-p6` (commits `d75b593`, `f947b26`) merged
into `main` via PR #2, merge SHA `066b123`. A new branch,
`exp/ppo-p6-tuning`, was created from that `main` and carries exactly
one setup commit adding `configs/ppo/ppo_p6_baseline.yaml` (byte-
identical to `ppo_base.yaml` — no tuning applied) and
[`docs/ppo/P6_EXPERIMENT_GUIDE.md`](P6_EXPERIMENT_GUIDE.md) (the
one-axis-per-experiment policy and W&B checklist). No PPO source code
was touched in that setup commit.

**No training was executed as part of this setup.** No tuning, no
W&B sweep, no TRAIN/TUNE dataset split, no canonical VAL evaluation.
`exp/ppo-p6-tuning` is ready for the user to run the first real P6
baseline experiment themselves:

```
PYTHONPATH=. python scripts/train_ppo.py \
  --ppo-config configs/ppo/ppo_p6_baseline.yaml \
  --max-maneuvers <YOUR_CHOSEN_COUNT> \
  --num-updates <YOUR_CHOSEN_COUNT> \
  --max-episode-steps <YOUR_CHOSEN_COUNT> \
  --checkpoint-path outputs/ppo_checkpoints/p6_baseline_seed0.pkl
```

**The user must review the real W&B diagnostics themselves before any
further PPO work is done.** Form your own judgment about the observed
episode returns / entropy / success-collision-offroad rates /
downstream-intervention rates / loss curves (see
[P6_EXPERIMENT_GUIDE.md](P6_EXPERIMENT_GUIDE.md) Rule 4 for the
metric checklist), and decide — based on that review, not on any
recommendation baked into this codebase or these docs — whether/how to
proceed (reward-term additions or reweighting, hyperparameter tuning,
creating a real PPO-FIT/PPO-TUNE dataset split, longer training runs,
canonical VAL evaluation, or an FSM-vs-PPO comparison). None of that
work has been started, scoped, or recommended here — the choice of
what (if anything) to tune next, and the training budget
(maneuvers/updates/episode-step-cap) to use, is explicitly the user's
call, not an automated next step for a future session to take on its
own initiative. Per Rule 2 in the experiment guide, create a new
experiment config under `configs/ppo/experiments/` for each change
rather than editing `ppo_p6_baseline.yaml` itself.

If a future session is asked to continue this work, it should treat
any further code/algorithm change as the START of a new,
separately-scoped effort (its own plan, its own branch decision, its
own commit cadence) built on top of this frozen P0-P5 + Pre-P6
foundation.
