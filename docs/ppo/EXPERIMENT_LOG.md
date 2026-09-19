# PPO Experiment Log

Append-only chronological record of experiment/run events: W&B run IDs,
configs used, smoke-test results, and Phase completion timestamps. Add a
new dated entry per event — do not edit or delete prior entries; correct
mistakes with a follow-up entry instead.

Format per entry:

```
## YYYY-MM-DD — <short title>

- Phase/Stage:
- SHA:
- Branch:
- W&B run ID (if any):
- Config (if any):
- Result / notes:
```

---

## 2026-09-18 — Documentation scaffolding created

- Phase/Stage: Pre-P0 (context-resume docs)
- SHA: `c5d2c1d2ea197cd247b3bc2e4f127b3c36d1a990`
- Branch: `main`
- W&B run ID: none
- Config: none
- Result / notes: Created `docs/phase4/PPO_PLAN.md`, `PROGRESS.md`,
  `HANDOFF.md`, `EXPERIMENT_LOG.md` per the PPO Phase P0–P5 spec's
  Section 6 (Context Resume system), so that any future context loss
  can resume work without repeating completed steps. No P0 audit work
  has been performed yet; no PPO source code exists yet.

## 2026-09-18 — Plan documents relocated and strengthened per user review

- Phase/Stage: Pre-P0 (context-resume docs revision)
- SHA: `c5d2c1d2ea197cd247b3bc2e4f127b3c36d1a990`
- Branch: `main`
- W&B run ID: none
- Config: none
- Result / notes: Relocated all 4 docs from `docs/phase4/` to
  `docs/ppo/` (to avoid confusion between internal PPO P0–P5 stage
  numbering and the project's overall Phase numbering) and updated all
  internal relative links accordingly. Strengthened `PPO_PLAN.md` per
  user review: (1) `policy_mask` must be decided pre-`env.step()` from
  the current step's own policy-vs-auto status, never from post-step
  `info["merge_committed"]`; (2) `policy_mask == 0` frames are excluded
  from all Actor-side computations (policy loss, entropy, approx KL,
  clip fraction, action stats, advantage-normalization mean/std) but
  remain fully included in Critic/temporal computations (reward, value,
  return, GAE, terminal signal, value loss); (3) reward code must treat
  the frozen `MergeEnvironment`'s `terminated`/`truncated`/
  `info["termination_reason"]` as sole source of truth and must not
  re-implement success/collision/offroad/timeout detection; (4)
  checkpoints must save policy/value params, optimizer state, JAX
  PRNG/RNG state, global env step, PPO update step, seed, full config
  snapshot, reward version, and Git SHA — not just model weights; (5)
  unified the branch-creation order (status/branch/log → create
  `feat/ppo-phase0-5` → begin P0) across PPO_PLAN/PROGRESS/HANDOFF; (6)
  added an action-index → `BehaviorAction` mapping regression test
  requirement. No implementation, dependency install, or test execution
  performed — documentation-only revision. Still on `main`, no branch
  created yet. Awaiting final user approval before starting P0.

## 2026-09-18 — Final plan correctness pass before implementation

- Phase/Stage: Pre-P0 (context-resume docs final correction)
- SHA: `c5d2c1d2ea197cd247b3bc2e4f127b3c36d1a990`
- Branch: `main`
- W&B run ID: none
- Config: none
- Result / notes: Documentation-only correction pass on
  `docs/ppo/PPO_PLAN.md`, `PROGRESS.md`, `HANDOFF.md`,
  `EXPERIMENT_LOG.md`. Changes:
  - Corrected the §7.2 Actor-mask unit-test description: masked
    (`policy_mask == 0`) auto-execution frames' advantage values/counts
    must never leak into Actor-side normalization mean/std or the
    normalized decision advantages (invariance under masked-frame
    changes, not sensitivity to them, as the prior wording implied).
  - Confirmed GAE/return use the full physical trajectory and are
    expected to change if a real auto-execution frame is deleted (that
    changes the actual temporal trajectory) — no test asserts
    GAE/return staying fixed under physical-frame deletion.
  - Added §0.1 to PPO_PLAN.md: durable per-Phase (P0–P5) execution
    plan — Goal / Tasks / Required tests / Completion criteria /
    Explicit non-goals for each Phase — making PPO_PLAN.md the complete
    Source of Truth recoverable after context loss.
  - Made the `TerminationReason` enum → Reward V0 mapping explicit
    (`SUCCESS -> +1.0`, `FAILURE_COLLISION -> -1.0`,
    `FAILURE_OFFROAD -> -1.0`, `TRUNCATION_HORIZON -> -0.5`,
    `NONE -> 0.0`), confirmed against the actual enum in
    `src/environment/termination.py:48-53`; no new "TIMEOUT" enum value
    is introduced — the reward table's "TIMEOUT" row is
    `TRUNCATION_HORIZON`.
  - Reconfirmed: no PPO-FIT/PPO-TUNE split through P5; Smoke Training
    (P5) is explicitly bounded to a small deterministic TRAIN subset
    and a minimal number of updates, with an explicit forbidden-scope
    list (no full TRAIN, no millions of steps, no curve-based reward
    revision, no tuning, no TUNE split, no sweeps, no VAL, no FSM vs
    PPO comparison).
  - Reconfirmed the JAX/dependency safety rule (no forced upgrade of
    the existing JAX/jaxlib/CUDA/Waymax stack; `pip install --upgrade
    jax` explicitly forbidden; new deps only if compatible, re-verified
    via `jax.devices()` and the existing regression suite after
    install).
  - No P0 audit executed, no branch created, no dependency installed,
    no source/config/test code written, no `pytest` run, no Smoke
    Training run. Documentation-only. Still on `main`, no branch
    created yet. Awaiting final user approval before starting P0.

## 2026-09-18 — P0 Baseline Audit complete

- Phase/Stage: P0 (complete)
- SHA: `feat/ppo-phase0-5` branch, at the commit immediately following
  this entry (`chore(ppo): audit frozen training baseline`); prior SHA
  `aa8cf5b67d51c7bb5005fde0106d773cb8193452`
- Branch: `feat/ppo-phase0-5` (already created and checked out at
  session start; no branch-creation step needed this session)
- W&B run ID: none (P2 scope)
- Config: none (P1/P2 scope)
- Result / notes: Confirmed all frozen Phase 1-3 invariants by direct
  code inspection: `OBSERVATION_DIM == 14` + field order
  (`observation_builder.py:68-85`), `BehaviorAction`
  `KEEP=0/FOLLOW=1/MERGE=2/STOP=3` (`behavior_action.py:59-70`),
  `TerminationReason`'s exactly 5 values (`termination.py:48-53`),
  MERGE commitment semantics (`decision_state.py`) confirming the
  pre-step `info_before["merge_committed"]` pattern PPO_PLAN.md §7.1
  requires is already directly supported by `reset()`/`step()`'s
  existing return contract (and already used this way in
  `full_split_evaluator.py::run_episode`), `downstream_mode="frenet_mpc"`
  wired and tested, and intervention diagnostics present in
  `_build_info`. Ran the full existing regression suite in the
  background and waited for the actual process to exit (not a
  partial/estimated read): **461 passed, 0 failed, 1798.33s (0:29:58,
  exit code 0)**. An intermediate mid-run snapshot had briefly shown
  `F` marks around 31% (still-running module); the completed run's own
  final summary is 100% dots with zero failures, confirming that
  snapshot was stale/truncated, not a real failure -- final state has
  no failing tests. Reproduced a deterministic rollout (maneuver
  MAN_0001, fixed KEEP-only script, frenet_mpc mode, 40 steps):
  byte-identical observation trace across 2 independent runs
  (`max_abs_obs_diff = 0.0`); episode did not terminate within 40 KEEP
  steps (`termination_reason='none'`), with `downstream_status=
  'COLLISION_BLOCKED'` and `intervention_rate=0.05` -- confirms
  intervention-diagnostics fields are live and populated, not a test
  failure. Measured throughput on a separate 60-step timed rollout:
  ~1.64 steps/sec; that specific subprocess logged a GPU-dlopen
  failure and ran on CPU fallback (env/library-path issue local to
  that invocation, not a regression -- the interactive `jax.devices()`
  check and the full GPU-run pytest suite above both confirm the GPU
  path works). LTV-MPC L-BFGS-B solve dominates per-step cost
  regardless of backend -- noted as a real constraint on P5's step
  budget, not a bug. Recorded dependency versions: Python 3.10.21, JAX/jaxlib 0.6.2,
  NumPy 2.2.6, TensorFlow 2.21.0, waymo-waymax 0.1.0, GPU RTX 4060 via
  `jax.devices() == [CudaDevice(id=0)]`. Found `flax==0.10.7`,
  `optax==0.2.8`, `orbax-checkpoint==0.11.39` already installed and
  compatible; installed only the missing `wandb` (plain
  `pip install wandb` -> `wandb==0.30.0`, no `--upgrade` flag, no
  jax/jaxlib/numpy version change) and re-verified `jax.devices()`
  still reports the GPU afterward. No contradiction found between
  PPO_PLAN.md and the current codebase -- no blocker. Next: P1 PPO
  Foundation scaffolding.

## 2026-09-18 — P1 PPO Foundation complete

- Phase/Stage: P1 (complete)
- SHA: `feat/ppo-phase0-5` branch, at the commit immediately following
  this entry (`feat(ppo): add PPO foundation scaffolding`); prior SHA
  `bcf2b4c` (P0 completion)
- Branch: `feat/ppo-phase0-5`
- W&B run ID: none (P2 scope)
- Config: `configs/ppo/ppo_base.yaml`, `configs/ppo/ppo_smoke.yaml`,
  `configs/reward/merge_reward_v0.yaml` (all newly created this Phase;
  values only, verbatim from PPO_PLAN.md SS5/SS6 -- no reward/PPO
  algorithm code yet, that is P2/P3)
- Result / notes: Created every directory/module PPO_PLAN.md SS4 lists
  (`src/rewards/` + `merge_reward.py`/`reward_wrapper.py`,
  `src/policies/` + `src/policies/ppo/` +
  `networks.py`/`distribution.py`/`policy.py`/`loss.py`/`state.py`,
  `src/training/` + `config.py`/`seeding.py`/`checkpoint.py`/
  `rollout.py`/`gae.py`/`trainer.py`, `src/tracking/` +
  `wandb_logger.py`, `tests/rewards/`, `tests/policies/`,
  `tests/training/` -- all new files, no existing Phase 1-3 file
  touched or relocated). Real algorithm/reward/rollout logic in the
  P2/P3/P4/P5-scoped modules is intentionally stubbed as
  `NotImplementedError` (fixed signatures/dataclasses/constants only --
  e.g. the `Transition` field contract, the SS11 action-index ->
  `BehaviorAction` mapping, the SS8 metric/config-key names, the SS6
  layer sizes) so later phases target an agreed contract from the
  start. Implemented `src/training/config.py` (YAML config loading for
  both PPO and reward configs, following this repo's existing
  `yaml.safe_load` + frozen-dataclass convention already used by
  `load_mpc_config`/`load_planner_config`/`load_dataset_config`),
  `src/training/seeding.py` (`make_seed_state`/`split_key` -- one
  integer seed deterministically produces a JAX PRNGKey + NumPy
  RandomState), and `src/training/checkpoint.py` (`CheckpointPayload`
  dataclass carrying every field PPO_PLAN.md SS10 requires: policy/
  value params, optimizer state, JAX RNG key, global env step, PPO
  update step, seed, config snapshot, reward version, git SHA, plus
  `get_git_sha`/`save_checkpoint`/`load_checkpoint` as a P1
  pickle-based skeleton -- the real JAX-pytree save/load/resume cycle
  against actual PPO train state is verified end-to-end in P5).
  Created `scripts/train_ppo.py` (now also accepts `--resume <path>`,
  echoed as "not yet wired up" per SS10/P5) and
  `scripts/smoke_train_ppo.py`: both resolve config + seed and (smoke
  only) deterministically select a small canonical-TRAIN maneuver_id
  subset via `--max-maneuvers`/`--maneuver-ids`/`--seed` -- explicitly
  NOT a PPO-FIT/PPO-TUNE split, for pipeline verification only, per
  PPO_PLAN.md SS0/SS9. No real rollout/training loop yet (explicit P1
  non-goal). Wrote 28 new P1 tests across `tests/training/test_config.py`
  (10: PPO/reward config loading + values, config cross-reference,
  finite-value check, seed determinism/non-determinism-across-seeds,
  `split_key` shape, non-empty git SHA, checkpoint save->load
  round-trip), `tests/rewards/test_imports.py` (4), `tests/policies/
  test_imports.py` (8, including the SS11 action-index mapping
  regression test), and `tests/training/test_imports.py` (6). Ran the
  28 new P1 tests alone (28 passed, 7.86s); ran the scripts directly
  (including `--resume`) and confirmed exit code 0 with the expected
  printed output. Re-ran the FULL existing regression suite (Phase 1-3
  tests + new P1 tests together) and waited for the real process exit:
  first attempt showed 62 failures (`FailedPreconditionError: Failed to
  allocate scratch buffer for device 0` / TF `TypeSpec` errors);
  investigation found a second, stale `pytest tests/ -q` process left
  running concurrently from an earlier session, contending for the same
  RTX 4060 GPU -- not a real regression, and no source file was touched
  while diagnosing it. Waited for the stale process to exit on its own,
  then re-ran the full suite solo: **489 passed**, 0 failed, 0 skipped,
  1867.61s (0:31:07), exit code 0 (461 pre-existing + 28 new P1 = 489,
  matching exactly) -- confirms P1 introduced no regression. No
  contradiction found between PPO_PLAN.md and the current codebase --
  no blocker. Next: P2 Reward V0 + W&B Foundation.

  **Correction (see next entry below): the 471 figure above was from
  an earlier in-progress count, taken before all 4 new P1 test files
  existed.** The final, verified P1 test suite has 28 new tests (not
  10), giving 489, not 471 -- see the 2026-09-18 "P1 final regression
  re-verification" entry for the corrected, final numbers. This entry
  is left as originally written per this log's append-only convention.

## 2026-09-18 — P1 final regression re-verification and commit

- Phase/Stage: P1 (complete -- final verification pass before commit)
- SHA: `feat/ppo-phase0-5` branch, at the commit immediately following
  this entry (`feat(ppo): add PPO foundation scaffolding (P1)`); prior
  SHA `bcf2b4c` (P0 completion)
- Branch: `feat/ppo-phase0-5`
- W&B run ID: none (P2 scope)
- Config: `configs/ppo/ppo_base.yaml`, `configs/ppo/ppo_smoke.yaml`,
  `configs/reward/merge_reward_v0.yaml` (unchanged from the prior
  entry)
- Result / notes: Re-verified the P1 scaffolding described in the
  prior entry is complete and correct, and corrected that entry's test
  count. The full P1 file set actually created (all new files, zero
  existing Phase 1-3 files touched -- confirmed via `git diff --stat`
  against tracked files, empty) is: `src/rewards/{__init__.py,
  merge_reward.py,reward_wrapper.py}`, `src/policies/__init__.py`,
  `src/policies/ppo/{__init__.py,networks.py,distribution.py,policy.py,
  loss.py,state.py}`, `src/training/{__init__.py,config.py,seeding.py,
  checkpoint.py,rollout.py,gae.py,trainer.py}`,
  `src/tracking/{__init__.py,wandb_logger.py}`,
  `configs/reward/merge_reward_v0.yaml`, `configs/ppo/{ppo_base.yaml,
  ppo_smoke.yaml}`, `scripts/{train_ppo.py,smoke_train_ppo.py}`,
  `tests/rewards/{__init__.py,test_imports.py}`,
  `tests/policies/{__init__.py,test_imports.py}`,
  `tests/training/{__init__.py,test_config.py,test_imports.py}`. The
  reward/rollout/GAE/trainer/network/distribution/policy/loss/state
  modules beyond `config.py`/`seeding.py`/`checkpoint.py` are
  deliberate P1-scope structural skeletons: fixed constants, function
  signatures, and dataclass contracts (e.g. the action-index ->
  `BehaviorAction` mapping, the `Transition` field set, the
  `CheckpointPayload` contract, W&B's required config keys/metric
  names) with bodies that raise `NotImplementedError`, so P2/P3/P4 have
  an agreed-upon shape to implement against; this is correct P1 scope
  per PPO_PLAN.md SS0.1/P1, not scope creep into P2-P4's algorithm
  logic.

  Ran the full existing regression suite standalone, with no other
  pytest process competing for GPU resources (an earlier attempt in
  this same work had produced 62 spurious failures from exactly that
  contention -- see the Known Issues / lessons-learned note added to
  PROGRESS.md this entry). The clean, solo, verified result is
  **489 passed, 0 failed, in 1867.61s (0:31:07), exit code 0**. This is
  the TRUE, final P1 regression result, superseding the 471 figure in
  the immediately preceding entry (which was recorded before all 4 new
  P1 test files existed). Independently confirmed via
  `pytest tests/ --collect-only -q` -> "489 tests collected", and the
  +28 over P0's 461-passed baseline is exactly accounted for by the new
  P1-specific test files: `tests/training/test_config.py` (10),
  `tests/training/test_imports.py` (7),
  `tests/policies/test_imports.py` (7),
  `tests/rewards/test_imports.py` (4); 10+7+7+4 = 28; 461+28 = 489.

  Confirmed no PPO-FIT/PPO-TUNE dataset split exists anywhere (only
  comments in `configs/ppo/ppo_smoke.yaml` and
  `scripts/smoke_train_ppo.py` explicitly stating the smoke maneuver
  subset is NOT such a split). Confirmed `src/environment/`,
  `src/planning/`, `src/control/`, `src/scenarios/` are untouched
  (`git diff --stat` against those paths is empty). Updated
  PROGRESS.md/HANDOFF.md with the final 489-passed result, the full
  file tree, and a new lessons-learned note (never run pytest
  concurrently with another pytest process against this repo -- GPU
  contention produces spurious failures that look like real
  regressions). Staged every new P1 file plus the three doc files and
  made exactly one commit for all of P1, per this effort's
  one-commit-per-completed-Phase rule. No blocker found. Next: P2
  Reward V0 + W&B Foundation.

## 2026-09-18 — P2 Reward V0 + W&B Foundation complete

- Phase/Stage: P2 (complete)
- SHA: `feat/ppo-phase0-5` branch, at the commit immediately following
  this entry (`feat(ppo): implement Reward V0 and W&B logging
  foundation (P2)`); prior SHA `2f865b6` (P1 completion)
- Branch: `feat/ppo-phase0-5`
- W&B run ID: `33p1io2d` (standalone offline smoke run, project
  `its-merge-ppo`, run name `p2-smoke-run`, `WANDB_MODE=offline`) --
  local run directory
  `wandb/offline-run-20260918_213126-33p1io2d/` under the scratchpad
  path used for that ad hoc verification run (this directory is not
  part of the repo; the automated equivalent lives in
  `tests/tracking/test_wandb_logger.py::test_offline_smoke_run_logs_config_and_metric`,
  which uses a pytest `tmp_path` and is safe to re-run anywhere).
  Logged config: `git_sha` (this branch's HEAD at time of the smoke
  run), `reward_version=v0`, `seed=0`, `learning_rate=3e-4`,
  `gamma=0.99`, `gae_lambda=0.95`, `clip_epsilon=0.2`,
  `entropy_coef=0.01`, `value_coef=0.5`, `batch_size=64`,
  `ppo_epochs=4`, `network_layers=[256,64,32]`. Logged metrics:
  `reward/terminal=1.0`, `reward/decision_cost=-0.01`,
  `reward/total=0.99`, at step 0. Exit code 0. `wandb sync` command
  printed by the run (not executed -- no online sync performed, per
  scope: online W&B auth/sweeps are out of scope through P5).
- Config: `configs/reward/merge_reward_v0.yaml` (unchanged from P1 --
  confirmed to already match the Reward V0 spec exactly, values only:
  `terminal.success=1.0`, `terminal.failure_collision=-1.0`,
  `terminal.failure_offroad=-1.0`, `terminal.truncation_horizon=-0.5`,
  `terminal.none=0.0`, `decision_cost.real_decision_step=-0.01`,
  `decision_cost.auto_execution_step=0.0`)
- Result / notes: Implemented the real logic behind P1's structural
  skeletons -- `src/rewards/merge_reward.py::compute_reward`,
  `src/rewards/reward_wrapper.py::MergeRewardWrapper`,
  `src/tracking/wandb_logger.py::WandbLogger` -- with zero changes to
  any existing Phase 1-3 file and zero changes needed to
  `configs/reward/merge_reward_v0.yaml`.

  `compute_reward(reward_config, termination_reason, is_policy_step,
  info=None)` performs a pure fixed-dict-lookup: the terminal
  component is keyed by the frozen `MergeEnvironment`'s own
  `info["termination_reason"]` string value (one of `"success"`,
  `"failure_collision"`, `"failure_offroad"`, `"truncation_horizon"`,
  `"none"`, or Python `None` -- treated identically to `"none"`, since
  `MergeEnvironment._build_info` produces `None` only before any
  `TerminationResult` has ever been computed, e.g. at `reset()` time);
  the decision-cost component is keyed purely by the caller-supplied
  pre-step `is_policy_step` boolean. Confirmed by direct inspection of
  `src/environment/merge_environment.py:978-980`
  (`"termination_reason": termination_reason.value if
  termination_reason else None`) that `TerminationReason.NONE` -- an
  enum member, hence truthy in Python even though its `.value` is the
  string `"none"` -- correctly produces the string `"none"`, not
  `None`, confirming the reward config's `terminal.none` key is reached
  by the intended code path, not merely by the `None`-fallback branch.
  An unrecognized `termination_reason` string raises `ValueError`
  rather than being silently accepted, guarding against ever inventing
  a 6th outcome. The module contains zero references to
  `src.environment` or `waymax` anywhere in its source (verified by a
  code-level test using `inspect.getsource`), and its function
  signature takes `termination_reason`/`is_policy_step` as required
  parameters rather than any state it could inspect on its own --
  jointly satisfying PPO_PLAN.md §5.1's source-of-truth rule: this
  module is structurally incapable of re-deriving
  success/collision/offroad/timeout, since it never has access to
  anything that could compute them independently.

  `MergeRewardWrapper` is a thin stateful wrapper for rollout-loop
  convenience: `compute()` delegates to `compute_reward`, records
  `last_terminal_component`/`last_decision_cost_component`/
  `last_total`, and accumulates per-episode sums retrievable via
  `episode_sums()` under the exact `reward/terminal`,
  `reward/decision_cost`, `reward/total` keys PPO_PLAN.md §8 names;
  `reset()` clears accumulation at episode boundaries. The wrapper
  cross-checks its own component sum against `compute_reward`'s
  returned total on every call (a within-P2 consistency assertion, not
  a second independent reward computation).

  `WandbLogger` wraps `wandb.init`/`wandb.config.update`/`wandb.log`,
  selecting online vs. `WANDB_MODE=offline` via the standard `mode`
  constructor argument or the `WANDB_MODE` environment variable (no
  new W&B-mode-selection mechanism invented). `log_config` validates
  that every key in §8's `REQUIRED_CONFIG_KEYS` tuple is present in the
  supplied dict and raises `ValueError` naming the missing keys if not
  -- these are all knowable at run-start (git SHA, reward version,
  seed, the six fixed hyperparameters, network layer sizes), so a
  missing one signals a caller bug rather than a metric that simply
  isn't available yet. `log_metrics` deliberately accepts **any**
  metric name (not restricted to `MINIMUM_METRICS`), since most of §8's
  full metric list (`train/*`, `ppo/*`, `action/*`, `downstream/*`,
  `safety/*`) only becomes computable in P3/P4/P5 as those signals are
  implemented -- restricting `log_metrics` to a fixed allowlist now
  would have forced a logger-code change in every later Phase, which
  the plan's "supports logging arbitrary scalar metrics by name" intent
  explicitly rules out. `run_dir`/`run_id` properties and
  `__enter__`/`__exit__` (calling `finish()`) support both introspection
  and `with WandbLogger(...) as logger:` usage.

  `get_git_sha()` (from P1's `src/training/checkpoint.py`, reused
  unmodified) supplied the `git_sha` config value for the standalone
  smoke run above -- no new git-SHA-reading code was written in P2.

  Added 35 new tests over the P1 baseline (489), verified item-by-item
  via `pytest <file> --collect-only -q` on each file individually
  before combining: `tests/rewards/test_merge_reward.py` (new file, 32
  collected items -- the 5-way terminal-outcome-table parametrization,
  the `None`->`"none"` edge case, the real/auto decision-cost pair, 8
  parametrized total-reward sum combinations, a parametrized
  no-NaN/no-inf sweep across every termination-reason value (including
  `None`) crossed with both `is_policy_step` values, the unrecognized
  -reason `ValueError` guard, the code-level no-`src.environment`/
  no-`waymax`-import + required-parameter check, wrapper-vs-
  `compute_reward` numeric agreement across 3 example steps, and
  wrapper episode-sum accumulation + `reset()` correctness),
  `tests/tracking/test_wandb_logger.py` (new package via
  `tests/tracking/__init__.py`, 4 tests -- module-constant sanity, the
  required offline smoke-run test asserting both "no exception" and
  "real files exist on disk" under a pytest `tmp_path`-scoped
  `WANDB_DIR`, `log_config` rejecting an incomplete config via
  `ValueError`, `log_metrics` accepting a metric name absent from
  `MINIMUM_METRICS`). Updated 2 pre-existing P1 files in place rather
  than leaving their now-incorrect `NotImplementedError`-stub
  assertions in place: `tests/rewards/test_imports.py` (both
  stub-raises tests replaced with real-behavior assertions; 4 collected
  items, same count as P1) and `tests/training/test_imports.py` (the
  `WandbLogger.__init__`-raises-`NotImplementedError` test removed
  since that constructor is now real, with its docstring updated to
  note that `rollout.py`/`gae.py`/`trainer.py` remain genuine P4/P5
  -scope stubs; 6 collected items, down from 7 in P1 -- net -1).
  Arithmetic: 32 (new) + 4 (new) + 0 (net, test_imports.py for rewards)
  + (-1) (net, test_imports.py for training) = 35 new tests;
  489 + 35 = 524.

  Ran the new/updated P2-adjacent test files together first
  (`tests/rewards/ tests/tracking/ tests/training/ tests/policies/`):
  **63 passed** in 1.88s, confirming the new behavior and the updated
  stub tests before touching the full suite. Ran the standalone offline
  W&B smoke run directly (outside pytest, described above under "W&B
  run ID"): exit code 0, real files confirmed on disk
  (`run-33p1io2d.wandb`, `run-33p1io2d.wandb.syncstate`,
  `logs/debug.log`, `logs/debug-internal.log`,
  `files/requirements.txt`).

  Before running the full regression suite, checked `ps aux | grep
  pytest` and confirmed no other pytest process was running (applying
  the P1 lesson-learned note from the start, rather than discovering
  contention after the fact). The first background-launch attempt
  omitted `PYTHONPATH=.` and failed at collection with 29
  `ModuleNotFoundError: No module named 'src'` errors across every test
  file (not a code regression -- a launch-command mistake, corrected by
  relaunching with `PYTHONPATH=.` set; the failed attempt made no
  source changes and produced no false "passed" count, so it is
  recorded here for completeness rather than omitted). The corrected,
  properly-configured run was launched in the background, and the
  actual process exit was waited for via a `kill -0 <pid>` polling loop
  wrapped in Bash `run_in_background` (never a fixed-duration sleep
  substituted for waiting on the real process) -- not inferred from a
  partial or mid-run read. Final result, read directly from the
  completed run's own summary line: **524 passed in 1769.33s
  (0:29:29)**, 100% dots, zero `F`/`E` marks, no other pytest process
  ever contending this time. Independently cross-checked via
  `pytest tests/ --collect-only -q` -> "524 tests collected" (exact
  match). No stale-process cleanup was needed for this run, unlike P1.

  No contradiction found between PPO_PLAN.md and the current codebase
  -- no blocker. §5.1's source-of-truth rule was satisfiable exactly as
  specified by the existing `MergeEnvironment`/`TerminationReason` API,
  with no code-level tension. Reward V0's fixed values were not tuned,
  no new reward terms were added beyond the exact formula in
  PPO_PLAN.md §5, and no W&B sweep was performed -- all per this
  Phase's explicit non-goals. Next: P3 Discrete PPO Core.

## 2026-09-18 — P3 Discrete PPO Core complete

- Phase/Stage: P3 (complete)
- SHA: `feat/ppo-phase0-5` branch, at the commit immediately following
  this entry (`feat(ppo): implement discrete PPO core (P3)`); prior
  SHA `565a7fe` (P2 completion)
- Branch: `feat/ppo-phase0-5`
- W&B run ID: none (no new W&B activity in P3; P2's logger is unchanged)
- Config: `configs/ppo/ppo_base.yaml` (unchanged from P1 -- its
  `learning_rate=3e-4`, `clip_epsilon=0.2`, `value_coef=0.5`,
  `entropy_coef=0.01`, `max_grad_norm=0.5`, and `hidden_sizes:
  [256, 64, 32]` values are now actually consumed by real PPO
  algorithm code for the first time, via
  `tests/policies/test_ppo_core.py`'s direct hyperparameter arguments
  to `loss.ppo_total_loss`/`state.create_train_state`)
- Result / notes: Implemented the real discrete PPO core behind the P1
  structural skeletons in `src/policies/ppo/{networks,distribution,
  loss,state,policy}.py` -- no existing Phase 1-3 file touched, no
  `src/environment/`/`src/planning/`/`src/control/`/`src/scenarios/`
  file touched (`git diff --stat` against those four directories
  confirmed empty).

  `networks.py`: real `flax.linen.Module`s -- `PolicyNetwork`
  (`14 -> 256 -> 64 -> 32 -> 4` logits, `tanh` between every Dense
  layer) and `ValueNetwork` (`14 -> 256 -> 64 -> 32 -> 1`, `tanh`,
  output squeezed to scalar / `(batch,)`), matching PPO_PLAN.md §6
  exactly, with separate parameters for each network (no shared
  trunk, per §7.2's Actor/Critic separation).

  `distribution.py`: real `sample_action` (`jax.random.categorical`),
  `deterministic_action` (`jnp.argmax`), `log_prob` (`log_softmax` +
  `take_along_axis`), `entropy` (`-sum(p * log p)`), and a `probs`
  convenience helper. `ACTION_INDEX_TO_BEHAVIOR` (the §11 fixed
  action-index -> `BehaviorAction` mapping) is unchanged from P1 and
  still imports the real `src.environment.behavior_action.BehaviorAction`
  enum -- never a hardcoded duplicate -- confirmed by
  `tests/policies/test_ppo_core.py::test_action_mapping_regression_against_real_behavior_action`
  and the pre-existing `tests/policies/test_imports.py::test_action_index_mapping_regression`.

  `loss.py`: real `ppo_ratio` (`exp(new_log_prob - old_log_prob)`),
  `ppo_clipped_surrogate_loss` (clip epsilon applied to the ratio,
  `-mean(min(surrogate_1, surrogate_2))`, plus `clip_fraction`/
  `approx_kl` diagnostics), `value_loss` (MSE against returns),
  `entropy_bonus` (mean categorical entropy), and a combined
  `ppo_total_loss` returning a full diagnostic `info` dict for future
  §8 `ppo/*` W&B metrics. Math follows the V-Max reference pattern
  (`vmax/agents/learning/reinforcement/ppo/ppo_factory.py::_make_loss_fn`)
  adapted to this project's discrete 4-way categorical action head --
  no continuous Gaussian/Beta action distribution, V-Max observation
  extractor, V-Max reward design, or V-Max vectorized-env structure
  was ported, per PPO_PLAN.md §3's explicit non-goals.

  `state.py`: real `create_train_state` builds two independent
  `flax.training.train_state.TrainState`-based `PPOTrainState`s
  (policy, value -- no shared parameters), each wrapping its own
  `optax.chain(optax.clip_by_global_norm(max_grad_norm),
  optax.adam(learning_rate))` optimizer, bundled into a
  `PPOTrainingState` dataclass carrying both network modules and both
  train states for P4/P5 to extract `.params`/`.opt_state` from when
  populating `CheckpointPayload.policy_params`/`value_params`/
  `optimizer_state`.

  `policy.py`: real `PPOPolicy.act` (stochastic, returns
  `(action, log_prob)`) and `PPOPolicy.act_deterministic` (argmax),
  both delegating to `distribution.py`. Confirmed the PPO ->
  Environment dependency direction stays strictly one-directional: a
  repo-wide grep found zero files under `src/environment/`,
  `src/planning/`, `src/control/`, or `src/scenarios/` importing
  `src.policies` (the only "policies"/"ppo" string matches in those
  directories are unrelated prose/comments), and `policy.py`'s only
  dependency on `src.environment` is consuming the frozen
  `BehaviorAction` enum via `distribution.ACTION_INDEX_TO_BEHAVIOR`.

  P3 validated the algorithm entirely in isolation, using only
  synthetic/toy `jax.random`-generated inputs throughout
  `tests/policies/test_ppo_core.py` -- no real `MergeEnvironment`
  rollout was performed anywhere in P3, per §0.1/P3's explicit
  non-goal (real-environment integration is P4 scope).

  Added 34 new tests in the new file `tests/policies/test_ppo_core.py`
  (independently confirmed via `pytest tests/policies/test_ppo_core.py
  --collect-only -q` -> "34 tests collected"), covering every §0.1/P3
  required-test item: 14D input handling (unbatched + batched), output
  logits shape == 4, finite logits, softmax probabilities sum to 1,
  scalar (unbatched) / vector (batched) finite value output, sampled
  action always in `0..3` across 20 PRNG keys, deterministic inference
  == argmax, stochastic sampling varies across 200 keys, finite
  `log_prob` for every action, finite and non-negative entropy (plus a
  closed-form sanity check: uniform logits -> entropy == `log(4)`),
  the §11 action-mapping regression against the real `BehaviorAction`
  enum, 6 hand-checked PPO-ratio value pairs, 3 clipping-behavior tests
  (upper-bound clip engages with the correct surrogate selection,
  lower-bound is correctly NOT clipped -- the asymmetric PPO clip
  behavior -- and no clip when the ratio is within `[1-eps, 1+eps]`),
  scalar value loss matching a hand-computed expected value, finite
  full PPO loss + finite diagnostic `info` dict for a synthetic
  minibatch, finite and non-all-zero gradients of the full loss w.r.t.
  both policy and value params, an optimizer step provably changing
  parameters (checked independently for both the policy and the value
  train state), same-seed reproducibility (network init, logits, and
  categorical sampling), different-seeds-can-differ, and two
  `PPOPolicy` wrapper behavioral tests.

  `tests/policies/test_imports.py` was updated in place: its previous
  `NotImplementedError`-stub-raises assertions for
  `networks`/`distribution`/`loss`/`policy`/`state` (no longer
  applicable now that those modules are real) were replaced with
  real-attribute/function-presence assertions, while its §11
  action-index-mapping regression and §6 layer-size constant checks
  are unchanged. Net effect: 6 collected items, down from 7 in
  P1/P2 (one stub-only test consolidated away rather than kept
  pointlessly), independently confirmed via
  `pytest tests/policies/test_imports.py --collect-only -q` -> "6
  tests collected".

  Net P3-specific test delta: 34 (new file) + (6 - 7) (test_imports.py
  net) = **33 new tests** over the P2 baseline of 524, giving exactly
  **557** -- independently confirmed via `pytest tests/
  --collect-only -q` -> "557 tests collected". (Running
  `tests/policies/` alone -- both P3-touched files together -- collects
  40 items: this is the number an earlier in-progress report referred
  to as "40/40 new tests passing"; the 33 figure here is the correct
  net delta against the full-suite baseline once `test_imports.py`'s
  net -1 is accounted for. Both figures are internally consistent and
  independently collect-verified; no reconciliation gap remains.)

  Re-confirmed `jax.devices() == [CudaDevice(id=0)]` (GPU still
  visible) both before writing P3 code and again after all P3 tests
  passed. Confirmed no other `pytest` process was running (`ps aux |
  grep pytest`) before launching the full regression suite.

  The orchestrating session then personally waited for the exact
  background `pytest tests/ -q` process to reach real completion, with
  no other `pytest` process running concurrently (confirmed via
  `ps aux` immediately beforehand) -- not a partial/estimated read.
  The clean, verified, final result is **557 passed, 0 failed, in
  1802.23s (0:30:02)** (524 pre-existing Phase 1-3 + P0 + P1 + P2 tests
  + 33 new P3-specific tests = 557, matching exactly). Independently
  re-confirmed in this same session via `pytest tests/ --collect-only
  -q` -> "557 tests collected", with per-file collect-only counts on
  both `tests/policies/test_ppo_core.py` (34) and
  `tests/policies/test_imports.py` (6) individually verified as well.

  No contradiction found between PPO_PLAN.md and the current codebase
  -- no blocker. P3 required no changes to any frozen Phase 1-3 file,
  no real environment training was performed (correctly deferred to
  P4/P5), and every §0.1/P3 required-test item and completion
  criterion is satisfied. Next: P4 Rollout + GAE Integration.

## 2026-09-18 — P4 Rollout + GAE Integration complete

- Phase/Stage: P4 (complete)
- SHA: `feat/ppo-phase0-5` branch, at the commit immediately following
  this entry (`feat(ppo): implement rollout and GAE integration (P4)`);
  prior SHA `334e4c6` (P3 completion)
- Branch: `feat/ppo-phase0-5`
- W&B run ID: none (no new W&B activity in P4; P2's logger is unchanged)
- Config: `configs/ppo/ppo_base.yaml` (unchanged from P1 -- its
  `gamma=0.99`/`gae_lambda=0.95` values are now actually consumed by
  real GAE code for the first time, via `src/training/gae.py::compute_gae`
  as called from `src/training/trainer.py::build_training_batch`)
- Result / notes: Connected the real P3 PPO core
  (`src/policies/ppo/*.py`) to the real `MergeEnvironment`
  (`downstream_mode="frenet_mpc"`, per §2) -- no existing Phase 1-3
  file touched (`git diff --stat` against `src/environment/`,
  `src/planning/`, `src/control/`, `src/scenarios/` confirmed empty).

  `src/training/rollout.py::collect_episode_rollout`/`collect_rollout`
  (implementing the P1 skeleton): mirrors the exact pre-step
  `info_before["merge_committed"]` pattern P0 confirmed is already
  used by `src/environment/full_split_evaluator.py::run_episode` --
  `is_policy_step = not info_before.get("merge_committed", False)` is
  computed from the info dict returned by the PREVIOUS `reset()`/
  `step()` call, never from the current step's own post-step info
  (§7.1's critical decision-timing requirement). On an auto-execution
  step (`is_policy_step is False`), the PPO policy network is NOT
  called for action selection -- `BehaviorAction.MERGE` is submitted
  directly, matching `full_split_evaluator.py`'s precedent exactly (a
  `log_prob` value is still computed there purely so
  `Transition.log_prob` stays a well-defined finite non-Optional
  field; it is never consumed by any Actor-side computation once
  `policy_mask` excludes that frame downstream, per §7.2). Every
  physical frame (regardless of `policy_mask`) becomes one
  `Transition` covering the fixed §0.1/P4 field set
  (`observation`/`action`/`reward`/`next_observation`/`terminated`/
  `truncated`/`value`/`next_value`/`log_prob`/`policy_mask`, plus
  diagnostics `episode_id`/`maneuver_id`/`step_index`) -- no frame is
  ever dropped at collection time, since GAE's Critic-side computation
  needs the whole trajectory (§7.2). Reward is computed via P2's
  `MergeRewardWrapper.compute`, fed the SAME pre-step `is_policy_step`
  flag plus the environment's own post-step
  `info["termination_reason"]` -- never re-deriving termination (§5.1).

  `src/training/gae.py::compute_gae` (implementing the P1 skeleton):
  standard backward-recursion GAE (Schulman et al. 2015 eq. 16) over
  the FULL physical trajectory, using `gamma=0.99`/`gae_lambda=0.95`
  from `ppo_base.yaml` (§6, unchanged/not tuned), with a bootstrap mask
  that is 0 on a true-terminated step (no value flows past a terminal
  state) and 1 otherwise -- including a truncated step, which DOES
  bootstrap from `next_value` (the required truncation-vs-termination
  distinction). `normalize_advantages_masked`/`masked_mean_std`
  (implementing the P1 skeleton): Actor-side normalization mean/std
  computed using ONLY `policy_mask == 1` positions, via boolean-mask
  indexing performed BEFORE the `np.mean`/`np.std` reduction runs --
  structurally impossible for a masked-out value to enter the
  statistic (§7.2), which is what makes the required invariance test
  hold by construction rather than by coincidence.

  `src/training/trainer.py::build_training_batch` (new function,
  P4's end-to-end deliverable): wires rollout -> GAE -> masked
  normalization into one flat PPO-ready batch dict
  (`observation`/`action`/`reward`/`next_observation`/`terminated`/
  `truncated`/`value`/`next_value`/`log_prob`/`policy_mask`/
  `advantages_raw` [unmasked, full-trajectory GAE output, kept for
  diagnostics]/`advantages` [masked-normalized, Actor scope]/
  `returns` [full-trajectory GAE Critic target]), raising `ValueError`
  on an empty transition list or any non-finite value anywhere in the
  batch. `run_training` (the actual multi-epoch parameter-update loop)
  correctly remains an updated `NotImplementedError` stub -- P4's
  explicit non-goal, P5 scope.

  Added 38 new P4 tests, verified file-by-file before combining:
  `tests/training/test_gae.py` (new file, 11 collected items -- two
  independently hand-computed multi-step GAE trajectories matching
  exactly [a 2-step terminal-bootstrap case and a 3-step
  truncation-bootstrap case], a terminal-step-does-not-bootstrap test
  [changing `next_value` on a terminated step changes nothing], a
  truncated-step-DOES-bootstrap test [changing `next_value` on a
  truncated step DOES change the advantage], a mismatched-lengths
  `ValueError` guard, a no-NaN/inf sweep over 50 random steps, the
  required §7.2 test A [masked-frame invariance -- changing ONLY
  `policy_mask==0` advantage values leaves the `policy_mask==1`
  mean/std and normalized values byte-identical] plus its necessary
  non-vacuous counterpart [the statistic DOES change when
  `policy_mask==1` values themselves change], a manual-computation
  cross-check, an all-masked-out `ValueError` guard, and a
  shape-mismatch `ValueError` guard), `tests/training/test_rollout.py`
  (new file, 8 collected items -- a real 1-episode rollout against the
  actual `MergeEnvironment` with the real untrained PPO policy that
  runs to completion [terminated or truncated] on maneuver `MAN_0001`;
  rollout shape consistency; a full-episode no-NaN/inf sweep;
  multi-maneuver `collect_rollout` concatenation across 2 episodes;
  **the §7.1 pre-step MERGE `policy_mask` regression test** [see
  below]; **the §7.2 test D terminal-reward-propagation test** [see
  below]; an integration-level §7.2 test B [confirms masked frames'
  log_prob/action values provably never leak into the
  `policy_mask==1`-filtered subset a P5 loss computation would
  consume]; and a §11 action-mapping regression confirming
  `rollout.py` performs no second/separate action-index translation of
  its own, delegating entirely to `distribution.ACTION_INDEX_TO_BEHAVIOR`),
  and `tests/training/test_trainer.py` (new file, 3 collected items --
  a real end-to-end rollout -> GAE -> masked-normalized-batch test on
  `MAN_0001` asserting every required key/shape is present and the
  masked decision-frame advantages have approximately zero mean; an
  empty-transition-list `ValueError` guard; confirming `run_training`
  still raises `NotImplementedError`, the explicit P4 non-goal).
  `tests/training/test_imports.py` was updated in place (its previous
  stub-`NotImplementedError` checks for `rollout.collect_rollout()`/
  `gae.compute_gae()`/`gae.normalize_advantages_masked([], [])` -- no
  longer applicable now that those functions are real -- were replaced
  with real-attribute-presence assertions plus one narrower
  `run_training`-still-a-stub check; still 6 collected items, unchanged
  count from P2/P3).

  **The §7.1 pre-step MERGE `policy_mask` regression test**
  (`test_prestep_merge_policy_mask_regression`) is the single most
  important test in this Phase, guarding against the exact reversed
  post-step bug §7.1 describes: it runs a scripted-MERGE policy
  (always selects `BehaviorAction.MERGE`, backed by a real policy
  network so `log_prob`/logits stay well-defined finite numbers)
  against the real `MergeEnvironment` on maneuver `MAN_CAUSALITY`
  (`lane_chain=[527, 544]` -- a single-transition, non-chained
  maneuver, so MERGE at decision frame 0 commits immediately and every
  subsequent frame is unambiguously auto-execution for the rest of the
  episode; already confirmed by
  `tests/environment/test_merge_environment_frenet_mpc.py::
  test_single_maneuver_merge_reaches_success_frenet_mpc` to reach a
  real terminal outcome -- success/collision/offroad -- within 60
  steps of continuous MERGE). It asserts: the VERY FIRST transition
  (the decision frame where MERGE is selected) has `policy_mask == 1`,
  and every transition strictly after it (auto-execution,
  `merge_committed` already `True` going into the step) has
  `policy_mask == 0`. **PASSED.**

  **§7.2 test D** (`test_terminal_reward_propagates_to_merge_decision_frame`),
  using the same scripted-MERGE-on-`MAN_CAUSALITY` construction:
  confirms the episode reaches a real terminal reward
  (`abs(terminal_reward) >= 0.5`, i.e. a genuine +-1.0
  SUCCESS/COLLISION component, not merely a -0.01/0.0 decision-cost
  value), then runs `compute_gae` over the full transition list and
  asserts the MERGE decision frame's (index 0) GAE `return` is LARGER
  in magnitude than that frame's own tiny immediate reward, with the
  same sign as the terminal outcome -- i.e. the terminal
  SUCCESS/COLLISION reward's credit demonstrably propagates backward
  through GAE, across the intervening `policy_mask == 0`
  auto-execution frames, to the `policy_mask == 1` decision frame that
  triggered the commitment. **PASSED.**

  Ran the new tests file-by-file first (all passing individually), then
  `tests/training/ tests/policies/ tests/rewards/` together as an
  intermediate sanity check: **114 passed** in 148.94s. Confirmed no
  other `pytest` process was running (`ps aux`/`pgrep`) before
  launching the full regression suite. Launched the full suite as a
  background process and waited synchronously for the actual PID
  (confirmed via `pgrep -af "python -m pytest tests/"` to be the real
  `python` process, not the wrapping shell) to exit -- not a
  fixed-duration sleep, not a mid-run snapshot. Final result, read
  directly from the completed run's own summary line: **579 passed, 0
  failed, in 1920.04s (0:32:00)**, 100% dots, zero `F`/`E` marks, no
  other pytest process ever contending. Independently cross-checked
  via `pytest tests/ --collect-only -q` -> "579 tests collected" (557
  pre-existing + 22 net new P4 tests exactly: 11 gae + 8 rollout + 3
  trainer = 22; `test_imports.py` unchanged at 6 collected items).
  `jax.devices()` re-confirmed `[CudaDevice(id=0)]` immediately after
  the full-suite run -- GPU still working, no regression from any P4
  change (P4 added no new dependency).

  No contradiction found between PPO_PLAN.md and the current codebase
  -- no blocker. §7.1's pre-step decision-timing pattern was directly
  supported by the existing `reset()`/`step()` API exactly as P0
  predicted, with no code-level tension and no deviation required.
  §7.2's Actor-vs-Critic scope was implemented exactly as specified:
  GAE/return over the full trajectory, Actor-side normalization
  statistics excluding masked frames by construction. No actual
  multi-update training loop was implemented (`run_training` remains a
  stub) and no PPO-FIT/PPO-TUNE dataset split was created -- both
  correctly deferred/excluded per §0.1/P4's explicit non-goals. Next:
  P5 Smoke Training -- the final Phase of this P0-P5 effort.

---

## 2026-09-19 — P5 Smoke Training complete; entire P0-P5 PPO effort DONE

- Phase/Stage: P5 -- Smoke Training (final Phase of the P0-P5 effort)
- SHA: P5 completion commit `test(ppo): validate smoke training
  pipeline (P5)` on `feat/ppo-phase0-5`, immediately after `fe8edc9`
  (P4 completion) -- see `git log` for the exact SHA
- Branch: `feat/ppo-phase0-5`
- W&B run ID: two real offline runs, project `its-merge-ppo-smoke`,
  `WANDB_MODE=offline`:
  - Stage 1 (fresh): `wandb/offline-run-20260919_024059-dodmdn2z/`
  - Stage 2 (resumed via `--resume`): `wandb/offline-run-20260919_024226-bsp9erzm/`
  (not committed to git -- `wandb/` added to `.gitignore` this Phase;
  see SMOKE_TRAINING_REPORT.md §12)
- Config: `configs/ppo/ppo_smoke.yaml` (inherits `configs/ppo/ppo_base.yaml`
  verbatim: `learning_rate=3e-4`, `gamma=0.99`, `gae_lambda=0.95`,
  `clip_epsilon=0.2`, `value_coef=0.5`, `entropy_coef=0.01`,
  `max_grad_norm=0.5`, `ppo_epochs=4`; smoke-only overrides
  `num_minibatches=1`), `configs/reward/merge_reward_v0.yaml`, seed 0,
  `downstream_mode="frenet_mpc"`
- Result / notes:

  P5 implemented `src/training/trainer.py::run_update`/`run_training`
  (both previously `NotImplementedError` stubs from P1/P4), the real
  multi-epoch PPO update loop consuming P4's `build_training_batch`
  output directly: `run_update` filters every Actor-side quantity
  (`observation`/`action`/`log_prob`/`advantages`) to
  `policy_mask == 1` rows before computing policy loss/entropy/
  gradients (SS7.2), while the value-loss pass uses the FULL,
  unfiltered trajectory (Critic scope). Both `scripts/train_ppo.py`
  and `scripts/smoke_train_ppo.py` were implemented from their P1
  skeletons into real, working entry points: real config/env/
  train-state construction, real `--resume` handling (loads a
  checkpoint's policy/value params + optimizer state + JAX RNG key +
  `global_env_step`/`ppo_update_step` and CONTINUES training from
  there, per SS10), real W&B config+metric logging, and a real
  checkpoint save at the end (full SS10 contract, via
  `src/training/checkpoint.py`'s pickle-based `save_checkpoint`/
  `load_checkpoint`, unchanged from its P1 implementation but now
  exercised against real JAX pytrees for the first time).

  **Two real smoke-training stages** were run against the real
  `MergeEnvironment` (canonical-TRAIN maneuvers only, never a PPO-FIT/
  PPO-TUNE split):

  - Stage 1 (`scripts/smoke_train_ppo.py`, fresh init, no `--resume`):
    2 maneuvers (`MAN_0001`, `MAN_0002`), 2 updates, 60 max steps/
    episode. Finished in 61.1s. `global_env_step=70`,
    `ppo_update_step=2`. Update 0: `episode_return=0.885`,
    `policy_loss=-0.0917`, `value_loss=0.1455`, `entropy=1.2610`,
    `approx_kl=0.0894`, `clip_fraction=0.565`. Update 1:
    `episode_return=0.950`, `policy_loss=-0.0722`,
    `value_loss=0.0218`, `entropy=1.2470`, `approx_kl=0.0612`,
    `clip_fraction=0.200`. Checkpoint:
    `outputs/ppo_checkpoints/smoke_stage1_final.pkl` (542,445 bytes,
    md5 `eadae104a5362aa78e7f02da60168dc4`).
  - Stage 2 (same script, **separate process invocation**,
    `--resume outputs/ppo_checkpoints/smoke_stage1_final.pkl`): 4
    maneuvers (`MAN_0001..MAN_0004`), 3 more updates, 60 max steps/
    episode. Printed `Resumed: global_env_step=70,
    ppo_update_step=2` on load, then finished in 211.0s with
    `global_env_step=403`, `ppo_update_step=5` -- both counters
    genuinely CONTINUED across the resume boundary, never reset to 0.
    Update 0: `episode_return=0.725`, `policy_loss=-0.0517`,
    `value_loss=0.0104`, `entropy=1.2361`. Update 1:
    `episode_return=0.7325`, `policy_loss=-0.0222`,
    `value_loss=0.0093`, `entropy=1.2267`. Update 2:
    `episode_return=0.695`, `policy_loss=-0.0229`,
    `value_loss=0.0049`, `entropy=1.1752`. Checkpoint:
    `outputs/ppo_checkpoints/smoke_stage2_final.pkl` (542,468 bytes,
    md5 `87d66baa4080b409ca6c790254479a28` -- differs from Stage 1's,
    confirming genuinely different parameters after the additional
    real gradient updates). All metrics finite throughout both stages
    -- no NaN/inf, no crash, no OOM.

  Policy AND value network parameters were confirmed to provably
  differ (leaf-by-leaf numpy comparison over the real Flax param
  pytrees) before vs. after a real update, both via
  `tests/training/test_run_training.py::test_run_update_changes_both_policy_and_value_params`/
  `test_run_training_end_to_end_changes_parameters` and via the real
  Stage1-vs-Stage2 checkpoint md5 difference above. The full SS10
  save -> load -> resume -> additional-update cycle was verified two
  independent ways: (1) the real two-process Stage1->Stage2 CLI run
  above, and (2) `tests/training/test_checkpoint.py::
  test_full_save_load_resume_additional_update_cycle`, which
  round-trips real Flax/optax pytrees through `save_checkpoint`/
  `load_checkpoint` into a freshly-constructed training state (as a
  new process would) and confirms an ADDITIONAL real
  `run_training` update happens on top of the loaded state, with
  `ppo_update_step` continuing (not resetting) and `global_env_step`
  increasing past its checkpointed value.

  Added net **7 new P5 tests** over the P4 baseline (verified by
  checking out P4 completion commit `fe8edc9`'s own `tests/` tree
  against the current environment, which collects 580 -- not the
  "579" figure PROGRESS.md/HANDOFF.md recorded for that commit at the
  time; this is a pre-existing one-test off-by-one in that prior
  session's own count, not something this P5 session introduced or is
  retroactively correcting): `tests/training/test_checkpoint.py` (new
  file, 3 -- real-JAX-pytree checkpoint round-trip, reloaded params
  usable for a real forward pass, full save->load->resume->
  additional-update cycle), `tests/training/test_run_training.py`
  (new file, 5 -- `run_update` param-change + finite-metrics check,
  `run_update` raises on an all-`policy_mask==0` batch,
  `run_training` end-to-end param-change check, `run_training`
  continues nonzero starting step counters, action-ratio metrics
  reflect only `policy_mask==1` rows), `tests/training/test_trainer.py`
  (net 0 -- `test_run_training_still_raises_not_implemented` replaced
  with `test_run_training_is_real_not_a_stub`), `tests/training/
  test_imports.py` (-1 -- the now-obsolete `run_training`-stub check
  removed). 580 (P4 actual) + 7 = 587.

  Confirmed no other `pytest` process was running (`ps aux`/`pgrep`)
  before launching the full regression suite. Launched it as a
  background process and watched a `Monitor` until-loop poll the real
  PID (495630) to actual exit -- not a fixed-duration sleep, not a
  mid-run snapshot. Final result, read directly from the completed
  run's own summary line: **587 passed, 0 failed, 80 warnings (all
  `optax.global_norm`-deprecation warnings, not failures), in
  1948.37s (0:32:28)**, 100% dots, zero `F`/`E` marks, no other
  pytest process ever contending. Independently cross-checked via
  `pytest tests/ --collect-only -q` -> "587 tests collected".
  `jax.devices()` re-confirmed `[CudaDevice(id=0)]` immediately after.

  `git diff --stat -- src/environment/ src/planning/ src/control/
  src/scenarios/` confirmed EMPTY -- zero frozen Phase 1-3 files
  touched, across the ENTIRE P0-P5 effort including this final Phase.
  No PPO-FIT/PPO-TUNE dataset split was ever created; no reward
  weight/term change; no hyperparameter/network tuning; no W&B sweep;
  no canonical VAL evaluation; no FSM-vs-PPO comparison -- all
  correctly out of scope per §0.1/P5's explicit non-goals and
  confirmed absent from this Phase's diff.

  `wandb/` was added to `.gitignore` this Phase (offline-run
  directories are local run artifacts, not source -- see
  SMOKE_TRAINING_REPORT.md §12); the two real run directories
  referenced above remain on local disk but are not tracked in git.

  **This is the final entry for the P0-P5 PPO effort.** All items in
  PPO_PLAN.md §12's completion checklist hold. State is left as
  **P5 COMPLETE — WAITING FOR USER TUNING**. See
  docs/ppo/SMOKE_TRAINING_REPORT.md for the full report and
  docs/ppo/HANDOFF.md's NEXT OWNER ACTION for what happens next (a
  user decision, not a queued automated action).

## 2026-09-19 — Pre-P6 correctness/instrumentation hardening pass complete

  A follow-up effort on a new branch (`feat/ppo-pre-p6`, based on
  `main` at `c00743a` after the P0-P5 merge via PR #1) fixed 7
  correctness/instrumentation issues found in the P0-P5 implementation.
  Explicitly **not a tuning pass**: no reward weight, hyperparameter,
  network size, or dataset-split change was made anywhere in this
  phase — confirmed by `configs/ppo/ppo_smoke.yaml`/`ppo_base.yaml`
  being byte-identical to `main` and `configs/reward/merge_reward_v0.yaml`
  unchanged.

  **Fix 1 (episode-aware GAE):** `src/training/gae.py::compute_gae`
  now assumes ONE contiguous episode segment per call; a new
  `compute_gae_segmented` groups a flat multi-episode rollout by
  `episode_id` and calls `compute_gae` independently per segment. This
  closes a real bug where a later episode's backward-recursion
  `gae_running` could leak into an earlier truncated/cutoff episode's
  last steps under the old single-flat-call behavior.
  `src/training/trainer.py::build_training_batch` confirmed to call
  `compute_gae_segmented`, never bare `compute_gae`, over rollout
  output.

  **Fix 2 (exact categorical KL diagnostic):**
  `src/training/trainer.py::_exact_categorical_kl` computes full
  categorical KL(old || new) per policy_mask==1 row from a new
  `Transition.old_logits` field (populated only for policy_mask==1
  rows, zero-vector sentinel otherwise). Confirmed diagnostic-only by
  direct inspection of `_policy_loss_fn`: `total = policy_loss -
  entropy_coef * entropy` never includes exact_kl; it is returned only
  in the `info` dict for logging.

  **Fix 3 (PPO update metric aggregation):** `run_update` now
  accumulates every `ppo/*`/grad-norm metric (mean, and for grad-norms
  also max) across the full `ppo_epochs x num_minibatches` sweep via
  explicit `_mean`/`_max`-suffixed keys, fixing a bug where the
  previous version silently kept only the last minibatch/epoch's
  value.

  **Fix 4 (value_coef resolution):** confirmed `value_coef` is kept
  (not removed) in `PPOHyperparameters`, documented consistently as
  having no effect on this architecture's training dynamics (separate
  Actor/Critic `TrainState`s, never combined into one shared gradient)
  across `src/training/config.py`, `src/training/trainer.py::run_update`'s
  docstring, and `src/tracking/wandb_logger.py`'s config-key comment —
  no contradiction found. `value_coef=0.5` unchanged from P0-P5,
  matches PPO_PLAN.md §6.

  **Fix 5 (NumPy RNG checkpointing):** `CheckpointPayload.numpy_rng_state`
  (new, `Optional[tuple]`, defaults to `None` for backward
  compatibility with older checkpoints) now carries the NumPy
  `RandomState` governing PPO minibatch shuffling; `restore_numpy_rng`
  restores it on `--resume`, closing a gap where a resumed run's
  minibatch order used to silently diverge from an uninterrupted run's
  (only the JAX PRNG key was previously persisted).

  **Fix 6 (W&B full diagnostics):** `run_training` now actually
  computes and logs several `MINIMUM_METRICS` fields that were listed
  in `src/tracking/wandb_logger.py` but never populated:
  `train/success_rate`/`collision_rate`/`offroad_rate`/`timeout_rate`
  (from the environment's own `info["termination_reason"]`, never
  re-derived), `downstream/intervention_rate` and sibling rates (from
  the environment's own cumulative `intervention_rate`/
  `planner_infeasible_count`/etc. fields via
  `_aggregate_downstream_rates`, never re-judged),
  `ppo/explained_variance` (with a finite-value guard for near-zero
  variance), `train/policy_decision_count`/`physical_step_count`, and
  `runtime/env_steps_per_sec`. New metric names added to
  `MINIMUM_METRICS`: `ppo/exact_kl_mean`, `ppo/exact_kl_max`,
  `ppo/policy_loss_mean`, `ppo/value_loss_mean`, `ppo/entropy_mean`,
  `ppo/approx_kl_mean`, `ppo/clip_fraction_mean`,
  `ppo/policy_grad_norm_mean`, `ppo/policy_grad_norm_max`,
  `ppo/value_grad_norm_mean`, `ppo/value_grad_norm_max`,
  `train/policy_decision_count`, `train/physical_step_count`.

  **Fix 7 (reward component logging correctness):** `reward/terminal`
  in `run_training` now sums every step where `t.terminated OR
  t.truncated` is True (previously only `t.terminated`), so a
  TRUNCATION_HORIZON (-0.5) episode-end is correctly counted as a
  terminal-component event instead of silently leaking into
  `reward/decision_cost`. An artificial `rollout_cutoff` (Fix 1) is
  neither `terminated` nor `truncated`, so it is correctly excluded
  from `reward/terminal` without a special case.

  **Real smoke-training re-run required and performed:** the original
  P0-P5 smoke-training evidence
  (`outputs/ppo_checkpoints/smoke_stage1_final.pkl`/
  `smoke_stage2_final.pkl`,
  `wandb/offline-run-20260919_024059-dodmdn2z`/`...-bsp9erzm`) was
  found to predate every fix in this phase (W&B config recorded
  `git_sha=fe8edc9`, the P4 completion commit) and its logged metrics
  confirmed this directly -- no `success_rate`/`exact_kl`/
  `policy_decision_count`/`explained_variance`/etc. fields were
  present in either run's logged metric dicts, only the pre-existing
  `train/*`/`reward/*`/`ppo/*` (bare names)/`action/*` set. That
  evidence was therefore insufficient on its own, so this phase re-ran
  real fresh + resume smoke training against the current code
  (`WANDB_MODE=offline`, `--max-maneuvers 2 --max-episode-steps 60`).

  - Fresh run (`--num-updates 2`, `--checkpoint-path
    outputs/ppo_checkpoints/pre_p6_smoke.pkl`): finished in 61.4s.
    `global_env_step=70`, `ppo_update_step=2`. Update 0:
    `episode_return=0.885`, `success_rate=1.0`, `collision_rate=0.0`,
    `offroad_rate=0.0`, `policy_decision_count=23`,
    `physical_step_count=35`, `env_steps_per_sec=1.432`,
    `intervention_rate=0.350`, `policy_loss_mean=-0.0586`,
    `value_loss_mean=0.4194`, `entropy_mean=1.2427`,
    `approx_kl_mean=0.0434`, `exact_kl_mean=0.0172`,
    `exact_kl_max=0.0774`, `explained_variance=-1.746`. Update 1:
    `episode_return=0.95`, `success_rate=1.0`,
    `policy_decision_count=10`, `env_steps_per_sec=1.653`,
    `policy_loss_mean=-0.0396`, `value_loss_mean=0.0257`,
    `entropy_mean=1.2554`, `approx_kl_mean=0.0320`,
    `exact_kl_mean=0.0027`, `exact_kl_max=0.0194`,
    `explained_variance=-1.264`. All values finite. Checkpoint:
    `outputs/ppo_checkpoints/pre_p6_smoke.pkl` (545,023 bytes).
  - Resume run (separate process, `--resume
    outputs/ppo_checkpoints/pre_p6_smoke.pkl`, `--num-updates 1`):
    printed `Resumed: global_env_step=70, ppo_update_step=2` on load
    -- exactly matching the fresh run's final state, confirming
    JAX+NumPy RNG/step-counter resume genuinely continues rather than
    resets -- then finished in 34.5s with `global_env_step=103,
    ppo_update_step=3`. Update 0: `episode_return=0.98`,
    `success_rate=1.0`, `policy_decision_count=4`,
    `env_steps_per_sec=1.337`, `intervention_rate=0.233`,
    `collision_blocked_rate=0.233`, `policy_loss_mean=-0.0625`,
    `value_loss_mean=0.0240`, `entropy_mean=1.1692`,
    `approx_kl_mean=0.00866`, `exact_kl_mean=0.00623`,
    `exact_kl_max=0.0358`, `explained_variance=-1.444`. All values
    finite. Checkpoint: `outputs/ppo_checkpoints/pre_p6_smoke_resumed.pkl`
    (545,022 bytes). Both runs exited code 0.

  (Housekeeping note: an earlier attempt at this same re-run within
  this session hit a GPU `FailedPreconditionError: Failed to allocate
  scratch buffer` from a leftover, still-running process holding
  ~7.3GB of GPU memory from a prior turn in this same session; that
  process was identified via `ps`/`nvidia-smi`, killed, and its
  partial/failed artifacts -- a stray checkpoint path and an empty
  `wandb/offline-run-...` directory -- removed before the real, clean
  re-run reported above.)

  Targeted test run over the 5 changed/new test files
  (`tests/training/test_gae.py`, `test_checkpoint.py`,
  `test_run_training.py`, `test_trainer.py`, new
  `test_pre_p6_hardening.py`): **65 passed**, 0 failed, 238 warnings
  (all pre-existing `optax.global_norm` deprecation warnings), in
  233.97s; `--collect-only` over the same 5 files independently
  confirms 65 collected.

  Full regression suite result (verified solo by the orchestrating
  session, no concurrent GPU-contending process): **629 passed, 0
  failed, in 2169.30s (0:36:09)**. `pytest --collect-only -q`
  independently confirms 629 tests collected, consistent with the
  reported count (baseline going into this branch was 587; +42 net
  new/extended tests across the 7 fixes).

  `git diff --stat main -- src/environment/ src/planning/
  src/control/ src/scenarios/` confirmed EMPTY -- zero frozen Phase
  1-3 files touched. No PPO-FIT/PPO-TUNE dataset split exists
  anywhere. Reward V0 values unchanged (success=+1.0,
  failure_collision=-1.0, failure_offroad=-1.0,
  truncation_horizon=-0.5, none=0.0, decision_cost -0.01/0.0) --
  only logging/breakdown changed (Fix 7). PPO hyperparameters
  unchanged (learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
  clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5, network
  [256, 64, 32] tanh, ppo_epochs=4, num_minibatches=4) -- this was
  explicitly NOT a tuning pass. `configs/ppo/ppo_tune.yaml` was found
  on disk as an untracked, code-unreferenced P6-scoped scaffold and
  deliberately left untracked/uncommitted, out of scope for this
  phase.

  **This is the final entry for the Pre-P6 hardening pass.** See
  docs/ppo/PRE_P6_REPORT.md for the full report. State is left as
  **PRE-P6 HARDENING COMPLETE — WAITING FOR USER TUNING**. See
  docs/ppo/HANDOFF.md's NEXT OWNER ACTION for what happens next (a
  user decision, not a queued automated action).
