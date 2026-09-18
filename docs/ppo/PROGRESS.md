# PPO Progress Tracker

Live state, updated at the end of every Phase/Stage. See
[PPO_PLAN.md](PPO_PLAN.md) for the durable plan and
[HANDOFF.md](HANDOFF.md) for session-resume instructions.

---

## Current Phase

**P2 COMPLETE** (Reward V0 + W&B Foundation)

## Current Stage

P0, P1, and P2 are all complete. P0 confirmed the frozen baseline
(461 passed, 0 failed) and was audit-only (zero source changes). P1
added the full PPO scaffolding directory/module structure per
PPO_PLAN.md §4. P2 implemented Reward V0 (`src/rewards/merge_reward.py`,
`src/rewards/reward_wrapper.py`) and the W&B logging foundation
(`src/tracking/wandb_logger.py`), replacing their P1
`NotImplementedError` skeletons with real logic, per PPO_PLAN.md
§0.1/P2, §5, §5.1, §8.

`merge_reward.compute_reward(reward_config, termination_reason,
is_policy_step, info=None)` performs a pure fixed-table lookup: the
terminal component comes from `reward_config.terminal` keyed by the
`TerminationReason.value` string the frozen `MergeEnvironment` already
computed (`info["termination_reason"]`, or `None` before any
termination result exists, treated as `"none"`), and the decision-cost
component comes from `reward_config.decision_cost` keyed purely by the
caller-supplied pre-step `is_policy_step` flag. It never imports
anything from `src.environment` or `waymax` (verified both by a
code-level test asserting those strings are absent from the module
source, and by the function signature taking `termination_reason`/
`is_policy_step` as required inputs rather than any observation/state
it could inspect itself) — satisfying §5.1's source-of-truth rule.
`MergeRewardWrapper` (in `reward_wrapper.py`) is a thin stateful
wrapper used by a rollout loop: it calls `compute_reward`, records the
terminal/decision-cost component breakdown and running per-episode
sums under the exact `reward/terminal`, `reward/decision_cost`,
`reward/total` keys §8 requires (via `episode_sums()`), and asserts
its own component sum agrees with `compute_reward`'s total (a
same-Phase cross-check, not a second reward path).

`WandbLogger` (`src/tracking/wandb_logger.py`) wraps `wandb.init`/
`wandb.config.update`/`wandb.log`, supporting both online and
`WANDB_MODE=offline` modes via the `mode` constructor arg (or the
`WANDB_MODE` env var, standard `wandb` behavior). `log_config`
validates that all of §8's `REQUIRED_CONFIG_KEYS` are present (all
knowable up front — git SHA, reward version, seed, fixed
hyperparameters, network layer sizes — so a missing one is a caller
bug, not a not-yet-available metric) and raises `ValueError`
otherwise. `log_metrics` accepts **any** metric name (not just
`MINIMUM_METRICS`), by design, so P3/P4/P5 can add `train/*`, `ppo/*`,
`action/*`, `downstream/*`, `safety/*` metrics without changing this
module. `run_dir`/`run_id` properties and a context-manager
(`__enter__`/`__exit__` calling `finish()`) are provided for rollout-
loop convenience.

`configs/reward/merge_reward_v0.yaml` was already correct from P1
(values only) and required no changes — confirmed line-by-line against
the fixed table in this update.

35 new P2 tests were added (per-file counts independently confirmed via
`pytest --collect-only -q` on each file): `tests/rewards/test_merge_reward.py`
(32 collected test items — parametrization accounts for most of this:
the 5-way terminal-outcome table incl. `None`→"none", the
`TerminationReason.NONE` string-lookup edge case, real-decision-step
=-0.01/auto-step=0.0 decision cost, 8 total-reward
terminal+decision-cost sum combinations, a parametrized no-NaN/inf
sweep across every reason × both `is_policy_step` values (12 combos),
an unrecognized-reason `ValueError` guard, the code-level
no-`src.environment`/no-`waymax`-import + required-parameter-name
check, wrapper-vs-`compute_reward` agreement, and wrapper episode-sum
accumulation/reset), `tests/tracking/test_wandb_logger.py` (4 — module
constants, an offline smoke run that logs a config dict + a metric
point and asserts real files exist on disk under the offline run
directory, `log_config` rejecting a config missing a required key,
`log_metrics` accepting an arbitrary metric name not in
`MINIMUM_METRICS`). `tests/rewards/test_imports.py` (4 collected,
unchanged count from P1) and `tests/training/test_imports.py` (6
collected, was 7 in P1 — one redundant stub-`NotImplementedError`
check for the now-real `wandb_logger.WandbLogger` was consolidated into
`test_wandb_logger_module_imports`'s constant checks rather than kept
as a separate stub-only test) were both updated in place so their
`compute_reward`/`MergeRewardWrapper.compute`/`WandbLogger` assertions
reflect real, correct behavior instead of `NotImplementedError` —
`rollout.py`/`gae.py`/`trainer.py` (P4/P5 scope) remain untouched
`NotImplementedError` stubs and their tests still assert that. Net:
32 + 4 + (4-4) + (6-7) = 35 new tests over the P1 baseline of 489,
giving 524 — independently confirmed by `pytest tests/ --collect-only -q`
→ "524 tests collected" exactly.

A standalone offline W&B smoke run was also run directly (outside
pytest, `WANDB_MODE=offline`), logging the full §8 config-key set plus
`reward/terminal`/`reward/decision_cost`/`reward/total`, exit code 0,
with real files written under
`<run>/wandb/offline-run-<timestamp>-<id>/` (`run-<id>.wandb`,
`run-<id>.wandb.syncstate`, `logs/debug.log`,
`logs/debug-internal.log`, `files/requirements.txt`, etc.) — see
`docs/ppo/EXPERIMENT_LOG.md` for the exact run directory and
`wandb sync` command it printed.

The full existing regression suite passes with **zero failures**:
**524 passed, 0 failed, 0 skipped** in 1769.33s (0:29:29) (489
pre-existing Phase 1-3 + P1 tests + 35 new P2 tests = 524, matching
exactly; independently confirmed via `pytest tests/ --collect-only -q`
→ "524 tests collected"). No other `pytest` process was running
before or during this run (checked via `ps aux | grep pytest` before
starting, per the P1 lesson-learned note below), and the real process
exit (not a mid-run snapshot) was waited for and its final summary
line read directly from the log file. Ready to begin **P3 — Discrete
PPO Core**.

## SHA / Branch

- base SHA (plan creation): `c5d2c1d2ea197cd247b3bc2e4f127b3c36d1a990`
- plan-docs commit SHA: `aa8cf5b67d51c7bb5005fde0106d773cb8193452`
- P0 completion commit SHA: `bcf2b4c` (`chore(ppo): audit frozen
  training baseline`)
- P1 completion commit SHA: `2f865b6` (`feat(ppo): add PPO foundation
  scaffolding (P1)`)
- P2 completion commit SHA: (recorded in `git log` on this branch
  immediately after this update, per the one-commit-per-Phase rule)
- branch: `feat/ppo-phase0-5`

## Completed Tasks

- [x] Created `docs/ppo/PPO_PLAN.md`, `PROGRESS.md`, `HANDOFF.md`,
      `EXPERIMENT_LOG.md`, approved and committed on `feat/ppo-phase0-5`
      (`aa8cf5b`)
- [x] **P0 Baseline Audit**:
  - [x] `git status`/`git branch`/`git log` reviewed — clean tree,
        already on `feat/ppo-phase0-5`
  - [x] Recorded JAX/jaxlib/CUDA/NumPy/Waymax/Python versions (see
        below) — GPU confirmed visible via `jax.devices()`
  - [x] Reviewed `requirements.txt`
  - [x] Reviewed dataset manifests (`data/manifests/*.csv`) — canonical
        TRAIN split has 168 maneuvers
        (`data/manifests/phase2_dataset_split.csv`)
  - [x] Confirmed `OBSERVATION_DIM == 14` and exact field order in
        `src/environment/observation_builder.py:68-85`
  - [x] Confirmed `BehaviorAction` semantics
        (`KEEP=0, FOLLOW=1, MERGE=2, STOP=3`) in
        `src/environment/behavior_action.py:59-70`
  - [x] Confirmed `TerminationReason` enum has exactly the 5 values
        PPO_PLAN.md §0.1/P2 expects
        (`src/environment/termination.py:48-53`)
  - [x] Confirmed MERGE commitment behavior
        (`src/environment/decision_state.py`) — `DecisionState.advance`
        returns MERGE as the effective action on the very step MERGE
        is selected, then locks into `MERGE_COMMITTED`; `info["merge_committed"]`
        as returned by `reset()`/`step()` reflects the state as of
        that call, confirming the `info_before`-from-previous-call
        pattern in PPO_PLAN.md §7.1 is directly supported by the
        existing API (no code change needed) — this exact pattern is
        already used by `src/environment/full_split_evaluator.py`'s
        `run_episode` (`was_committed_before = info.get("merge_committed", False)`
        read BEFORE calling `policy.decide`/`env.step`)
  - [x] Confirmed `downstream_mode="frenet_mpc"` path exists
        (`MergeEnvironment.__init__`) and is exercised by
        `tests/environment/test_merge_environment_frenet_mpc.py`
  - [x] Confirmed intervention diagnostics intact
        (`info["intervention_rate"]`, `downstream_failure_count`,
        `planner_infeasible_count`, `collision_blocked_count`,
        `controller_failure_count`, `invalid_reference_count` — all
        present in `_build_info`)
  - [x] Ran full existing `pytest` suite in the background and waited
        for real completion (not a partial/estimated read) — **461
        passed, 0 failed** in 1798.33s (0:29:58 wall clock; real
        Waymax/WOMD-backed tests; `[exited with code 0]`). An
        intermediate `tail -60` snapshot taken mid-run had briefly
        shown `F` marks around the 31% mark (module 2 in progress);
        the completed run's own final summary line shows 100% dots
        and `461 passed` with zero failures, confirming that
        intermediate snapshot was a stale/truncated read of a
        still-running module, not a real failure — the actual
        terminal state has no failing tests.
  - [x] Reproduced a short deterministic rollout (single maneuver
        `MAN_0001`, fixed KEEP-action policy, frenet_mpc mode, 40
        steps): byte-identical observation trace across 2 independent
        runs (`max_abs_obs_diff_between_two_runs = 0.0000000000`) —
        determinism confirmed. Episode did not terminate within 40
        steps under a KEEP-only script (`termination_reason='none'`,
        `steps_elapsed=40`); `downstream_status='COLLISION_BLOCKED'`
        with `collision_blocked_count=2`,
        `intervention_rate=0.05` — intervention-diagnostics fields
        are present and populated as expected, confirming the contract
        is intact (this is the intervention *diagnostic signal* firing
        correctly under a deliberately non-merging fixed script, not a
        test failure).
  - [x] Took a runtime/throughput measurement (separate timed rollout,
        5-step warmup then 60 timed steps, `frenet_mpc` mode): 60
        steps in 36.69s → **~1.64 steps/sec**. Note: this measurement
        subprocess logged `Cannot dlopen some GPU libraries ... Skipping
        registering GPU devices` and ran on CPU fallback for that
        invocation specifically (library-path/env issue local to that
        one script run, not a code regression — the interactive
        version-check in this same audit confirmed
        `jax.devices() == [CudaDevice(id=0)]` and the full pytest GPU
        run above executed correctly). Throughput is dominated by the
        LTV-MPC's `scipy.optimize.minimize(L-BFGS-B)` solve per step
        regardless of backend, consistent with
        `configs/phase3_downstream.yaml`'s `mpc.max_iterations: 100`
        — not a bug. **Implication carried into P5**: Smoke Training
        step budgets must stay very small (single-digit to
        low-double-digit maneuvers, minimal update counts) to keep
        wall-clock time reasonable, exactly as PPO_PLAN.md §0.1/P5
        already requires. A future phase should re-run this throughput
        check with GPU dlopen confirmed working if a tighter number is
        needed.
  - [x] Installed `wandb==0.30.0` via plain `pip install wandb` (no
        `--upgrade` flag, no touch to jax/jaxlib/numpy). `flax==0.10.7`,
        `optax==0.2.8`, and `orbax-checkpoint==0.11.39` were **already
        present** in the `its-merge` conda env — compatible with the
        installed `jax==0.6.2`/`jaxlib==0.6.2`, no install needed for
        those three.
  - [x] Re-verified `jax.devices()` still reports the GPU after the
        wandb install (`[CudaDevice(id=0)]`) — no regression.
- [x] **P1 PPO Foundation**:
  - [x] Created every directory/module PPO_PLAN.md §4 lists, with
        `__init__.py` package markers: `src/rewards/` (+
        `merge_reward.py`, `reward_wrapper.py`), `src/policies/` +
        `src/policies/ppo/` (+ `networks.py`, `distribution.py`,
        `policy.py`, `loss.py`, `state.py`), `src/training/` (+
        `config.py`, `seeding.py`, `checkpoint.py`, `rollout.py`,
        `gae.py`, `trainer.py`), `src/tracking/` (+
        `wandb_logger.py`), `tests/rewards/`, `tests/policies/`,
        `tests/training/`. Modules whose real logic is out of P1
        scope (`merge_reward.py`, `reward_wrapper.py`, all of
        `src/policies/ppo/*.py` except the fixed §11 action-index
        mapping, `rollout.py`, `gae.py`, `trainer.py`,
        `wandb_logger.py`) are structural skeletons: fixed
        signatures/dataclasses/constants (e.g. `Transition`'s required
        fields, the §11 action-index mapping, the §8 metric/config-key
        name lists, the §6 layer sizes) whose bodies raise
        `NotImplementedError` rather than silently no-op, so a caller
        gets a clear "not yet implemented, lands in P<n>" signal
        instead of a stub that looks like a real (but wrong) answer.
  - [x] Created `configs/reward/merge_reward_v0.yaml` (Reward V0
        terminal-outcome table + decision-cost values, matching
        PPO_PLAN.md §5/§0.1-P2 exactly — values only, no reward CODE
        yet; that's P2)
  - [x] Created `configs/ppo/ppo_base.yaml` and
        `configs/ppo/ppo_smoke.yaml` (network architecture +
        hyperparameters from PPO_PLAN.md §6, verbatim, not tuned)
  - [x] Implemented `src/training/config.py`: `load_ppo_config`/
        `load_reward_config`, following this repo's existing
        `yaml.safe_load` + frozen-dataclass convention (matches
        `load_mpc_config`/`load_planner_config`/`load_dataset_config`)
  - [x] Implemented `src/training/seeding.py`: `make_seed_state`
        (JAX PRNGKey + NumPy RandomState from one integer seed),
        `split_key` wrapper
  - [x] Implemented `src/training/checkpoint.py`: `CheckpointPayload`
        dataclass carrying every field PPO_PLAN.md §10 requires
        (policy/value params, optimizer state, JAX RNG key, global env
        step, PPO update step, seed, config snapshot, reward version,
        git SHA, optional extra), plus `get_git_sha`,
        `save_checkpoint`/`load_checkpoint` (P1: plain pickle
        skeleton — full JAX-pytree save/load verified end-to-end in
        P5, see checkpoint.py's docstring for why pickle is
        sufficient now and what P5 will actually exercise)
  - [x] Created `scripts/train_ppo.py` and `scripts/smoke_train_ppo.py`
        entry-point skeletons: load configs, resolve seed, (smoke only)
        deterministically select a small canonical-TRAIN maneuver_id
        subset via `--max-maneuvers`/`--maneuver-ids`/`--seed` — no
        real rollout/training loop yet (explicit P1 non-goal); both
        scripts run successfully end-to-end at this scope
  - [x] Wrote 28 new P1-specific tests across 4 files:
        `tests/policies/test_imports.py` (7 — module imports, the
        fixed action-index→`BehaviorAction` mapping regression per
        PPO_PLAN.md §11, and stub functions raising
        `NotImplementedError`), `tests/rewards/test_imports.py`
        (4 — module imports + reward-stub `NotImplementedError`
        checks against the real V0 config), `tests/training/test_imports.py`
        (7 — rollout/GAE/trainer/wandb_logger module imports,
        `Transition` dataclass field contract, stub
        `NotImplementedError` checks), `tests/training/test_config.py`
        (10 — PPO/reward config loading + field values, config
        cross-reference (`ppo_config.reward_config_path` resolves),
        finite-value check, seed determinism (same seed → identical
        `jax_key`/RNG stream), different seeds differ, `split_key`
        shape, non-empty git SHA, checkpoint save→load round-trip
        preserves every field)
  - [x] Ran the 28 new P1 tests alone: **28 passed** in 1.08s
  - [x] Ran both scripts directly (`PYTHONPATH=. python scripts/train_ppo.py`,
        `... scripts/smoke_train_ppo.py --max-maneuvers 2`): both
        printed the expected resolved config/seed/maneuver-subset and
        exited 0
  - [x] Re-ran the FULL existing regression suite (Phase 1-3 tests +
        new P1 tests together) and waited for the real process to
        exit (not a partial/estimated read): **489 passed**, 0 failed,
        0 errors, 1867.61s (0:31:07), exit code 0 — confirms P1's new
        scaffolding did not break anything (461 pre-existing + 28 new
        = 489) and the dependency install from P0 remains sound. A
        first attempt at this rerun showed 62 failures with
        `FailedPreconditionError: Failed to allocate scratch buffer
        for device 0` / TF `TypeSpec` dataset-loading errors;
        investigation found a second, stale `pytest tests/ -q`
        process left running concurrently from an earlier session,
        contending for the same GPU with this rerun — not a real
        regression, and no source file was touched while diagnosing
        it. Waited for the stale process to exit on its own, then
        re-ran the full suite solo; the clean, uncontended result is
        the 489-passed figure above, verified directly against the
        log file's final summary line and `EXIT_CODE=0` marker.
- [x] **P2 Reward V0 + W&B Foundation**:
  - [x] Implemented `src/rewards/merge_reward.py::compute_reward`:
        fixed terminal-outcome table lookup (keyed by the frozen
        `MergeEnvironment`'s own `info["termination_reason"]` string,
        `None` treated as `"none"`) + fixed decision-cost lookup
        (keyed by caller-supplied pre-step `is_policy_step`); rejects
        any unrecognized `termination_reason` string via `ValueError`;
        asserts the returned total is finite; contains no import of
        `src.environment` or `waymax` (§5.1 source-of-truth rule)
  - [x] Implemented `src/rewards/reward_wrapper.py::MergeRewardWrapper`:
        thin stateful wrapper calling `compute_reward`, exposing
        `last_terminal_component`/`last_decision_cost_component`/
        `last_total`, `episode_sums()` (returns
        `reward/terminal`/`reward/decision_cost`/`reward/total` per
        §8), and `reset()` for per-episode accumulation; cross-checks
        its own component sum against `compute_reward`'s total
  - [x] Implemented `src/tracking/wandb_logger.py::WandbLogger`: real
        `wandb.init`/`wandb.config.update`/`wandb.log` wiring,
        online-or-offline via `mode`/`WANDB_MODE`; `log_config`
        validates all §8 `REQUIRED_CONFIG_KEYS` present
        (`ValueError` if not); `log_metrics` accepts any metric name;
        `run_dir`/`run_id` properties; context-manager support
  - [x] Confirmed `configs/reward/merge_reward_v0.yaml` already
        matches the Reward V0 spec exactly (P1 had already created it
        correctly with real values) — no change needed
  - [x] Confirmed `wandb==0.30.0` (installed in P0) imports and works
        without reinstall
  - [x] Wrote 35 new P2 tests over the P1 baseline: net effect across
        `tests/rewards/test_merge_reward.py` (new file, 32 collected
        items — most from parametrization), `tests/tracking/test_wandb_logger.py`
        (new package + 4 tests), `tests/rewards/test_imports.py`
        (updated in place, 4 collected — unchanged count, contents
        changed from `NotImplementedError` stub checks to real-behavior
        assertions), `tests/training/test_imports.py` (updated in
        place, 6 collected — was 7 in P1, net -1 since the
        `WandbLogger.__init__`-raises-`NotImplementedError` stub check
        was removed and folded into `test_wandb_logger_module_imports`'s
        constant checks). Net: 32 + 4 + 0 + (-1) = 35 new tests over
        the P1 baseline of 489, giving 524 exactly.
  - [x] Ran the new P2 tests together with all pre-existing P1 tests
        (`tests/rewards/ tests/tracking/ tests/training/
        tests/policies/`): **63 passed** in 1.88s
  - [x] Ran a standalone offline W&B smoke run directly (outside
        pytest): logged the full §8 config-key set +
        `reward/terminal`/`reward/decision_cost`/`reward/total`, exit
        code 0, real files written to a local offline-run directory
        (see EXPERIMENT_LOG.md for the exact path)
  - [x] Confirmed no other `pytest` process was running
        (`ps aux | grep pytest`) before launching the full regression
        suite
  - [x] Ran the FULL existing regression suite (Phase 1-3 + P1 + P2
        tests together) and waited for the real process to exit (not
        a partial/estimated read): **524 passed**, 0 failed, 0
        skipped, 1769.33s (0:29:29) — independently confirmed via
        `pytest tests/ --collect-only -q` → "524 tests collected"
        (489 pre-existing + 35 new P2 = 524, matching exactly)

## Changed Files (this session)

- `docs/ppo/PROGRESS.md` (this update)
- `docs/ppo/HANDOFF.md` (companion update)
- `docs/ppo/EXPERIMENT_LOG.md` (P0 + P1 completion entries appended)
- P0: no `src/`, `configs/`, or `tests/` files changed (read-only audit)
- P1 (new files only — no existing Phase 1-3 file touched):
  - `src/rewards/__init__.py`, `src/rewards/merge_reward.py`,
    `src/rewards/reward_wrapper.py`
  - `src/policies/__init__.py`, `src/policies/ppo/__init__.py`,
    `src/policies/ppo/networks.py`, `src/policies/ppo/distribution.py`,
    `src/policies/ppo/policy.py`, `src/policies/ppo/loss.py`,
    `src/policies/ppo/state.py`
  - `src/training/__init__.py`, `src/training/config.py`,
    `src/training/seeding.py`, `src/training/checkpoint.py`,
    `src/training/rollout.py`, `src/training/gae.py`,
    `src/training/trainer.py`
  - `src/tracking/__init__.py`, `src/tracking/wandb_logger.py`
  - `configs/reward/merge_reward_v0.yaml`
  - `configs/ppo/ppo_base.yaml`, `configs/ppo/ppo_smoke.yaml`
  - `scripts/train_ppo.py` (incl. `--resume` flag),
    `scripts/smoke_train_ppo.py`
  - `tests/rewards/__init__.py`, `tests/rewards/test_imports.py`
  - `tests/policies/__init__.py`, `tests/policies/test_imports.py`
  - `tests/training/__init__.py`, `tests/training/test_config.py`,
    `tests/training/test_imports.py`
- P2 (implementation of existing P1 skeletons + new files; no
  existing Phase 1-3 file touched):
  - `src/rewards/merge_reward.py` (implemented; was a P1 stub)
  - `src/rewards/reward_wrapper.py` (implemented; was a P1 stub)
  - `src/tracking/wandb_logger.py` (implemented; was a P1 stub)
  - `configs/reward/merge_reward_v0.yaml` (unchanged — already
    correct from P1)
  - `tests/rewards/test_merge_reward.py` (new, 32 collected test items)
  - `tests/tracking/__init__.py`, `tests/tracking/test_wandb_logger.py`
    (new package + 4 tests)
  - `tests/rewards/test_imports.py` (updated: stub-`NotImplementedError`
    checks replaced with real-behavior assertions; 4 collected,
    unchanged count)
  - `tests/training/test_imports.py` (updated: the
    stub-`NotImplementedError` test for `WandbLogger.__init__` was
    removed since it is now real; docstring updated; 6 collected, was
    7 in P1)

## Current Test Results

- Existing regression suite (P0 audit run, before P1 changes):
  **461 passed**, 0 failed, 1798.33s (0:29:58)
- Full suite after P1 additions (Phase 1-3 + new P1 tests together,
  clean solo run, real process exit confirmed): **489 passed**,
  0 failed, 0 errors, 1867.61s (0:31:07), exit code 0
- New P1-specific tests alone (`tests/policies/test_imports.py`,
  `tests/rewards/test_imports.py`, `tests/training/test_imports.py`,
  `tests/training/test_config.py`): **28 passed** in 1.08s
- P0 rollout determinism check: PASSED (byte-identical observations
  across 2 runs of the same 40-step fixed KEEP-action script,
  `frenet_mpc` mode)
- P0 throughput check: ~1.64 steps/sec over 60 timed steps (that one
  subprocess ran on CPU fallback due to a GPU dlopen issue local to
  that invocation — see note above; not a regression)
- New P2 tests alone (`tests/rewards/ tests/tracking/ tests/training/
  tests/policies/`, includes all P1 + P2 tests in those dirs):
  **63 passed** in 1.88s
- Full suite after P2 additions (Phase 1-3 + P1 + P2 tests together,
  no concurrent pytest process, real process exit confirmed):
  **524 passed**, 0 failed, 0 skipped, 1769.33s (0:29:29) —
  independently confirmed via `pytest tests/ --collect-only -q` →
  "524 tests collected" (489 pre-existing + 35 new P2 = 524)
- Standalone offline W&B smoke run (outside pytest): exit code 0, real
  files written under a local `wandb/offline-run-<timestamp>-<id>/`
  directory (see EXPERIMENT_LOG.md for the exact path and contents)
- Real PPO-algorithm/rollout/GAE logic is still not implemented
  (P3/P4/P5 scope) — `src/policies/ppo/*.py`, `src/training/rollout.py`,
  `src/training/gae.py`, `src/training/trainer.py` remain P1
  structural stubs raising `NotImplementedError`, confirmed still true
  by the (updated) P1 stub tests

## Environment / Dependency Versions (recorded per PPO_PLAN.md §0.1 P0)

```
Python:            3.10.21 (conda env "its-merge")
JAX:                0.6.2
jaxlib:             0.6.2
jax-cuda12-plugin:  0.6.2
NumPy:              2.2.6
TensorFlow:         2.21.0
waymo-waymax:       0.1.0 (pinned rev a64dfec9be8576b60d9cecc94f406d9812d4a7d0
                    per requirements.txt)
flax:               0.10.7 (pre-existing, unmodified)
optax:              0.2.8  (pre-existing, unmodified)
orbax-checkpoint:   0.11.39 (pre-existing, unmodified)
wandb:              0.30.0 (newly installed this session)
GPU:                NVIDIA GeForce RTX 4060 (8 GiB), driver 590.57,
                    CUDA 13.1 runtime reported by nvidia-smi
jax.devices():      [CudaDevice(id=0)]  (confirmed both before and
                    after the wandb install)
```

`pip install --upgrade jax` was never run. No jax/jaxlib/numpy version
changed during this session.

## Current Reward Version

**V0 (implemented).** Values are codified in
`configs/reward/merge_reward_v0.yaml` (spec:
[PPO_PLAN.md § 5](PPO_PLAN.md#5-reward-v0-fixed-through-p5)), loadable
via `src.training.config.load_reward_config`. The reward-computation
code is now real as of P2: `src/rewards/merge_reward.py::compute_reward`
performs the fixed terminal-outcome + decision-cost table lookup
against the caller-supplied `termination_reason`/`is_policy_step`
(never re-deriving termination itself, per §5.1), and
`src/rewards/reward_wrapper.py::MergeRewardWrapper` is the rollout-loop
call-site glue with per-episode component accumulation
(`episode_sums()`). Fully covered by `tests/rewards/test_merge_reward.py`
(32 tests) plus the updated `tests/rewards/test_imports.py`.

## Current PPO Config

Config loader implemented in P1 (`src/training/config.py`); the actual
PPO algorithm that consumes it is still P3+ scope. Full contents of
`configs/ppo/ppo_base.yaml` (verbatim from
[PPO_PLAN.md § 6](PPO_PLAN.md#6-ppo-baseline-architecture--hyperparameters),
not tuned):

```
seed: 0

network:
  observation_dim: 14
  num_actions: 4
  hidden_sizes: [256, 64, 32]   # 14 -> 256 -> 64 -> 32 -> {4 logits | 1 value}
  activation: "tanh"            # separate Actor and Critic stacks, no shared params

hyperparameters:
  learning_rate: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
  ppo_epochs: 4
  num_minibatches: 4

rollout:
  downstream_mode: "frenet_mpc"   # PPO always outputs the 4-way BehaviorAction,
                                   # executed through the common Frenet MPC downstream

reward_config_path: "configs/reward/merge_reward_v0.yaml"

tracking:
  wandb_project: "its-merge-ppo"
  wandb_mode: "offline"   # override with WANDB_MODE env var for online runs
```

(`configs/ppo/ppo_smoke.yaml` inherits all of the above and only
overrides `tracking.wandb_project`, `hyperparameters.num_minibatches:
1`, and adds a `smoke: {max_maneuvers: 2, num_updates: 2,
max_episode_steps: 60}` block for P5 pipeline verification.)

## W&B Run ID

No real training run yet (that's P5 scope), but W&B logging itself is
implemented and verified in P2: a standalone offline smoke run
(`WANDB_MODE=offline`, project `its-merge-ppo-test`) logged the full
§8 `REQUIRED_CONFIG_KEYS` config set plus
`reward/terminal`/`reward/decision_cost`/`reward/total` metric points,
exited 0, and produced real local files under an
`offline-run-<timestamp>-<id>/` directory (`run-<id>.wandb`,
`logs/debug.log`, `logs/debug-internal.log`, `files/requirements.txt`,
etc. — see `tests/tracking/test_wandb_logger.py` for the automated
version of this same check, and `docs/ppo/EXPERIMENT_LOG.md` for the
dated entry). No online W&B run has been started (not required through
P5; `WANDB_MODE=offline` is the supported no-network-required path per
§8).

## Checkpoint Path

None yet — no real training run has produced one. P1 implemented and
tested the `CheckpointPayload` contract and `save_checkpoint`/
`load_checkpoint` (pickle-based skeleton, round-trip verified in
`tests/training/test_config.py::test_checkpoint_save_load_roundtrip`);
the full JAX-pytree save/load/resume cycle against real PPO train
state is exercised end-to-end in P5.

## Last Command

```
pytest tests/ -q                     # P2: 524 passed, 0 failed, 1769.33s (0:29:29)
pytest tests/rewards/ tests/tracking/ tests/training/ tests/policies/ -q  # new P2 + P1 tests: 63 passed, 1.88s
pytest tests/ -q                     # P1: 489 passed, 0 failed, 1867.61s (0:31:07)
pytest tests/policies/ tests/rewards/ tests/training/ -v   # new P1 tests alone: 28 passed, 1.08s
PYTHONPATH=. python scripts/train_ppo.py
PYTHONPATH=. python scripts/smoke_train_ppo.py --max-maneuvers 2
```

## Known Issues

No contradiction found between PPO_PLAN.md and the current frozen
codebase during P0, P1, or P2 — the `info_before`/pre-step pattern
required by §7.1 is directly supported by the existing
`reset()`/`step()` API (both return `info` reflecting
`merge_committed` state as of that call), and an equivalent pattern is
already used in `src/environment/full_split_evaluator.py`. P2's reward
module was implemented as a pure fixed-table lookup consuming the
environment's own `info["termination_reason"]` string with zero
imports from `src.environment`/`waymax` — no blocker, no deviation
from §5.1 required.

**Lesson learned (process, not a code issue):** running two `pytest`
processes against this repo concurrently causes spurious GPU-contention
failures (observed: 62 failures with `FailedPreconditionError: Failed
to allocate scratch buffer for device 0` and TF `TypeSpec`
dataset-loading errors) that look like real regressions but are not —
they disappear when the suite is re-run solo. This is because the
JAX/TF GPU-backed tests in this suite are not designed to share the
single RTX 4060 across two simultaneous pytest processes. **Future
phases (P3–P5) must never run `pytest` concurrently with another
pytest process** (or any other GPU-heavy process) against this repo —
always confirm no other `pytest` process is running before starting a
full-suite run, and if a stale one is found, wait for it to exit (or
kill it deliberately) rather than trusting a concurrent run's failure
count. P2's own full-suite run followed this rule from the start (no
stale process found) and completed clean on the first attempt, with
no spurious failures to diagnose.

## Next Exact Action

**Begin P3 Discrete PPO Core.** Per
[PPO_PLAN.md §0.1/P3](PPO_PLAN.md#p3--discrete-ppo-core): implement the
policy network, value network, categorical distribution, stochastic
action sampling, deterministic inference, log_prob, entropy, PPO ratio,
clipped surrogate objective, value loss, entropy regularization,
gradient clipping, and optimizer/train state in
`src/policies/ppo/{networks,distribution,policy,loss,state}.py`
(replacing their P1 `NotImplementedError` skeletons), validated in
isolation from the real environment (P3 is algorithm-only — no real
`MergeEnvironment` rollout yet, that's P4). Required tests per §0.1/P3:
14D input handling, output logits shape == 4, finite logits,
probability sum == 1, action range 0..3, deterministic inference ==
argmax, stochastic sampling, finite log_prob/entropy, correct PPO ratio
and clipping, scalar value output, finite PPO loss/gradients, optimizer
update changes parameters, same-seed reproducibility, and the §11
action-mapping regression (`0->KEEP, 1->FOLLOW, 2->MERGE, 3->STOP`).
Re-run the full regression suite, update PROGRESS.md/HANDOFF.md, append
an EXPERIMENT_LOG.md entry, and make exactly one P3 commit on
`feat/ppo-phase0-5`.
