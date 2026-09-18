# PPO Progress Tracker

Live state, updated at the end of every Phase/Stage. See
[PPO_PLAN.md](PPO_PLAN.md) for the durable plan and
[HANDOFF.md](HANDOFF.md) for session-resume instructions.

---

## Current Phase

**P1 COMPLETE** (PPO Foundation)

## Current Stage

P0 and P1 both complete. P0 confirmed the frozen baseline
(461 passed, 0 failed) and was audit-only (zero source changes). P1
added the full PPO scaffolding directory/module structure per
PPO_PLAN.md §4 (every file listed there now exists, including the
structural skeletons for `src/rewards/merge_reward.py`,
`src/rewards/reward_wrapper.py`, `src/policies/ppo/{networks,
distribution,policy,loss,state}.py`, `src/training/{rollout,gae,
trainer}.py`, `src/tracking/wandb_logger.py` — all P1-scoped as
raising-`NotImplementedError` stubs with fixed signatures/constants,
since their real logic is P2/P3/P4/P5's job), plus YAML config loading
(PPO + reward configs), seed handling, and a checkpoint save/load
skeleton, plus the two `scripts/` entry points (`train_ppo.py` now
also accepts `--resume`, echoed but not yet acted on). All 28 new P1
tests pass (import checks across every new package, config-load
checks, seed determinism, checkpoint round-trip, and the §11
action-index → `BehaviorAction` mapping regression test). Both
entry-point scripts run successfully end-to-end at their P1 scope
(config + seed [+ smoke maneuver-subset] resolution only — no real
training loop, as scoped). The full existing regression suite passes
with **zero failures**: **489 passed, 0 failed, 0 skipped** (461
pre-existing Phase 1-3 tests + 28 new P1 tests), confirmed by a clean
solo run after clearing an unrelated GPU-memory-contention artifact
(see "Known Issues" below). Ready to begin **P2 — Reward V0 + W&B
Foundation**.

## SHA / Branch

- base SHA (plan creation): `c5d2c1d2ea197cd247b3bc2e4f127b3c36d1a990`
- plan-docs commit SHA: `aa8cf5b67d51c7bb5005fde0106d773cb8193452`
- P0 completion commit SHA: `bcf2b4c` (`chore(ppo): audit frozen
  training baseline`)
- P1 completion commit SHA: (recorded in `git log` on this branch
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
- Real reward/PPO-algorithm/rollout logic is not implemented yet
  (P2/P3/P4 scope) — the P1 tests above confirm those modules'
  structural skeletons import correctly and raise `NotImplementedError`
  rather than a silent/fake result

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

V0. Values are codified in `configs/reward/merge_reward_v0.yaml`
(spec: [PPO_PLAN.md § 5](PPO_PLAN.md#5-reward-v0-fixed-through-p5)),
loadable via `src.training.config.load_reward_config`. The actual
reward-computation code (`src/rewards/merge_reward.py`,
`src/rewards/reward_wrapper.py`) is still a P1 structural stub raising
`NotImplementedError` — real logic lands in P2.

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

None yet (P2 scope).

## Checkpoint Path

None yet — no real training run has produced one. P1 implemented and
tested the `CheckpointPayload` contract and `save_checkpoint`/
`load_checkpoint` (pickle-based skeleton, round-trip verified in
`tests/training/test_config.py::test_checkpoint_save_load_roundtrip`);
the full JAX-pytree save/load/resume cycle against real PPO train
state is exercised end-to-end in P5.

## Last Command

```
pytest tests/ -q                     # P1: 489 passed, 0 failed, 1867.61s (0:31:07)
pytest tests/policies/ tests/rewards/ tests/training/ -v   # new P1 tests alone: 28 passed, 1.08s
PYTHONPATH=. python scripts/train_ppo.py
PYTHONPATH=. python scripts/smoke_train_ppo.py --max-maneuvers 2
```

## Known Issues

No contradiction found between PPO_PLAN.md and the current frozen
codebase during P0 or P1 — the `info_before`/pre-step pattern required
by §7.1 is directly supported by the existing `reset()`/`step()` API
(both return `info` reflecting `merge_committed` state as of that
call), and an equivalent pattern is already used in
`src/environment/full_split_evaluator.py`.

**Lesson learned (process, not a code issue):** running two `pytest`
processes against this repo concurrently causes spurious GPU-contention
failures (observed: 62 failures with `FailedPreconditionError: Failed
to allocate scratch buffer for device 0` and TF `TypeSpec`
dataset-loading errors) that look like real regressions but are not —
they disappear when the suite is re-run solo. This is because the
JAX/TF GPU-backed tests in this suite are not designed to share the
single RTX 4060 across two simultaneous pytest processes. **Future
phases (P2–P5) must never run `pytest` concurrently with another
pytest process** (or any other GPU-heavy process) against this repo —
always confirm no other `pytest` process is running before starting a
full-suite run, and if a stale one is found, wait for it to exit (or
kill it deliberately) rather than trusting a concurrent run's failure
count.

## Next Exact Action

**Begin P2 Reward V0 + W&B Foundation.** Per
[PPO_PLAN.md §0.1/P2](PPO_PLAN.md#p2--reward-v0--wb-foundation):

1. Implement `src/rewards/merge_reward.py`: maps the frozen
   `MergeEnvironment`'s own `terminated`/`truncated`/
   `info["termination_reason"]` to the Reward V0 terminal-outcome table
   already codified in `configs/reward/merge_reward_v0.yaml` (loaded via
   `src.training.config.load_reward_config`), plus the pre-step
   decision-cost term (`-0.01` real decision step / `0.0` auto
   -execution step, decided from `policy_mask`/`is_policy_step`
   computed BEFORE `env.step()`, per §7.1 — never re-derive
   success/collision/offroad/timeout; §5.1 explicitly forbids
   re-implementing any of those detectors)
2. Implement `src/rewards/reward_wrapper.py`: the actual call-site glue
   that a rollout loop uses each step to get `(reward, decision_cost,
   terminal_component)` from one step's `(terminated, truncated, info,
   is_policy_step)`
3. Implement `src/tracking/wandb_logger.py`: supports both online and
   `WANDB_MODE=offline` modes; logs the config fields from §8
   (`git_sha`, `reward_version`, `seed`, `learning_rate`, `gamma`,
   `gae_lambda`, `clip_epsilon`, `entropy_coef`, `value_coef`,
   `batch_size`, `ppo_epochs`, `network_layers`)
4. Required tests (`tests/rewards/`): SUCCESS=+1, COLLISION=-1,
   OFFROAD=-1, TIMEOUT/TRUNCATION_HORIZON=-0.5, NONE=0; real decision
   step=-0.01, auto-committed step=0; total reward = terminal +
   decision cost, correctly summed; no NaN/inf in any reward path; a
   W&B smoke run (offline mode is fine) produces logged config + at
   least one metric point
5. Re-run the full existing regression suite to confirm nothing broke
6. Update PROGRESS.md/HANDOFF.md, append an EXPERIMENT_LOG.md entry,
   commit `feat(ppo): add Reward V0 and W&B tracking foundation` on
   `feat/ppo-phase0-5` (one commit for all of P2, per this effort's
   one-commit-per-completed-Phase convention)
