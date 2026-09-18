# PPO Progress Tracker

Live state, updated at the end of every Phase/Stage. See
[PPO_PLAN.md](PPO_PLAN.md) for the durable plan and
[HANDOFF.md](HANDOFF.md) for session-resume instructions.

---

## Current Phase

**P0 COMPLETE**

## Current Stage

P0 Baseline Audit complete. All confirmation checks passed; existing
regression suite passes in full (461 passed, 0 failed); determinism
and throughput sanity checks passed. Zero source-code changes made in
P0 (audit-only, as scoped). Ready to begin **P1 PPO Foundation**.

## SHA / Branch

- base SHA (plan creation): `c5d2c1d2ea197cd247b3bc2e4f127b3c36d1a990`
- plan-docs commit SHA: `aa8cf5b67d51c7bb5005fde0106d773cb8193452`
- current SHA: (see P0 completion commit on this branch)
- branch: `feat/ppo-phase0-5` (already created and checked out at
  session start; carries the approved plan docs from `aa8cf5b`)

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

## Changed Files (this session)

- `docs/ppo/PROGRESS.md` (this update)
- `docs/ppo/HANDOFF.md` (companion update)
- `docs/ppo/EXPERIMENT_LOG.md` (P0 completion entry appended)
- No `src/`, `configs/`, or `tests/` files changed in P0 (read-only
  audit, per PPO_PLAN.md §0.1's explicit non-goal for this Phase)

## Current Test Results

- Existing regression suite: **461 passed**, 0 failed, 1798.33s
  (0:29:58) (`pytest tests/ -q`, run inside the `its-merge` conda env,
  waited for real process exit and confirmed `[exited with code 0]`)
- P0 rollout determinism check: PASSED (byte-identical observations
  across 2 runs of the same 40-step fixed KEEP-action script,
  `frenet_mpc` mode)
- P0 throughput check: ~1.64 steps/sec over 60 timed steps (that one
  subprocess ran on CPU fallback due to a GPU dlopen issue local to
  that invocation — see note above; not a regression)
- No PPO-specific tests yet (P1+ will add them under `tests/rewards/`,
  `tests/policies/`, `tests/training/`)
- No regression failures of any kind were observed in the final,
  complete pytest run. Since P0 made zero source-code changes, there
  is nothing to attribute a regression to in any case.

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

V0 (not yet implemented — spec defined in
[PPO_PLAN.md § 5](PPO_PLAN.md#5-reward-v0-fixed-through-p5)). P0 does
not implement reward code (that's P2).

## Current PPO Config

Not yet implemented (P1/P3 scope). Baseline hyperparameters recorded in
[PPO_PLAN.md § 6](PPO_PLAN.md#6-ppo-baseline-architecture--hyperparameters):

```
learning_rate  = 3e-4
gamma          = 0.99
gae_lambda     = 0.95
clip_epsilon   = 0.2
value_coef     = 0.5
entropy_coef   = 0.01
```

## W&B Run ID

None yet (P2 scope).

## Checkpoint Path

None yet (P1 skeleton / P5 full contract).

## Last Command

```
pytest tests/ -q                     # 461 passed, 0 failed, 1798.33s (0:29:58)
PYTHONPATH=. python <scratch>/p0_rollout_throughput.py   # determinism+throughput check
pip install wandb                    # no --upgrade, no jax/numpy touched
```

## Known Issues

None. No contradiction found between PPO_PLAN.md and the current
frozen codebase during P0 — the `info_before`/pre-step pattern required
by §7.1 is directly supported by the existing `reset()`/`step()` API
(both return `info` reflecting `merge_committed` state as of that
call), and an equivalent pattern is already used in
`src/environment/full_split_evaluator.py`.

## Next Exact Action

**Begin P1 PPO Foundation.** Per the user's fixed one-commit-per-phase
constraint for this whole P0-P5 effort, do all P1 implementation, all
required P1 tests, and all P1 doc updates first, and create exactly
ONE commit at the very end of P1 (no intermediate/WIP/sub-stage
commits). Concretely, per
[PPO_PLAN.md §0.1/P1](PPO_PLAN.md#p1--ppo-foundation):

1. Create `src/rewards/`, `src/policies/ppo/`, `src/training/`,
   `src/tracking/` package directories (with `__init__.py`)
2. Create `configs/reward/`, `configs/ppo/` directories
3. Create `scripts/train_ppo.py`, `scripts/smoke_train_ppo.py` entry-point
   skeletons
4. Implement a config loader (PPO config + reward config, YAML-based to
   match this repo's existing `pyyaml` convention)
5. Implement seed handling
6. Implement checkpoint skeleton (structure only — full save/load
   contract lands in P5 per §10)
7. Run import checks + config-load checks; re-run existing regression
   suite to confirm nothing broke
8. Update PROGRESS.md/HANDOFF.md, append EXPERIMENT_LOG.md entry, commit
   `feat(ppo): add PPO foundation scaffolding` on `feat/ppo-phase0-5`
