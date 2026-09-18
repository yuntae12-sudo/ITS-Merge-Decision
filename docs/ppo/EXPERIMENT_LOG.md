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
