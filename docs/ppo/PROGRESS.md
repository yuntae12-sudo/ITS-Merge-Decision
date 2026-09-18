# PPO Progress Tracker

Live state, updated at the end of every Phase/Stage. See
[PPO_PLAN.md](PPO_PLAN.md) for the durable plan and
[HANDOFF.md](HANDOFF.md) for session-resume instructions.

---

## Current Phase

**P0 — Baseline Audit (not started)**

## Current Stage

Final plan correction applied; awaiting user final approval.

`PPO_PLAN.md` now contains the complete per-Phase P0–P5 execution and
acceptance-criteria reference (§0.1) and is the durable Source of Truth
for the whole effort — a future session can recover the full plan from
that file alone. This pass also corrected the §7.2 Actor-mask unit-test
description (it previously implied the wrong invariant — see
[HANDOFF.md](HANDOFF.md) for the exact correction) and made the
`TerminationReason` → Reward V0 mapping explicit in §0.1's P2 section.
Docs were previously relocated from `docs/phase4/` to `docs/ppo/`.
Awaiting final user approval before any implementation work (including
P0 audit) begins.

## SHA / Branch

- base SHA: `c5d2c1d2ea197cd247b3bc2e4f127b3c36d1a990`
- current SHA: `c5d2c1d2ea197cd247b3bc2e4f127b3c36d1a990`
- branch: `main`
  - Spec requires creating `feat/ppo-phase0-5` before real implementation
    work starts — this has **not** been done yet.

## Completed Tasks

- [x] Created `docs/ppo/PPO_PLAN.md`, `PROGRESS.md`, `HANDOFF.md`,
      `EXPERIMENT_LOG.md` (originally under `docs/phase4/`)
- [x] Relocated docs from `docs/phase4/` to `docs/ppo/`
- [x] Strengthened plan per user review: policy_mask pre-step timing,
      Actor-vs-Critic mask scope, reward source-of-truth rules,
      checkpoint/resume contract, unified branch order, action-mapping
      regression test
- [x] Corrected §7.2 Actor-mask unit-test description (masked frames
      must never leak into Actor normalization stats — GAE/return is
      not expected to stay unchanged if a real auto-execution frame is
      removed, since that changes the physical trajectory itself)
- [x] Added durable per-Phase P0–P5 execution plan to PPO_PLAN.md §0.1
      (Goal / Tasks / Required tests / Completion criteria / Explicit
      non-goals for every Phase)
- [x] Made `TerminationReason` enum → Reward V0 mapping explicit
      (SUCCESS/FAILURE_COLLISION/FAILURE_OFFROAD/TRUNCATION_HORIZON/NONE)
- [ ] Final user approval of plan documents (pending)

## Changed Files

- `docs/ppo/PPO_PLAN.md` (new, revised)
- `docs/ppo/PROGRESS.md` (new, revised)
- `docs/ppo/HANDOFF.md` (new, revised)
- `docs/ppo/EXPERIMENT_LOG.md` (new, revised)

## Current Test Results

Not run yet for PPO. Existing Phase 1–3 regression tests have not been
re-run as part of this effort yet (P0 will do this first).

## Current Reward Version

V0 (not yet implemented — spec defined in
[PPO_PLAN.md § 5](PPO_PLAN.md#5-reward-v0-fixed-through-p5)).

## Current PPO Config

Not yet implemented. Baseline hyperparameters recorded in
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

None yet.

## Checkpoint Path

None yet.

## Last Command

None (documentation-only step; only `mkdir`/`mv` used to relocate
`docs/phase4/` → `docs/ppo/`, no implementation commands run).

## Known Issues

None yet.

## Next Exact Action

**Wait for user's final approval of the plan documents.** Do not start
P0, do not install dependencies, do not create source/config/test files,
and do not create the `feat/ppo-phase0-5` branch until approval is given.

Once approved, begin in this order (unified with HANDOFF.md and
[PPO_PLAN.md § 9](PPO_PLAN.md#9-git-conventions)):

1. `git status`, `git branch`, `git log -n 20`
2. Create `feat/ppo-phase0-5` branch (carry current uncommitted docs
   into it)
3. Begin **P0 Baseline Audit**:
   1. Record JAX / jaxlib / CUDA device / NumPy / Waymax versions
      (`python -c "import jax; print(jax.devices())"`)
   2. Confirm Waymax import works
   3. Review `requirements.txt`, `configs/`, `data/manifests/`
   4. Confirm `OBSERVATION_DIM == 14` and field order
      ([observation_builder.py](../../src/environment/observation_builder.py))
   5. Confirm `BehaviorAction` semantics
      ([behavior_action.py](../../src/environment/behavior_action.py))
   6. Confirm termination semantics
      ([termination.py](../../src/environment/termination.py))
   7. Confirm `downstream_mode="frenet_mpc"` path exists and is used
   8. Confirm intervention diagnostics are intact
   9. Run full existing test suite (`pytest`)
   10. Reproduce a short rollout for determinism sanity check
   11. Take a simple runtime throughput measurement
4. Update this file + `HANDOFF.md`, commit
   `chore(ppo): audit frozen training baseline` on `feat/ppo-phase0-5`
