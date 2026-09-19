# Pre-P6 Correctness/Instrumentation Hardening Report

Base `main` SHA: `c00743aaf0e613f70df3b85ebf7e9a4935f40644` (`c00743a`)
Branch: `feat/ppo-pre-p6`
Prior effort this builds on: `feat/ppo-phase0-5` (P0-P5, merged into
`main` at `c00743a` via PR #1).

This is a **correctness and instrumentation hardening pass**, not a
tuning pass. No reward weight, hyperparameter, network size, or
dataset-split change was made anywhere in this phase. Its sole purpose
is to fix bugs found in the P0-P5 implementation and add diagnostic
visibility the user will need to make informed P6 decisions.

---

## 1. PPO-paper-alignment summary

- **PPO-Clip objective: unchanged.** `ppo_ratio = exp(new_log_prob -
  old_log_prob)`; clipped surrogate
  `min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)`
  (`src/policies/ppo/loss.py::ppo_clipped_surrogate_loss`). Nothing in
  this phase touches that function's math.
- **Episode-aware GAE (Fix 1):** `src/training/gae.py::compute_gae`
  now assumes it is given ONE contiguous episode segment; a new
  `compute_gae_segmented` groups a flat, multi-episode rollout by
  `episode_id` and calls `compute_gae` independently per segment,
  concatenating results back in order. This closes a real bug: the old
  single flat call let a later episode's backward-recursion
  `gae_running` leak into an earlier truncated/cutoff episode's last
  steps. `build_training_batch` (`src/training/trainer.py`) now calls
  `compute_gae_segmented`, never bare `compute_gae`, over rollout
  output.
- **Separate Actor/Critic architecture — unchanged; value_coef
  resolved (Fix 4):** Policy and value networks remain fully
  independent `flax.training.train_state.TrainState`s with separate
  `apply_gradients` calls (`src/policies/ppo/state.py`). Because
  their losses are never combined into one shared gradient,
  `value_coef` (PPO paper c1) has no effect on this architecture's
  training dynamics. Resolution: **kept, not removed**, for backward
  compatibility with existing configs/checkpoints/tests and because
  `src/policies/ppo/loss.py::ppo_total_loss` (a standalone
  combined-loss helper, validated by its own P3 tests, never called by
  `run_update`) still legitimately consumes it. Documented consistently
  in `src/training/config.py::PPOHyperparameters.value_coef`'s
  docstring, `src/training/trainer.py::run_update`'s docstring, and
  `src/tracking/wandb_logger.py`'s config-key comment — no
  contradiction between code, config, and docs. `value_coef=0.5` in
  `configs/ppo/ppo_smoke.yaml`/`ppo_base.yaml` is unchanged from P0-P5
  and matches `PPO_PLAN.md` SS6.
- **Exact KL confirmed diagnostic-only (Fix 2):**
  `src/training/trainer.py::_exact_categorical_kl` computes the full
  categorical `KL(old || new)` per policy_mask==1 row from
  `Transition.old_logits` (populated only for `policy_mask==1` rows; a
  zero-vector sentinel otherwise) vs. the current policy's logits. It
  is added to the returned diagnostics dict (`exact_kl_mean`,
  `exact_kl_max`) but **never** added to `_policy_loss_fn`'s `total`
  loss and never used for early stopping — confirmed by direct
  inspection of `_policy_loss_fn` (the only place `total` is
  constructed: `total = policy_loss - entropy_coef * entropy`).

---

## 2. Summary of each fix

1. **Episode-aware GAE** (`src/training/gae.py`,
   `src/training/trainer.py::build_training_batch`) — see above.
   Also adds a `rollout_cutoff` flag (an artificial trainer-side
   `max_steps` cutoff, distinct from environment
   `terminated`/`truncated`) that bootstraps from `next_values` like
   truncation does, since a collection-loop cutoff is never grounds to
   withhold a value estimate the Critic already has.
2. **Exact categorical KL diagnostic** (`src/training/trainer.py`,
   `src/training/rollout.py::Transition.old_logits`) — see above.
   Never used as a loss term or stopping criterion.
3. **PPO update metric aggregation** (`src/training/trainer.py::run_update`)
   — every `ppo/*`/grad-norm metric is now accumulated (mean, and for
   grad-norms also max) across the FULL `ppo_epochs x
   num_minibatches` sweep of an update, via explicit `_mean`/`_max`
   suffixed keys, rather than being silently overwritten by the last
   minibatch/epoch's value. Additive to the existing metric-name
   contract — old bare names (`ppo/policy_loss` etc.) still exist,
   now populated from the last minibatch for backward compatibility,
   with the aggregated `_mean`/`_max` variants as the new
   correctness-fixed source of truth.
4. **value_coef resolution** — see above; kept + documented, not
   removed, with no code/config/docstring contradiction.
5. **NumPy RNG checkpointing** (`src/training/checkpoint.py::CheckpointPayload.numpy_rng_state`,
   `restore_numpy_rng`) — the NumPy `RandomState` governing PPO
   minibatch shuffling (`_minibatch_indices`) is now saved to and
   restored from checkpoints, closing a gap where only the JAX PRNG
   key (rollout action sampling) was persisted; a resumed run's
   minibatch order used to silently diverge from an uninterrupted
   run's. `numpy_rng_state` defaults to `None` so older checkpoints
   (saved before this field existed) still load, falling back to
   re-seeding from `seed`.
6. **W&B full diagnostics** (`src/training/trainer.py::run_training`,
   `src/tracking/wandb_logger.py::MINIMUM_METRICS`) — new metrics
   computed from the environment's own signals, never re-derived: episode
   outcome rates from `info["termination_reason"]`
   (`train/success_rate`/`collision_rate`/`offroad_rate`/`timeout_rate`),
   downstream-intervention rates from the environment's own cumulative
   `intervention_rate`/`planner_infeasible_count`/etc. fields
   (`_aggregate_downstream_rates`), `ppo/explained_variance` (standard
   `1 - Var(returns - values) / Var(returns)`, with a finite-value
   guard near-zero variance), `train/policy_decision_count`
   (`sum(policy_mask)`) / `train/physical_step_count`
   (`len(transitions)`), and `runtime/env_steps_per_sec` (wall-clock
   rollout throughput).
7. **Reward component logging correctness** (`src/training/trainer.py::run_training`)
   — `reward/terminal` now sums every step where `t.terminated OR
   t.truncated` is True (previously only `t.terminated`), so a
   `TRUNCATION_HORIZON` (-0.5) episode-end is correctly counted as a
   terminal-component event instead of silently leaking into
   `reward/decision_cost`. An artificial `rollout_cutoff` (Fix 1) is
   neither `terminated` nor `truncated`, so it is correctly excluded
   from `reward/terminal` without a special case.

---

## 3. New W&B diagnostics (full list, `src/tracking/wandb_logger.py::MINIMUM_METRICS`)

Pre-existing (P0-P5): `train/episode_return`, `train/success_rate`,
`train/collision_rate`, `train/offroad_rate`, `train/timeout_rate`,
`train/episode_length`, `reward/terminal`, `reward/decision_cost`,
`reward/total`, `ppo/policy_loss`, `ppo/value_loss`, `ppo/entropy`,
`ppo/approx_kl`, `ppo/clip_fraction`, `ppo/explained_variance`,
`ppo/grad_norm`, `action/keep_ratio`, `action/follow_ratio`,
`action/merge_ratio`, `action/stop_ratio`, `downstream/intervention_rate`,
`downstream/planner_infeasible_rate`, `downstream/collision_blocked_rate`,
`downstream/controller_failure_rate`, `downstream/invalid_reference_rate`,
`runtime/env_steps_per_sec`.

**New this phase:** `ppo/exact_kl_mean`, `ppo/exact_kl_max`,
`ppo/policy_loss_mean`, `ppo/value_loss_mean`, `ppo/entropy_mean`,
`ppo/approx_kl_mean`, `ppo/clip_fraction_mean`,
`ppo/policy_grad_norm_mean`, `ppo/policy_grad_norm_max`,
`ppo/value_grad_norm_mean`, `ppo/value_grad_norm_max`,
`train/policy_decision_count`, `train/physical_step_count`.

(`train/success_rate`/`collision_rate`/`offroad_rate`/`timeout_rate`
and the `downstream/*` fields and `runtime/env_steps_per_sec` were
already in the P0-P5 `MINIMUM_METRICS` list per `PPO_PLAN.md` SS8, but
were not actually being *computed and logged* by `run_training` until
this phase's Fix 6 — confirmed live during this phase's smoke re-run,
see §5.)

---

## 4. Targeted test results

| File | Tests | Result |
|---|---|---|
| `tests/training/test_gae.py` | (extended; +Fix 1 coverage) | pass |
| `tests/training/test_checkpoint.py` | (extended; +Fix 5 coverage) | pass |
| `tests/training/test_run_training.py` | (extended; +Fix 3/6 coverage) | pass |
| `tests/training/test_trainer.py` | (extended; +Fix 2 coverage) | pass |
| `tests/training/test_pre_p6_hardening.py` (new file) | 13 | pass |

Combined run of all 5 files: **65 passed, 0 failed, 238 warnings
(all `optax.global_norm` deprecation warnings, not failures), in
233.97s**. `--collect-only` over the same 5 files also reports 65,
confirming no silent skips/errors. `tests/training/test_pre_p6_hardening.py`
covers Fix 2 (exact KL diagnostic correctness/non-negativity/masking),
Fix 3 (aggregation not last-value), Fix 5 (NumPy RNG round-trip +
resumed-minibatch-order match), Fix 6 (downstream-rate aggregation,
explained-variance edge cases), and Fix 7 (reward component split
including truncation). Fix 1 has its own dedicated coverage added to
`tests/training/test_gae.py`.

---

## 5. Smoke-training results (real, both runs, this phase's code)

The original P0-P5 smoke-training evidence
(`outputs/ppo_checkpoints/smoke_stage1_final.pkl`/`smoke_stage2_final.pkl`,
`wandb/offline-run-20260919_024059-dodmdn2z`/`...-bsp9erzm`) predates
every fix in this phase (their W&B config recorded `git_sha=fe8edc9`,
the P4 completion commit) and its logged metrics confirm this: no
`success_rate`/`collision_rate`/`exact_kl`/`policy_decision_count`/
`explained_variance`/etc. fields are present, only the pre-existing
`train/*`, `reward/*`, `ppo/*` (bare names), `action/*` set. That
evidence is therefore insufficient to demonstrate this phase's fixes
actually work end-to-end, so this phase re-ran real fresh + resume
smoke training against the current code
(`WANDB_MODE=offline`, `--max-maneuvers 2 --max-episode-steps 60`).

**Fresh run** (`--num-updates 2`, no `--resume`,
`--checkpoint-path outputs/ppo_checkpoints/pre_p6_smoke.pkl`):
finished in 61.4s. `global_env_step=70`, `ppo_update_step=2`.

- Update 0: `episode_return=0.885`, `success_rate=1.0`,
  `collision_rate=0.0`, `offroad_rate=0.0`, `timeout_rate=0.0`,
  `policy_decision_count=23`, `physical_step_count=35`,
  `env_steps_per_sec=1.432`, `intervention_rate=0.350`,
  `planner_infeasible_rate=0.225`, `collision_blocked_rate=0.125`,
  `policy_loss_mean=-0.0586`, `value_loss_mean=0.4194`,
  `entropy_mean=1.2427`, `approx_kl_mean=0.0434`,
  `exact_kl_mean=0.0172`, `exact_kl_max=0.0774`,
  `explained_variance=-1.746`.
- Update 1: `episode_return=0.95`, `success_rate=1.0`,
  `collision_rate=0.0`, `offroad_rate=0.0`,
  `policy_decision_count=10`, `physical_step_count=35`,
  `env_steps_per_sec=1.653`, `intervention_rate=0.271`,
  `policy_loss_mean=-0.0396`, `value_loss_mean=0.0257`,
  `entropy_mean=1.2554`, `approx_kl_mean=0.0320`,
  `exact_kl_mean=0.0027`, `exact_kl_max=0.0194`,
  `explained_variance=-1.264`.

All values finite (no NaN/inf). Checkpoint saved to
`outputs/ppo_checkpoints/pre_p6_smoke.pkl` (545,023 bytes).

**Resume run** (separate process, `--resume
outputs/ppo_checkpoints/pre_p6_smoke.pkl`, `--num-updates 1`,
`--checkpoint-path outputs/ppo_checkpoints/pre_p6_smoke_resumed.pkl`):
printed `Resumed: global_env_step=70, ppo_update_step=2` on load
(exactly matching the fresh run's final state — confirming JAX+NumPy
RNG/state resume works, counters continue rather than reset), then
finished in 34.5s with `global_env_step=103, ppo_update_step=3`.

- Update 0: `episode_return=0.98`, `success_rate=1.0`,
  `collision_rate=0.0`, `offroad_rate=0.0`,
  `policy_decision_count=4`, `physical_step_count=33`,
  `env_steps_per_sec=1.337`, `intervention_rate=0.233`,
  `collision_blocked_rate=0.233`, `planner_infeasible_rate=0.0`,
  `policy_loss_mean=-0.0625`, `value_loss_mean=0.0240`,
  `entropy_mean=1.1692`, `approx_kl_mean=0.00866`,
  `exact_kl_mean=0.00623`, `exact_kl_max=0.0358`,
  `explained_variance=-1.444`.

All values finite throughout both runs; no NaN/inf, no crash, no OOM.
Both runs exited with code 0. Checkpoint saved to
`outputs/ppo_checkpoints/pre_p6_smoke_resumed.pkl` (545,022 bytes).

Note on `explained_variance` reading negative: expected and not a bug
— this is a ~2-update, ~2-maneuver smoke run (pipeline verification
only, per this phase's own non-goals), far too little training signal
for the value function to explain return variance meaningfully yet;
the metric itself is computed correctly (guarded for the near-zero-
variance edge case, confirmed by
`test_explained_variance_finite_and_correct_near_zero_variance`/
`test_explained_variance_normal_case_between_reasonable_bounds`).

(Housekeeping note: an earlier attempt at this same re-run within this
session hit a GPU `FailedPreconditionError: Failed to allocate scratch
buffer` from a leftover, still-running process holding ~7.3GB of GPU
memory from a prior turn in this same session; that process was
identified and killed, its partial/failed artifacts — a stray
checkpoint path and an empty `wandb/offline-run-...` directory — were
removed before the real, clean re-run above.)

---

## 6. Full regression result

**629 passed, 0 failed, in 2169.30s (0:36:09)**, run solo (no
concurrent GPU-contending process), verified by the orchestrating
session. Baseline going into this branch (P0-P5 completion) was 587;
`pytest --collect-only -q` over the current tree independently
confirms **629 tests collected**, consistent with the reported count.
`pytest --collect-only -q` restricted to the 5 changed test files
reports 65, matching the standalone 65-test run in §4 exactly (no
silent skips).

---

## 7. Frozen-invariant / scope-compliance verification

- `git diff --stat main -- src/environment/ src/planning/ src/control/ src/scenarios/`:
  **empty**. Zero frozen Phase 1-3 files touched.
- No PPO-FIT/PPO-TUNE dataset split exists anywhere in the repo
  (grepped for the string; only comments/docs reaffirming its absence
  are present, as in P0-P5).
- Reward V0 values unchanged (`configs/reward/merge_reward_v0.yaml`):
  `success=+1.0`, `failure_collision=-1.0`, `failure_offroad=-1.0`,
  `truncation_horizon=-0.5`, `none=0.0`,
  `decision_cost.real_decision_step=-0.01`,
  `decision_cost.auto_execution_step=0.0`. Only logging/breakdown
  changed (Fix 7).
- PPO hyperparameters unchanged (`configs/ppo/ppo_smoke.yaml`,
  `ppo_base.yaml` byte-identical to `main`;
  `src/training/config.py`'s diff against `main` is docstring-only):
  `learning_rate=3e-4`, `gamma=0.99`, `gae_lambda=0.95`,
  `clip_epsilon=0.2`, `entropy_coef=0.01`, `value_coef=0.5`,
  network `[256, 64, 32]` tanh, `ppo_epochs=4`, `num_minibatches=4`.
  This was explicitly not a tuning pass.
- `configs/ppo/ppo_tune.yaml` exists on disk as an **untracked**,
  unreferenced-by-any-code file (a P6-scoped scaffold, presumably
  staged by a prior session for the user's own future tuning work).
  It is intentionally **left untracked and excluded from this phase's
  commit** — it is P6-scoped material, out of bounds for this phase
  per its own explicit stop condition, and this phase makes no
  judgment about whether the user wants to keep, edit, or discard it.

---

## 8. Known limitations

- Smoke-training numbers above are from a ~2-maneuver, 2-3-update
  pipeline-verification run — not statistically meaningful for
  judging policy quality (by design; see PPO_PLAN.md SS0.1's
  non-goals for this class of run). `explained_variance` reading
  negative is expected at this scale, not a defect.
  `ppo/approx_kl_mean` and `ppo/exact_kl_mean` diverge somewhat
  between updates/runs (e.g. update 0 fresh-run: approx=0.0434,
  exact=0.0172) because they measure genuinely different quantities
  (a linearized/sampled approximation vs. the full closed-form
  categorical KL) — both are diagnostic-only and this is expected,
  not a bug.
- `value_coef` remains in the config/checkpoint contract for backward
  compatibility even though it has no effect on this architecture's
  training dynamics; a future session should not spend time "tuning"
  it.
- `numpy_rng_state` is `None`-tolerant for backward compatibility with
  any pre-Fix-5 checkpoint; resuming from such an old checkpoint will
  re-seed the NumPy stream from `seed` rather than truly continuing
  it (a real, if minor, degradation documented in
  `CheckpointPayload`'s own docstring, not a crash).
- This pass did not re-verify P0-P5's own already-covered semantics
  beyond what changed; it assumes P0-P5's SMOKE_TRAINING_REPORT.md and
  its own test suite remain the source of truth for everything this
  phase did not touch.

---

## 9. What the user should do next (P6)

Nothing in this phase recommends or scopes any P6 action. As before
(see HANDOFF.md's NEXT OWNER ACTION), the user should review the real
W&B diagnostics now available — including the newly-fixed
`success_rate`/`collision_rate`/`offroad_rate`,
`policy_decision_count`/`physical_step_count`, `exact_kl_mean/max`,
and `downstream/*` intervention rates — themselves (e.g. via `wandb
sync` on a local offline run directory, or a fresh longer run) and
decide, on their own judgment, whether/how to proceed into reward
tuning, hyperparameter tuning, a real PPO-FIT/PPO-TUNE split, longer
training, canonical VAL evaluation, or an FSM-vs-PPO comparison. None
of that has been started, scoped, or recommended by this phase.
`configs/ppo/ppo_tune.yaml` (untracked, see §7) may be a useful
starting scaffold for that future work, at the user's discretion.

---

## 10. Follow-up fix: reward component logging correctness (post-report)

Landed in the commit immediately following `d75b593` on this branch
(this report's own commit). Scope: a narrow, additive bug fix to Fix
7's `reward/terminal`/`reward/decision_cost` W&B breakdown — **not** a
reopening of §9's "no P6 action recommended" stance.

**What was wrong:** Fix 7 (§1/§6 above) corrected `reward/terminal` to
include TRUNCATION_HORIZON's `-0.5` component (previously only rows
with `terminated == True` were counted), but its aggregation formula
was still `sum(t.reward for t in transitions if t.terminated or
t.truncated)`. `t.reward` on a terminal/truncated row is
`terminal_component + decision_cost_component` combined (see
`src/rewards/merge_reward.py`), not just the terminal part. So whenever
a terminal/truncated outcome landed on a *real policy-decision* step
(e.g. SUCCESS `+1.0` with decision cost `-0.01`, `t.reward == +0.99`),
the old formula dumped the **whole** `+0.99` into `reward/terminal`
instead of splitting it into `+1.0` terminal / `-0.01` decision cost —
skewing both metrics whenever that combination occurred.

**The fix — Transition-level component propagation:**
`src/rewards/reward_wrapper.py`'s `MergeRewardWrapper.compute` already
computed and exposed the exact terminal/decision-cost components for
the step it just processed, via `last_terminal_component`/
`last_decision_cost_component` (unchanged by this fix — it is the
frozen source of truth). The bug was that neither component was ever
propagated onto the `Transition` the rollout loop builds for that
step, leaving `trainer.py` to guess from `terminated`/`truncated`
alone. Fixed by:

1. `src/training/rollout.py`: `Transition` gains two additive fields,
   `reward_terminal_component: float = 0.0` and
   `reward_decision_cost_component: float = 0.0` (defaulting to 0.0 so
   any older/synthetic `Transition` construction still works).
2. `collect_episode_rollout` reads `reward_wrapper.
   last_terminal_component`/`last_decision_cost_component` immediately
   after the existing `reward_wrapper.compute(...)` call for that step
   and stores them on the `Transition` it constructs — a pure
   pass-through, never re-derived from `terminated`/`truncated`.
3. `src/training/trainer.py`'s `run_training` per-update metrics block
   now computes `reward_terminal = sum(t.reward_terminal_component ...)`
   and `reward_decision_cost = sum(t.reward_decision_cost_component
   ...)` directly, replacing the old `terminated`/`truncated`-based
   guess. `reward_total` is still `np.sum(batch["reward"])`.

**Artificial rollout-cutoff edge case, verified not assumed:** when
`collect_episode_rollout`'s `max_steps` loop exhausts without the
environment itself reporting `terminated`/`truncated`
(`rollout_cutoff=True`), `reward_wrapper.compute` is called with that
step's real `info_after["termination_reason"]` — which the
environment itself set to `"none"` (non-terminal), since it did not
terminate — so `last_terminal_component` is already `0.0` for that
call with no special-casing needed. This was confirmed empirically
(not assumed) by
`tests/training/test_rollout.py::test_reward_components_zero_terminal_on_artificial_cutoff`
against the real environment.

**Reward V0 values confirmed unchanged:** `success=+1.0`,
`failure_collision=-1.0`, `failure_offroad=-1.0`,
`truncation_horizon=-0.5`, `none=0.0`, `decision_cost` `-0.01`
(real policy-decision step) / `0.0` (auto-execution step) — verified
via `git diff --stat -- configs/ src/rewards/` (empty) and by every
new test asserting these exact fixed-table values.
`src/rewards/reward_wrapper.py`'s `MergeRewardWrapper` itself was not
modified — only read.

**Targeted test results:** added 12 new tests (Tests A-H from this
fix's task scope, plus two supporting real-environment/edge-case
tests) across `tests/training/test_pre_p6_hardening.py`,
`tests/training/test_rollout.py`, and `tests/training/
test_run_training.py`. Full targeted run —
`pytest tests/training/test_run_training.py
tests/training/test_pre_p6_hardening.py tests/rewards/
tests/training/test_trainer.py tests/training/test_rollout.py -q` —
**93 passed, 0 failed**.

**Smoke-training confirmation:** `PYTHONPATH=. WANDB_MODE=offline
python scripts/smoke_train_ppo.py --max-maneuvers 2 --num-updates 1
--max-episode-steps 60` on 2 real canonical-TRAIN maneuvers, both
reaching real SUCCESS terminal outcomes on real policy-decision steps
— exactly the case the bug mishandled. Result:
`train/success_rate=1.0`, `reward/terminal=2.0` (exactly `2 × +1.0`,
the fixed SUCCESS table value — no decision-cost contamination),
`reward/decision_cost=-0.23` (`23 × -0.01`, matching
`train/policy_decision_count=23.0` exactly), `reward/total=1.77`
(`2.0 + -0.23`). This is the genuine component split the fix is meant
to produce; under the pre-fix code, `reward/terminal` here would have
been inflated to `1.98` (`2 × 0.99`) and `reward/decision_cost`
correspondingly deflated.

**Full regression suite result:** `pytest tests/ -q` in the
`its-merge` conda environment (confirmed no concurrent `pytest`
process first, per §"Lesson learned" in HANDOFF.md/EXPERIMENT_LOG.md)
— **641 passed, 0 failed, in 2072.81s (0:34:32)**. Baseline going into
this fix (from `d75b593`) was 629 passed; 629 + 12 new tests = 641,
confirming no test was lost or silently skipped.

State remains **PRE-P6 HARDENING COMPLETE — WAITING FOR USER TUNING**
(§9 above still applies unchanged) — this was a correction within the
same phase, not new P6-scoped work.
