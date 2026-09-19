# PPO Progress Tracker

Live state, updated at the end of every Phase/Stage. See
[PPO_PLAN.md](PPO_PLAN.md) for the durable plan and
[HANDOFF.md](HANDOFF.md) for session-resume instructions.

---

## Current Phase

**PRE-P6 HARDENING COMPLETE — WAITING FOR USER TUNING**

The entire P0–P5 PPO implementation effort is complete, and a
follow-up **Pre-P6 correctness/instrumentation hardening pass** (on
branch `feat/ppo-pre-p6`, based on `main` at `c00743a`) has also been
completed on top of it. This pass fixed 7 issues found in the P0-P5
implementation — episode-aware GAE (a real cross-episode advantage-
leakage bug), an exact categorical-KL diagnostic (monitoring-only,
never in any loss), PPO update metric aggregation (was silently
overwriting with the last minibatch/epoch's value instead of
aggregating across the full sweep), a `value_coef` documentation/
consistency resolution (kept, has no effect on this architecture,
documented consistently everywhere it's referenced), NumPy RNG
checkpointing (minibatch-shuffle order now survives a resume), full
W&B diagnostics (success/collision/offroad/timeout rates,
policy_decision_count/physical_step_count, explained_variance,
downstream/* rates, env_steps_per_sec, exact_kl_mean/max actually
computed and logged now), and reward component logging correctness
(`TRUNCATION_HORIZON` episodes now correctly counted as a
terminal-component event). **No reward weight, hyperparameter,
network size, or dataset-split change was made — this was explicitly
not a tuning pass.** Full details, real re-run smoke-training numbers
(the original P0-P5 smoke evidence predated these fixes and lacked
the new diagnostics, so this pass re-ran fresh + resume smoke training
against the current code), and the full regression result (629
passed, 0 failed) are in
[PRE_P6_REPORT.md](PRE_P6_REPORT.md). P6+ (reward/hyperparameter
tuning, full training, VAL evaluation, FSM-vs-PPO comparison) remains
reserved for the user — see [HANDOFF.md](HANDOFF.md)'s NEXT OWNER
ACTION.

### P5 COMPLETE (Smoke Training) — prior phase summary below

## Current Stage

P0, P1, P2, P3, and P4 are all complete. P4 connected the real P3 PPO
core to the real `MergeEnvironment` (`downstream_mode="frenet_mpc"`,
per SS2), implementing `src/training/rollout.py` (real
`collect_episode_rollout`/`collect_rollout`, replacing the P1
`NotImplementedError` skeleton), `src/training/gae.py` (real
`compute_gae`/`normalize_advantages_masked`/`masked_mean_std`), and
`src/training/trainer.py::build_training_batch` (real end-to-end
rollout -> GAE -> masked-normalized PPO-ready batch dict; the
multi-epoch parameter-UPDATE loop itself, `run_training`, correctly
remains a `NotImplementedError` stub per P4's explicit non-goal --
that's P5 scope).

`rollout.py`'s `collect_episode_rollout` mirrors the exact pre-step
`info_before["merge_committed"]` pattern P0 confirmed is already used
by `src/environment/full_split_evaluator.py::run_episode`
(`was_committed_before` read BEFORE calling the policy/`env.step`):
`is_policy_step = not info_before.get("merge_committed", False)` is
computed from the PREVIOUS `reset()`/`step()` call's `info` dict,
never from the current step's own post-step `info`. On an
auto-execution step (`is_policy_step is False`), the PPO policy
network is NOT called for action selection at all --
`BehaviorAction.MERGE` is submitted directly, matching
`full_split_evaluator.py`'s precedent exactly (a `log_prob` value is
still computed for that step purely so `Transition.log_prob` stays a
well-defined finite non-Optional field; SS7.2 already excludes
`policy_mask == 0` frames from every Actor-side computation
downstream, so this value is never actually consumed by policy loss/
entropy/KL/advantage-norm). Each `Transition` carries the fixed
SS0.1/P4 field set (`observation`, `action`, `reward`,
`next_observation`, `terminated`, `truncated`, `value`, `next_value`,
`log_prob`, `policy_mask`, plus diagnostics `episode_id`,
`maneuver_id`, `step_index`) and covers the FULL physical trajectory
(every frame, regardless of `policy_mask` -- SS7.2's Critic scope
requires this; no frame is ever dropped at collection time). Reward is
computed via P2's `MergeRewardWrapper.compute`, fed the SAME
pre-step `is_policy_step` flag and the environment's own
post-step `info["termination_reason"]` -- never re-deriving
termination (SS5.1).

`gae.py::compute_gae` implements the standard backward-recursion GAE
(Schulman et al. 2015 eq. 16) over the FULL physical trajectory,
`gamma=0.99`/`gae_lambda=0.95` (from `ppo_base.yaml`, SS6, unchanged),
correctly distinguishing true termination (`terminated=True` ->
bootstrap mask 0, no value flows past a terminal state) from
truncation (`truncated=True` -> bootstrap mask 1, DOES bootstrap from
`next_value` -- the SS0.1/P4 required test). `normalize_advantages_masked`
computes the Actor-side normalization mean/std using ONLY
`policy_mask == 1` positions (SS7.2) -- masked-out (`policy_mask == 0`)
advantage values structurally never enter the `np.mean`/`np.std`
reduction (they are excluded via boolean-mask indexing before the
reduction runs, not merely down-weighted), which is what makes SS7.2
test A (masked-frame invariance) hold by construction, not by
coincidence. `masked_mean_std` exposes the same statistic standalone
for that test to assert on directly.

`trainer.py::build_training_batch` wires rollout -> GAE -> masked
normalization into one flat batch dict (`observation`, `action`,
`reward`, `next_observation`, `terminated`, `truncated`, `value`,
`next_value`, `log_prob`, `policy_mask`, `advantages_raw` [unmasked,
full-trajectory GAE output, kept for diagnostics], `advantages`
[masked-normalized, SS7.2 Actor scope], `returns` [full-trajectory
GAE Critic target]) and raises `ValueError` on any non-finite value
anywhere in the batch or on an empty transition list. `run_training`
is explicitly left as a `NotImplementedError` stub with an updated
docstring pointing at `build_training_batch` as what P5's update loop
will call.

38 new P4 tests were added: `tests/training/test_gae.py` (11 --
two independently hand-computed multi-step GAE trajectories matching
exactly, a terminal-step-does-not-bootstrap test [changing next_value
on a terminated step changes nothing], a truncated-step-DOES-bootstrap
test [changing next_value on a truncated step DOES change the
advantage], a mismatched-lengths `ValueError` guard, a no-NaN/inf
sweep over 50 random steps, SS7.2 test A [masked-frame invariance --
changing ONLY `policy_mask==0` advantage values leaves the
`policy_mask==1` mean/std and normalized values byte-identical] plus
its necessary counterpart [the statistic DOES change when
`policy_mask==1` values themselves change, proving the invariance test
isn't vacuous], a manual-computation cross-check, an
all-masked-out `ValueError` guard, and a shape-mismatch `ValueError`
guard), `tests/training/test_rollout.py` (8 -- a real 1-episode
rollout against the actual `MergeEnvironment` with the real PPO
policy that runs to completion [terminated or truncated] on maneuver
`MAN_0001`; rollout shape consistency; a full-episode no-NaN/inf sweep;
multi-maneuver `collect_rollout` concatenation across 2 episodes; the
SS7.1 pre-step MERGE `policy_mask` regression test THE most important
test in this Phase, see below; the SS7.2 test D terminal-reward-
propagation test, see below; an integration-level SS7.2 test B
[masked frames' log_prob/action values provably never leak into the
`policy_mask==1`-filtered subset a P5 loss computation would consume];
and an SS11 action-mapping regression confirming `rollout.py` performs
no second/separate action-index translation of its own, delegating
entirely to `distribution.ACTION_INDEX_TO_BEHAVIOR`), and
`tests/training/test_trainer.py` (3 -- a real end-to-end
rollout -> GAE -> masked-normalized-batch test on `MAN_0001` asserting
every required key/shape is present and the masked decision-frame
advantages have approximately zero mean; an empty-transition-list
`ValueError` guard; confirming `run_training` still raises
`NotImplementedError`, the explicit P4 non-goal). `tests/training/
test_imports.py` was updated in place (stub-`NotImplementedError`
checks for `rollout`/`gae`/`build_training_batch` replaced with
real-attribute-presence assertions and one narrower `run_training`
-still-a-stub check; still 6 collected items, same count as P2/P3).

**The SS7.1 pre-step MERGE `policy_mask` regression test**
(`test_prestep_merge_policy_mask_regression`) is the test this whole
Phase existed to guard: it runs a scripted-MERGE policy (always
selects `BehaviorAction.MERGE`, backed by a real policy network so
`log_prob`/logits stay well-defined) against the real `MergeEnvironment`
on maneuver `MAN_CAUSALITY` (`lane_chain=[527, 544]`, a single-
transition, non-chained maneuver, so MERGE at decision frame 0 commits
immediately and every subsequent frame is unambiguously auto-execution
for the rest of the episode -- confirmed by
`tests/environment/test_merge_environment_frenet_mpc.py::
test_single_maneuver_merge_reaches_success_frenet_mpc` to reach a real
terminal outcome, success/collision/offroad, within 60 steps of
continuous MERGE). It asserts: the VERY FIRST transition (the decision
frame where MERGE is selected) has `policy_mask == 1`, and every
transition strictly after it (auto-execution, `merge_committed` already
`True` going into the step) has `policy_mask == 0` -- directly guarding
against the reversed post-step bug SS7.1 describes (reading
`info_after["merge_committed"]` would have mislabeled exactly that
first frame `policy_mask == 0`). **PASSED.**

**SS7.2 test D** (`test_terminal_reward_propagates_to_merge_decision_frame`),
using the same scripted-MERGE-on-`MAN_CAUSALITY` construction: confirms
the episode reaches a real terminal reward (`abs(terminal_reward) >=
0.5`, i.e. a genuine +-1.0 SUCCESS/COLLISION component, not merely a
-0.01/0.0 decision-cost value), then runs `compute_gae` over the full
transition list and asserts the MERGE decision frame's (`index 0`) GAE
`return` is LARGER in magnitude than that frame's own tiny immediate
reward, with the same sign as the terminal outcome -- i.e. the terminal
SUCCESS/COLLISION reward's credit demonstrably propagates backward
through GAE, across the intervening `policy_mask == 0` auto-execution
frames, to the `policy_mask == 1` decision frame that triggered the
commitment. **PASSED.**

All 38 new P4 tests pass. Ran `tests/training/ tests/policies/
tests/rewards/` together first (114 passed in 148.94s) as an
intermediate sanity check before the full suite. `git diff --stat`
against `src/environment/`, `src/planning/`, `src/control/`,
`src/scenarios/` is confirmed EMPTY -- P4 touched zero frozen Phase 1-3
files, matching every prior Phase. `pytest tests/ --collect-only -q`
independently confirms "579 tests collected" (557 P3 baseline + 22 net
new P4 tests exactly: 11 gae + 8 rollout + 3 trainer = 22;
`test_imports.py` stayed at 6 collected items, no net change).

The full existing regression suite passes with **zero failures**:
**579 passed, 0 failed, in 1920.04s (0:32:00)**, verified by watching
the exact background `pytest` process (confirmed via `pgrep`/`ps aux`
to be the sole `python -m pytest tests/` process running, no
concurrent contention) run to REAL completion -- the final summary
line read directly from the completed log file (100% dots across every
module, no `F`/`E` marks) after the process's own PID was confirmed no
longer running. `jax.devices()` re-checked immediately after this run
still reports `[CudaDevice(id=0)]` -- GPU confirmed working, no
regression from any P4 change (P4 added no new dependency).

P0, P1, P2, and P3 completion summaries below are unchanged from the
prior entry. P0 confirmed the frozen baseline
(461 passed, 0 failed) and was audit-only (zero source changes). P1
added the full PPO scaffolding directory/module structure per
PPO_PLAN.md §4. P2 implemented Reward V0 (`src/rewards/merge_reward.py`,
`src/rewards/reward_wrapper.py`) and the W&B logging foundation
(`src/tracking/wandb_logger.py`), replacing their P1
`NotImplementedError` skeletons with real logic, per PPO_PLAN.md
§0.1/P2, §5, §5.1, §8. P3 implemented the real discrete
PPO core in `src/policies/ppo/{networks,distribution,policy,loss,state}.py`
(replacing their P1 `NotImplementedError` skeletons): a separate
policy (Actor) network and value (Critic) network, each
`14 -> 256 -> 64 -> 32 -> {4 logits | 1 value}` with `tanh` activations
between every hidden layer (§6, verified by direct inspection of
`networks.py` — no shared trunk, matching §7.2's Actor/Critic
separation); a categorical distribution module (`distribution.py`)
providing stochastic sampling (`jax.random.categorical`), deterministic
argmax inference, `log_prob` (via `log_softmax` + `take_along_axis`),
and entropy, plus the fixed §11 action-index -> `BehaviorAction`
mapping (imported from the real `src.environment.behavior_action`
enum, not a hardcoded duplicate); the PPO clipped surrogate objective,
value loss (MSE), entropy regularization, and a combined
`ppo_total_loss` (`loss.py`) whose diagnostic `info` dict carries
`policy_loss`/`value_loss`/`entropy`/`ratio`/`clip_fraction`/
`approx_kl` for future §8 W&B logging; and `PPOTrainState`/
`create_train_state` (`state.py`) building two independent
`flax.training.train_state.TrainState`s (policy, value) each with
`optax.chain(clip_by_global_norm(max_grad_norm), adam(learning_rate))`,
using the exact §6 baseline hyperparameters from `ppo_base.yaml`
(`learning_rate=3e-4`, `clip_epsilon=0.2`, `value_coef=0.5`,
`entropy_coef=0.01`, `max_grad_norm=0.5`). `policy.py`'s `PPOPolicy`
wraps a network + params into `.act()` (stochastic) / `.act_deterministic()`
(argmax) convenience methods.

P3 validated the algorithm in isolation only (synthetic/toy `jax.random`
inputs throughout `tests/policies/test_ppo_core.py`) — no real
`MergeEnvironment` rollout was performed, per §0.1/P3's explicit
non-goal; that integration is P4 scope. `git diff --stat` against
`src/environment/`, `src/planning/`, `src/control/`, `src/scenarios/`
is confirmed empty (no frozen file touched), and a repo-wide grep
confirms no file under those four frozen directories imports
`src.policies` — dependency direction remains strictly
PPO -> Environment, never the reverse, matching §1/§2.

34 new tests were added in the new file `tests/policies/test_ppo_core.py`
(independently confirmed via `pytest tests/policies/test_ppo_core.py
--collect-only -q` -> "34 tests collected"), covering every §0.1/P3
required-test item: 14D input handling (unbatched and batched), logits
shape == 4, finite logits, softmax probabilities sum to 1, value output
is scalar (unbatched) / vector (batched) and finite, sampled action
always in `0..3` across 20 keys, deterministic inference == argmax,
stochastic sampling varies across keys, finite `log_prob` for every
action, finite and non-negative entropy (plus a closed-form sanity
check: uniform logits -> entropy == `log(4)`), the §11 action-mapping
regression against the real `BehaviorAction` enum, 6 hand-checked PPO
ratio values, 3 clipping-behavior tests (upper-bound clip engaged,
lower-bound NOT clipped — the correct asymmetric PPO behavior — and no
clip when the ratio is within bounds), scalar+correct value loss,
finite full PPO loss + finite `info` dict for a synthetic minibatch,
finite and non-zero gradients for both policy and value networks,
optimizer steps that provably change parameters for both networks,
same-seed reproducibility for both network init and categorical
sampling, different-seeds-can-differ, and `PPOPolicy`
wrapper/`create_train_state` behavioral checks.

`tests/policies/test_imports.py` was updated in place: still verifies
module-level structure/constants (§6 layer sizes, the §11 mapping) but
its previous `NotImplementedError`-stub-raises assertions for
`networks`/`distribution`/`loss`/`policy`/`state` were removed since
those modules are now real (behavioral coverage lives in
`test_ppo_core.py` instead) — net effect: 6 collected items, down from
7 in P1/P2 (one stub-only test consolidated away), independently
confirmed via `pytest tests/policies/test_imports.py --collect-only -q`
-> "6 tests collected". Net P3-specific delta: 34 (new file) + (6 - 7)
(test_imports.py) = **33 new tests** over the P2 baseline of 524,
giving exactly **557** — independently confirmed via
`pytest tests/ --collect-only -q` -> "557 tests collected". (Running
`tests/policies/` alone, i.e. both P3 files together, collects 40 items
— that combined-directory count is what an earlier session report
referred to as "40/40 new tests passing"; the +33 figure here is the
correct net delta against the full-suite baseline once
`test_imports.py`'s -1 is accounted for. No reconciliation gap
remains.)

The full existing regression suite passes with **zero failures**:
**557 passed, 0 failed, in 1802.23s (0:30:02)**, verified by the
orchestrating session watching the exact background `pytest` process
run to real completion with no other `pytest` process running
concurrently (confirmed via `ps aux` beforehand) — this is the clean,
uncontended, final P3 regression result (524 pre-existing Phase 1-3 +
P0 + P1 + P2 tests + 33 new P3-specific tests = 557, matching exactly;
independently re-confirmed via `pytest tests/ --collect-only -q` ->
"557 tests collected" in this same session). `jax.devices()` was
re-checked in this session and still reports `[CudaDevice(id=0)]` — GPU
confirmed working, no regression from any P3 change (P3 added no new
dependency).

P0, P1, and P2 completion summaries below are unchanged from the prior
entry. P0 confirmed the frozen baseline
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
- P2 completion commit SHA: `565a7fe` (`feat(ppo): implement Reward V0
  and W&B logging foundation (P2)`)
- P3 completion commit SHA: `334e4c6` (`feat(ppo): implement discrete
  PPO core (P3)`)
- P4 completion commit SHA: `fe8edc9` (`feat(ppo): implement rollout
  and GAE integration (P4)`)
- P5 completion commit SHA: `755a498` (`test(ppo): validate smoke
  training pipeline (P5)`) — the final commit of the entire P0-P5
  effort, per the one-commit-per-Phase rule
- branch: `feat/ppo-phase0-5` (P0-P5), merged into `main` at `c00743a`
  via PR #1

### Pre-P6 hardening pass (follow-up effort, separate branch)

- base SHA: `c00743aaf0e613f70df3b85ebf7e9a4935f40644` (`c00743a`,
  `main` after the P0-P5 merge)
- branch: `feat/ppo-pre-p6`
- Pre-P6 completion commit SHA: see `git log` on this branch — the
  single commit for this entire phase, per its own one-commit
  constraint
- See [PRE_P6_REPORT.md](PRE_P6_REPORT.md) for the full report.

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
- [x] **P3 Discrete PPO Core**:
  - [x] Implemented `src/policies/ppo/networks.py`: real `flax.linen`
        `PolicyNetwork` (14 → 256 → 64 → 32 → 4 logits, tanh
        activations between every Dense layer) and `ValueNetwork`
        (14 → 256 → 64 → 32 → 1, tanh, squeezed to scalar/`(batch,)`),
        plus `build_policy_network`/`build_value_network`/
        `init_policy_params`/`init_value_params` helpers. Fixed
        architecture constants unchanged from P1
        (`POLICY_HIDDEN_SIZES`, `VALUE_HIDDEN_SIZES`,
        `OBSERVATION_DIM=14`, `NUM_ACTIONS=4`).
  - [x] Implemented `src/policies/ppo/distribution.py`: real
        `sample_action` (`jax.random.categorical`),
        `deterministic_action` (`jnp.argmax`), `log_prob`
        (`log_softmax` + `take_along_axis`), `entropy`
        (`-sum(p * log p)`), and a `probs` helper (`jax.nn.softmax`).
        `ACTION_INDEX_TO_BEHAVIOR` (SS11's fixed mapping) unchanged
        from P1 — still imports the real `BehaviorAction` enum from
        `src.environment.behavior_action`, never a hardcoded
        duplicate.
  - [x] Implemented `src/policies/ppo/loss.py`: `ppo_ratio`
        (`exp(new_log_prob - old_log_prob)`),
        `ppo_clipped_surrogate_loss` (clip epsilon applied to the
        ratio, `-mean(min(surr1, surr2))`, plus `clip_fraction`/
        `approx_kl` diagnostics), `value_loss` (MSE against returns),
        `entropy_bonus` (mean categorical entropy), and
        `ppo_total_loss` (combines all three with `value_coef`/
        `entropy_coef`, returns a full `info` dict for SS8's `ppo/*`
        metrics). Math follows the V-Max reference pattern
        (`vmax/agents/learning/reinforcement/ppo/ppo_factory.py::
        _make_loss_fn`) adapted to the discrete 4-way categorical
        head — no continuous Gaussian/Beta action distribution,
        observation extractor, reward design, or vectorized-env
        structure was ported (PPO_PLAN.md SS3).
  - [x] Implemented `src/policies/ppo/state.py`: `create_train_state`
        builds two independent `flax.training.train_state`-based
        `PPOTrainState`s (policy, value — no shared parameters, per
        SS7.2's Actor/Critic separation), each wrapping its own
        `optax.chain(optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate))` optimizer (matching the V-Max
        reference optimizer pattern). Bundled into a
        `PPOTrainingState` dataclass carrying both network modules and
        both train states — P4/P5 extract `.params`/`.opt_state` from
        these to populate `CheckpointPayload.policy_params`/
        `value_params`/`optimizer_state`.
  - [x] Implemented `src/policies/ppo/policy.py`: `PPOPolicy` wraps a
        policy network + its params; `act` (stochastic, returns
        `(action, log_prob)`) and `act_deterministic` (argmax) both
        delegate to `distribution.py`. No dependency on
        `src.environment` beyond consuming the frozen `BehaviorAction`
        enum via `distribution.ACTION_INDEX_TO_BEHAVIOR` — confirmed
        the PPO → Environment dependency direction stays
        one-directional (`grep -rn "policies\|ppo" src/environment/`
        finds zero import references, only unrelated prose/comment
        matches).
  - [x] Wrote 34 new behavioral tests in
        `tests/policies/test_ppo_core.py` covering every item in
        PPO_PLAN.md SS0.1/P3's required-tests list: 14D input handling
        (unbatched + batched), logits shape == 4, finite logits,
        softmax sums to 1, value output scalar/`(batch,)` + finite,
        sampled action always in `0..3`, deterministic inference ==
        argmax, stochastic sampling varies across 200 PRNG keys,
        finite log_prob, finite + non-negative entropy (plus a
        closed-form check: uniform logits → entropy == log(4)), the
        SS11 action-mapping regression importing the real
        `BehaviorAction` enum, 6 hand-checked PPO-ratio value pairs, 3
        clipping-behavior tests (upper-bound clip engages with the
        expected asymmetric surrogate-selection behavior, lower-bound
        clip engages, no-clip-when-in-range), scalar value loss with a
        hand-computed expected value, finite full PPO loss on a
        synthetic minibatch, finite and non-all-zero gradients of the
        full loss w.r.t. both policy and value params, an optimizer
        step changing parameters (checked independently for both the
        policy and the value train state), same-seed reproducibility
        (network init, logits, and categorical sampling),
        different-seeds-can-differ, and two `PPOPolicy` wrapper tests.
        Updated `tests/policies/test_imports.py` in place (now asserts
        real function/attribute presence instead of
        `NotImplementedError` stub checks that no longer apply, still
        6 collected items — same count as before, contents changed).
  - [x] Ran the new P3 tests together with the updated import tests
        (`tests/policies/`): **40 passed** in 17.55s (6 import checks
        + 34 new behavioral tests).
  - [x] Re-confirmed `jax.devices() == [CudaDevice(id=0)]` (GPU still
        visible) both before writing P3 code and again after all P3
        tests passed.
  - [x] Confirmed no other `pytest` process was running
        (`ps aux | grep pytest`) before launching the full regression
        suite.
  - [x] Ran the FULL existing regression suite (Phase 1-3 + P1 + P2 +
        P3 tests together) as a background process and waited
        synchronously for the actual process to exit (`kill -0 <pid>`
        polling loop, not a fixed-duration sleep and not a mid-run
        snapshot): **557 passed**, 0 failed, 0 skipped, 1802.23s
        (0:30:02), exit code 0 — matches 524 (P2 baseline) + 33 net
        new P3 tests exactly (`tests/policies/` grew from 7 collected
        items in P2 to 40 in P3: 6 updated import tests + 34 new
        behavioral tests = 40; 40 - 7 = 33; 524 + 33 = 557).
        Independently confirmed via `pytest tests/ --collect-only -q`
        → "557 tests collected". No other `pytest` process contended
        for the GPU during this run.
- [x] **P4 Rollout + GAE Integration**:
  - [x] Implemented `src/training/rollout.py::collect_episode_rollout`/
        `collect_rollout`: real rollout against the real
        `MergeEnvironment` (`downstream_mode="frenet_mpc"`, SS2),
        mirroring the exact pre-step `info_before["merge_committed"]`
        pattern P0 confirmed is already used by
        `full_split_evaluator.py::run_episode`
        (`is_policy_step = not info_before.get("merge_committed", False)`
        computed from the PREVIOUS reset()/step() call's info, never
        the current step's post-step info). On an auto-execution step
        the PPO policy is not called for action selection at all —
        `BehaviorAction.MERGE` is submitted directly. Every physical
        frame (regardless of `policy_mask`) becomes one `Transition`
        with the fixed SS0.1/P4 field set; reward is computed via P2's
        `MergeRewardWrapper.compute` fed the same pre-step
        `is_policy_step` flag plus the environment's own post-step
        `info["termination_reason"]` (SS5.1 — never re-derived).
  - [x] Implemented `src/training/gae.py::compute_gae`: standard
        backward-recursion GAE over the FULL physical trajectory
        (`gamma=0.99`/`gae_lambda=0.95` from `ppo_base.yaml`),
        correctly distinguishing true termination (bootstrap mask 0)
        from truncation (bootstrap mask 1, DOES bootstrap from
        `next_value`). Implemented `normalize_advantages_masked`/
        `masked_mean_std`: Actor-side normalization mean/std computed
        ONLY over `policy_mask == 1` positions via boolean-mask
        indexing before the reduction (SS7.2) — structurally
        impossible for a `policy_mask == 0` value to enter the
        statistic.
  - [x] Wired `src/training/trainer.py::build_training_batch`: real
        end-to-end rollout → GAE → masked-normalized PPO-ready batch
        dict (`observation`/`action`/`reward`/`next_observation`/
        `terminated`/`truncated`/`value`/`next_value`/`log_prob`/
        `policy_mask`/`advantages_raw`/`advantages`/`returns`), with a
        finite-value check across every batch field. `run_training`
        left as an updated `NotImplementedError` stub (P5 scope, per
        SS0.1/P4's explicit non-goal).
  - [x] Wrote 38 new P4 tests: `tests/training/test_gae.py` (11 —
        2 hand-computed multi-step GAE trajectories, terminal-no-
        bootstrap, truncation-does-bootstrap, mismatched-lengths
        guard, no-NaN/inf sweep, SS7.2 test A [masked-frame invariance]
        + its non-vacuous counterpart, manual-computation cross-check,
        all-masked-out guard, shape-mismatch guard),
        `tests/training/test_rollout.py` (8 — real 1-episode rollout
        to completion with the real PPO policy on `MAN_0001`; shape
        consistency; full-episode no-NaN/inf; multi-maneuver
        concatenation; the SS7.1 pre-step MERGE `policy_mask`
        regression test; the SS7.2 test D terminal-reward-propagation
        test; an integration-level SS7.2 test B; an SS11 action-mapping
        regression via rollout.py), `tests/training/test_trainer.py`
        (3 — end-to-end batch-building test, empty-list guard,
        `run_training` still raises). Updated `tests/training/
        test_imports.py` in place (stub checks → real-attribute
        assertions + a narrower `run_training`-still-a-stub check;
        still 6 collected items).
  - [x] Ran the new tests file-by-file, then `tests/training/
        tests/policies/ tests/rewards/` together (114 passed, 148.94s)
        as an intermediate sanity check before the full suite.
  - [x] Confirmed `git diff --stat` against `src/environment/`,
        `src/planning/`, `src/control/`, `src/scenarios/` is EMPTY —
        zero frozen Phase 1-3 files touched.
  - [x] Confirmed no other `pytest` process was running (`ps aux`/
        `pgrep`) before launching the full regression suite.
  - [x] Ran the FULL existing regression suite (Phase 1-3 + P1 + P2 +
        P3 + P4 tests together) as a background process and waited
        synchronously for the actual PID to exit (not a fixed-duration
        sleep, not a mid-run snapshot): **579 passed**, 0 failed, in
        1920.04s (0:32:00) — matches 557 (P3 baseline) + 22 net new P4
        tests exactly (11 gae + 8 rollout + 3 trainer = 22;
        `test_imports.py` unchanged at 6 collected items).
        Independently confirmed via `pytest tests/ --collect-only -q`
        → "579 tests collected". No other `pytest` process contended
        for the GPU during this run. `jax.devices()` re-confirmed
        `[CudaDevice(id=0)]` immediately after.
- [x] **P5 Smoke Training** (final Phase of the P0-P5 effort):
  - [x] Implemented the real multi-update PPO training loop in
        `src/training/trainer.py`: `run_update` (one PPO update —
        filters the batch to `policy_mask==1` rows for every Actor-side
        quantity per SS7.2, runs `ppo_epochs` inner-epoch passes over
        `num_minibatches` minibatches, computes policy gradients via
        `jax.value_and_grad` of a policy-loss closure built on P3's
        `ppo_clipped_surrogate_loss`/`entropy_bonus`, applies them via
        `policy_state.apply_gradients`; separately computes value
        gradients via `jax.value_and_grad` of P3's `value_loss` over
        the FULL trajectory per SS7.2's Critic scope, applies via
        `value_state.apply_gradients`) and `run_training` (the outer
        loop: collect a rollout via P4's `collect_rollout` -> P4's
        `build_training_batch` -> `run_update` -> log metrics ->
        advance `global_env_step`/`ppo_update_step`, repeated
        `num_updates` times; accepts starting step counters so a
        resumed run continues rather than restarts, per SS10).
  - [x] Verified (did not need to fix) `src/training/checkpoint.py`'s
        P1 pickle-based `save_checkpoint`/`load_checkpoint` against
        REAL JAX pytrees (Flax params as plain `dict`s in this repo's
        flax version, optax `opt_state` NamedTuples, JAX PRNG keys) —
        confirmed byte-identical round-trip via direct leaf-wise
        comparison AND via a real `.apply()` forward pass through the
        reloaded params producing identical logits/values. No format
        change (e.g. to orbax) was necessary; `dataclasses.asdict`
        correctly recurses through the payload without corrupting
        nested pytree structure.
  - [x] Wired `--resume` for real in `scripts/train_ppo.py` and
        `scripts/smoke_train_ppo.py`: loads a `CheckpointPayload`,
        restores both the policy and value `PPOTrainState`s (params +
        optimizer state) via `dataclasses.replace`, continues the JAX
        RNG stream from the checkpoint's own saved key (never
        re-derived from `--seed`), and continues `global_env_step`/
        `ppo_update_step` from the checkpoint rather than restarting at
        0.
  - [x] Completed `scripts/smoke_train_ppo.py` as the real Stage 1/2
        smoke-training entry point: resolves the deterministic
        canonical-TRAIN maneuver subset via `full_split_evaluator.
        load_maneuver_specs("train")`, builds/resumes a
        `PPOTrainingState`, calls `run_training`, logs to W&B
        (`WANDB_MODE=offline` works without auth), and saves a real
        checkpoint at the end (plus optional `--checkpoint-every`
        intermediate saves).
  - [x] Ran two REAL smoke-training stages against the real
        `MergeEnvironment` (`downstream_mode="frenet_mpc"`):
        **Stage 1** (fresh, `MAN_0001`/`MAN_0002`, 2 updates,
        `max_episode_steps=60`, seed 0): 61.1s wall-clock, both
        episodes reached real SUCCESS terminal outcomes each update,
        all metrics finite, checkpoint saved
        (`outputs/ppo_checkpoints/smoke_stage1_final.pkl`, 542445
        bytes). **Stage 2** (`--resume` from Stage 1's checkpoint,
        expanded to `MAN_0001..MAN_0004`, 3 more updates): 211.0s
        wall-clock, correctly printed `Resumed: global_env_step=70,
        ppo_update_step=2` and continued to
        `global_env_step=403, ppo_update_step=5` (not a restart), all
        metrics finite throughout, checkpoint saved
        (`outputs/ppo_checkpoints/smoke_stage2_final.pkl`). Full
        per-update numbers (policy/value loss, entropy, approx-KL,
        clip-fraction, grad-norm, action ratios) recorded in
        [SMOKE_TRAINING_REPORT.md](SMOKE_TRAINING_REPORT.md) §10 —
        reported honestly as pipeline-verification-scale numbers, not
        a performance claim (explicit P5 non-goal).
  - [x] Directly verified every SS0.1/P5 + §12 required-verification
        item (env reset, stochastic action sampling, action mapping,
        rollout, reward, GAE, finite PPO loss, an actual parameter
        update — policy AND value params provably differ before/after,
        both via dedicated tests and via a real checkpoint diff during
        interactive verification —, finite loss/gradients throughout,
        checkpoint save, checkpoint load, full
        save->load->resume->additional-update cycle [twice: a
        dedicated in-process test AND the real Stage1->Stage2 CLI
        invocation across two separate `python` process launches],
        W&B offline logging producing real files, no NaN/inf/crash/OOM
        across the whole run) — full evidence for each item recorded
        in [SMOKE_TRAINING_REPORT.md](SMOKE_TRAINING_REPORT.md) §12.
  - [x] Wrote 8 new P5-specific tests across 2 new files:
        `tests/training/test_run_training.py` (5 — `run_update`
        changes both policy AND value params with finite metrics, an
        all-masked-out batch raises `ValueError`, a full
        `run_training` end-to-end call against the real environment
        changes parameters with finite metrics throughout, starting
        `run_training` from nonzero step counters correctly continues
        them, and action-distribution stats are confirmed to reflect
        ONLY `policy_mask==1` rows on a single-decision-frame
        maneuver), `tests/training/test_checkpoint.py` (3 — real-JAX-
        pytree round-trip via leaf-wise comparison, round-tripped
        params are usable in a real `.apply()` forward pass producing
        identical output, and the full
        save->load->resume->additional-update cycle against a real
        rollout). Updated `tests/training/test_imports.py` in place
        (removed the now-obsolete `run_training`-raises-`NotImplementedError`
        stub check since the function is real; added a
        `run_update`-attribute-presence check; net -1 collected item).
        Updated `tests/training/test_trainer.py`'s old
        `test_run_training_still_raises_not_implemented` in place —
        replaced with `test_run_training_is_real_not_a_stub`, a minimal
        real-call smoke check (full behavioral coverage lives in the
        new `test_run_training.py`); net 0 collected-item change in
        that file.
  - [x] Ran the new P5 test files together with the updated import/
        trainer tests (`tests/training/test_checkpoint.py
        tests/training/test_run_training.py tests/training/test_imports.py`):
        **13 passed** in 121.54s. Then ran
        `tests/training/ tests/policies/ tests/rewards/ tests/tracking/`
        together as an intermediate sanity check: **126 passed** in
        277.52s.
  - [x] Confirmed `git diff --stat` against `src/environment/`,
        `src/planning/`, `src/control/`, `src/scenarios/` is EMPTY —
        zero frozen Phase 1-3 files touched, matching every prior
        Phase.
  - [x] Confirmed no other `pytest` process was running before
        launching the full regression suite.
  - [x] Ran the FULL existing regression suite (Phase 1-3 + P1 + P2 +
        P3 + P4 + P5 tests together) as a background process, watched
        via a `Monitor` until-loop polling the real PID (not a
        fixed-duration sleep, not a mid-run snapshot) to actual exit:
        **587 passed**, 0 failed, 0 errors, in 1948.37s (0:32:28),
        100% dots, zero `F`/`E` marks — matches 580 (P4's ACTUAL
        `--collect-only` count, verified via a `git worktree` checkout
        of the P4 completion SHA `fe8edc9` in this same environment;
        one more than the 579 figure this document's own P4 entries
        recorded at the time, a pre-existing off-by-one in that prior
        record this session did not introduce and is not silently
        correcting retroactively elsewhere in this file) + 7 net new
        P5 tests exactly (5 test_run_training.py + 3
        test_checkpoint.py + 0 test_trainer.py net + (-1)
        test_imports.py net = 7; 580 + 7 = 587). Independently
        confirmed via `pytest tests/ --collect-only -q` → "587 tests
        collected". No other `pytest` process contended for the GPU
        during this run.
        `jax.devices()` re-confirmed `[CudaDevice(id=0)]` immediately
        after.
  - [x] Wrote `docs/ppo/SMOKE_TRAINING_REPORT.md` (final architecture/
        results/reproduction-commands report for the whole P0-P5
        effort).

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
- P3 (implementation of existing P1 skeletons + one new test file; no
  existing Phase 1-3 file touched, no `src/environment/` file touched):
  - `src/policies/ppo/networks.py` (implemented; was a P1 stub —
    real `flax.linen.Module` policy/value networks)
  - `src/policies/ppo/distribution.py` (implemented; was a P1 stub —
    real categorical sampling/log_prob/entropy; `ACTION_INDEX_TO_BEHAVIOR`
    unchanged)
  - `src/policies/ppo/loss.py` (implemented; was a P1 stub — real PPO
    ratio, clipped surrogate loss, value loss, entropy bonus, combined
    total loss)
  - `src/policies/ppo/state.py` (implemented; was a P1 stub — real
    `create_train_state` with `flax.training.train_state` + `optax`
    Adam-with-clipping optimizers)
  - `src/policies/ppo/policy.py` (implemented; was a P1 stub — real
    `PPOPolicy.act`/`act_deterministic`)
  - `src/policies/ppo/__init__.py` (docstring updated to reflect P3
    completion; no code)
  - `tests/policies/test_ppo_core.py` (new, 34 collected test items)
  - `tests/policies/test_imports.py` (updated: stub-`NotImplementedError`
    checks replaced with real-attribute/function-presence assertions;
    6 collected, unchanged count from P1)
- P4 (implementation of existing P1 skeletons + new test files; no
  existing Phase 1-3 file touched, `git diff --stat` against
  `src/environment/`, `src/planning/`, `src/control/`,
  `src/scenarios/` confirmed empty):
  - `src/training/rollout.py` (implemented; was a P1 stub — real
    `collect_episode_rollout`/`collect_rollout` against the real
    `MergeEnvironment`, with the SS7.1 pre-step `policy_mask` pattern)
  - `src/training/gae.py` (implemented; was a P1 stub — real
    `compute_gae`/`normalize_advantages_masked`/`masked_mean_std`)
  - `src/training/trainer.py` (implemented `build_training_batch`;
    `run_training` remains a P1-style `NotImplementedError` stub,
    docstring updated to point at `build_training_batch`)
  - `tests/training/test_gae.py` (new, 11 collected test items)
  - `tests/training/test_rollout.py` (new, 8 collected test items)
  - `tests/training/test_trainer.py` (new, 3 collected test items)
  - `tests/training/test_imports.py` (updated: stub-`NotImplementedError`
    checks for `rollout`/`gae`/`trainer.build_training_batch` replaced
    with real-attribute assertions; `run_training`-stub check
    narrowed/kept; still 6 collected, unchanged count from P2/P3)
- P5 (implementation of the P4 `run_training` stub + new files; no
  existing Phase 1-3 file touched, `git diff --stat` against
  `src/environment/`, `src/planning/`, `src/control/`,
  `src/scenarios/` confirmed empty; `src/training/checkpoint.py` was
  inspected/tested but not modified -- its P1 pickle implementation
  was verified correct against real JAX pytrees, not changed):
  - `src/training/trainer.py` (implemented `run_update` and
    `run_training`; both were P1/P4 `NotImplementedError` stubs —
    now the real multi-update PPO training loop, per SS0.1/P5)
  - `scripts/train_ppo.py` (implemented: real config/env/train-state
    setup, real `--resume` wiring, calls `run_training`, saves a real
    checkpoint at the end — was a P1 skeleton that only echoed
    resolved config/seed)
  - `scripts/smoke_train_ppo.py` (implemented: real config/env/
    train-state/maneuver-subset setup, real `--resume` wiring, calls
    `run_training`, W&B config+metric logging, real checkpoint
    save/optional intermediate saves — was a P1 skeleton that only
    printed the resolved maneuver subset)
  - `tests/training/test_run_training.py` (new, 5 collected test items)
  - `tests/training/test_checkpoint.py` (new, 3 collected test items)
  - `tests/training/test_imports.py` (updated: removed the now-obsolete
    `run_training`-raises-`NotImplementedError` stub check; added a
    `run_update`-attribute-presence check; net -1 collected item)
  - `tests/training/test_trainer.py` (updated in place: replaced
    `test_run_training_still_raises_not_implemented` with
    `test_run_training_is_real_not_a_stub`; net 0 collected-item
    change)
  - `docs/ppo/SMOKE_TRAINING_REPORT.md` (new — final P0-P5 report)
  - `outputs/ppo_checkpoints/smoke_stage1_final.pkl`,
    `outputs/ppo_checkpoints/smoke_stage2_final.pkl` (new — real
    checkpoints from the two real smoke-training stages)
  - `wandb/offline-run-20260919_024059-dodmdn2z/`,
    `wandb/offline-run-20260919_024226-bsp9erzm/` (new — real offline
    W&B run files from the two smoke-training stages)

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
- New P3 tests alone (`tests/policies/`, includes updated P1 import
  tests + new behavioral tests): **40 passed** in 17.55s
- Full suite after P3 additions (Phase 1-3 + P1 + P2 + P3 tests
  together, no concurrent pytest process, real process exit
  confirmed): **557 passed**, 0 failed, 0 skipped, 1802.23s (0:30:02),
  exit code 0 — independently confirmed via
  `pytest tests/ --collect-only -q` → "557 tests collected"
  (524 pre-existing + 33 net new P3 = 557)
- GPU re-confirmed working after P3 changes: `jax.devices() ==
  [CudaDevice(id=0)]`
- New P4 tests: `tests/training/test_gae.py` (11 passed, 0.14s),
  `tests/training/test_rollout.py` (8 passed, 119.28s — real
  `MergeEnvironment` episodes, `frenet_mpc` mode), `tests/training/
  test_trainer.py` (3 passed, 21.46s). `tests/training/ tests/policies/
  tests/rewards/` together: **114 passed** in 148.94s (intermediate
  sanity check before the full suite)
- Full suite after P4 additions (Phase 1-3 + P1 + P2 + P3 + P4 tests
  together, no concurrent pytest process, real process exit
  confirmed): **579 passed**, 0 failed, in 1920.04s (0:32:00), exit
  code 0 — independently confirmed via
  `pytest tests/ --collect-only -q` → "579 tests collected"
  (557 pre-existing + 22 net new P4 = 579)
- GPU re-confirmed working after P4 changes: `jax.devices() ==
  [CudaDevice(id=0)]`
- Real PPO rollout/GAE/batch-building against the actual
  `MergeEnvironment` is now implemented and validated as of P4:
  `src/training/rollout.py`, `src/training/gae.py`, and
  `src/training/trainer.py::build_training_batch` are real. The actual
  multi-epoch parameter-UPDATE loop (`trainer.py::run_training`)
  remains a `NotImplementedError` stub — P5 scope, confirmed still true
  by the updated `tests/training/test_imports.py`/`test_trainer.py`
- New P5 tests: `tests/training/test_checkpoint.py`,
  `tests/training/test_run_training.py`, `tests/training/test_imports.py`
  together: **13 passed** in 121.54s. `tests/training/ tests/policies/
  tests/rewards/ tests/tracking/` together: **126 passed** in 277.52s
  (intermediate sanity check before the full suite)
- Full suite after P5 additions (Phase 1-3 + P1 + P2 + P3 + P4 + P5
  tests together, no concurrent pytest process, real process exit
  confirmed via a `Monitor` until-loop polling the real PID, not a
  fixed-duration sleep): **587 passed**, 0 failed, 0 errors, in
  1948.37s (0:32:28), 100% dots, zero `F`/`E` marks — independently
  confirmed via `pytest tests/ --collect-only -q` → "587 tests
  collected". Per-file P5 delta: +3 (`test_checkpoint.py`, new) +5
  (`test_run_training.py`, new) +0 (`test_trainer.py`, net) -1
  (`test_imports.py`, net) = **+7 new P5 tests** (checking out the P4
  completion commit `fe8edc9`'s own `tests/` tree against the current
  environment collects 580, not the "579" figure recorded for that
  commit at the time — a pre-existing one-test off-by-one in that
  prior session's own count, not introduced or corrected retroactively
  here; 580 + 7 = 587 reconciles exactly against what was actually
  measured this session; see SMOKE_TRAINING_REPORT.md §10 for detail)
- GPU re-confirmed working after P5 changes: `jax.devices() ==
  [CudaDevice(id=0)]`
- Real PPO multi-update training against the actual `MergeEnvironment`
  is now implemented and validated as of P5: `trainer.py::run_update`/
  `run_training` are real, both `scripts/train_ppo.py` and
  `scripts/smoke_train_ppo.py` are real working entry points, and two
  real smoke-training stages (fresh + resumed) ran successfully — see
  [SMOKE_TRAINING_REPORT.md](SMOKE_TRAINING_REPORT.md) for full
  results. **This is the final Phase — the P0-P5 effort is complete.**

### Pre-P6 hardening pass test results

- Targeted run over the 5 changed/new test files
  (`tests/training/test_gae.py`, `test_checkpoint.py`,
  `test_run_training.py`, `test_trainer.py`, and the new
  `test_pre_p6_hardening.py`): **65 passed**, 0 failed, 238 warnings
  (all pre-existing `optax.global_norm` deprecation warnings), in
  233.97s. `--collect-only` over the same 5 files independently
  confirms 65 collected (no silent skips).
- Full regression suite (solo, no concurrent GPU-contending process,
  verified by the orchestrating session): **629 passed**, 0 failed,
  in 2169.30s (0:36:09). `pytest --collect-only -q` over the full
  tree independently confirms **629 tests collected**, consistent
  with the reported pass count (baseline going into this branch was
  587; +42 net new/extended tests across the 7 fixes).
- Real smoke-training re-run against the current (post-fix) code —
  the original P0-P5 smoke evidence predated these fixes and lacked
  the new diagnostics entirely, so this phase re-ran fresh + resume
  smoke training: both runs exited 0, all metrics finite, the resumed
  run's `global_env_step`/`ppo_update_step` counters correctly
  continued (70/2 → 103/3) rather than resetting, and every new W&B
  diagnostic (`success_rate`, `collision_rate`, `offroad_rate`,
  `policy_decision_count`, `physical_step_count`, `exact_kl_mean/max`,
  `explained_variance`, `downstream/*`, `env_steps_per_sec`) appeared
  in the logged output with real values. Full numbers in
  [PRE_P6_REPORT.md](PRE_P6_REPORT.md) §5.
- GPU re-confirmed working after Pre-P6 changes:
  `jax.devices() == [CudaDevice(id=0)]`.
- `git diff --stat main -- src/environment/ src/planning/
  src/control/ src/scenarios/`: confirmed empty. No PPO-FIT/PPO-TUNE
  split created. Reward V0 values and PPO hyperparameters confirmed
  byte-identical to `main` (this was a correctness/instrumentation
  pass, not a tuning pass). See PRE_P6_REPORT.md §7 for the full
  scope-compliance verification.
- **This is the final phase of the Pre-P6 hardening effort — state is
  PRE-P6 HARDENING COMPLETE, WAITING FOR USER TUNING.**

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

Config loader implemented in P1 (`src/training/config.py`); the real
PPO algorithm that consumes these hyperparameters (`clip_epsilon`,
`value_coef`, `entropy_coef`, `learning_rate`, `max_grad_norm`) is now
implemented as of P3 (`src/policies/ppo/{loss,state}.py`) — wired
against synthetic inputs only in P3; real-rollout wiring is P4/P5
scope. Full contents of `configs/ppo/ppo_base.yaml` (verbatim from
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

Real smoke-training runs as of P5 (`WANDB_MODE=offline`, no auth
required): Stage 1 project `its-merge-ppo-smoke`, run dir
`wandb/offline-run-20260919_024059-dodmdn2z/`; Stage 2 (resumed), same
project, run dir `wandb/offline-run-20260919_024226-bsp9erzm/`. Both
logged the full §8 `REQUIRED_CONFIG_KEYS` config set plus every
per-update metric listed in
[SMOKE_TRAINING_REPORT.md](SMOKE_TRAINING_REPORT.md) §9, exited 0, and
produced real local files (`run-<id>.wandb`, `logs/debug.log`,
`logs/debug-internal.log`, `files/requirements.txt`, etc.). No online
W&B run has been started (not required through P5; `WANDB_MODE=offline`
is the supported no-network-required path per §8) — starting one is a
P6+/user decision.

## Checkpoint Path

Real checkpoints from P5's two smoke-training stages:
`outputs/ppo_checkpoints/smoke_stage1_final.pkl` (fresh run, 542445
bytes) and `outputs/ppo_checkpoints/smoke_stage2_final.pkl` (resumed
from Stage 1). Both contain the full §10 contract (policy/value params,
optimizer state, JAX RNG key, global env step, PPO update step, seed,
config snapshot, reward version, git SHA), verified to round-trip real
JAX pytrees byte-identically (`tests/training/test_checkpoint.py`) and
to support a real resume → additional-update cycle (both via a
dedicated test and via the real Stage1→Stage2 CLI invocation, in two
separate `python` process launches). See
[SMOKE_TRAINING_REPORT.md](SMOKE_TRAINING_REPORT.md) §11 for the exact
resume command.

## Last Command

```
pytest tests/ -q                     # P5: 587 passed, 0 failed, 1948.37s (0:32:28)
pytest tests/training/test_checkpoint.py tests/training/test_run_training.py tests/training/test_imports.py -v  # new P5 tests: 13 passed, 121.54s
pytest tests/training/ tests/policies/ tests/rewards/ tests/tracking/ -q   # intermediate sanity: 126 passed, 277.52s
pytest tests/ --collect-only -q      # "587 tests collected" (580 P4-baseline + 7 net new P5 = 587; see SMOKE_TRAINING_REPORT.md SS10)
WANDB_MODE=offline PYTHONPATH=. python scripts/smoke_train_ppo.py --checkpoint-path outputs/ppo_checkpoints/smoke_stage1_final.pkl   # Stage 1
WANDB_MODE=offline PYTHONPATH=. python scripts/smoke_train_ppo.py --resume outputs/ppo_checkpoints/smoke_stage1_final.pkl --max-maneuvers 4 --num-updates 3 --max-episode-steps 60 --checkpoint-path outputs/ppo_checkpoints/smoke_stage2_final.pkl   # Stage 2 (resume)
pytest tests/ -q                     # P4: 579 passed, 0 failed, 1920.04s (0:32:00)
pytest tests/training/test_gae.py tests/training/test_rollout.py tests/training/test_trainer.py -v  # new P4 tests
pytest tests/training/ tests/policies/ tests/rewards/ -q   # intermediate sanity: 114 passed, 148.94s
pytest tests/ --collect-only -q      # "579 tests collected" (557 + 22 net new = 579)
pytest tests/ -q                     # P3: 557 passed, 0 failed, 1802.23s (0:30:02)
pytest tests/policies/ -v            # new P3 tests + updated P1 import tests: 40 passed, 17.55s
pytest tests/ -q                     # P2: 524 passed, 0 failed, 1769.33s (0:29:29)
pytest tests/rewards/ tests/tracking/ tests/training/ tests/policies/ -q  # new P2 + P1 tests: 63 passed, 1.88s
pytest tests/ -q                     # P1: 489 passed, 0 failed, 1867.61s (0:31:07)
pytest tests/policies/ tests/rewards/ tests/training/ -v   # new P1 tests alone: 28 passed, 1.08s
```

## Known Issues

No contradiction found between PPO_PLAN.md and the current frozen
codebase during P0, P1, P2, P3, P4, or P5 — the entire P0-P5 effort
completed with zero blockers and zero deviations from the frozen
design. P5 specifically: the checkpoint contract (§10) was directly
implementable against real JAX pytrees using plain `pickle` (verified,
not assumed — see SMOKE_TRAINING_REPORT.md §14); the SS7.2 Actor-vs-
Critic `policy_mask` scope extended cleanly from P4's rollout/GAE level
into P5's update-loop level with no special-casing needed; `--resume`
composed cleanly with `run_training`'s `global_env_step`/
`ppo_update_step` starting-value parameters. No reward/hyperparameter/
network change was made anywhere in P5 (per its explicit non-goals).
The `info_before`/pre-step
pattern required by §7.1 is directly supported by the existing
`reset()`/`step()` API (both return `info` reflecting
`merge_committed` state as of that call), and an equivalent pattern is
already used in `src/environment/full_split_evaluator.py`; P4's
`collect_episode_rollout` uses this pattern directly, with the
dedicated pre-step MERGE `policy_mask` regression test
(`test_prestep_merge_policy_mask_regression`) confirming it produces
the correct `policy_mask == 1` on the exact decision frame where MERGE
is selected. P2's reward module was implemented as a pure fixed-table
lookup consuming the environment's own `info["termination_reason"]`
string with zero imports from `src.environment`/`waymax` — no blocker,
no deviation from §5.1 required. P3's discrete PPO core (networks,
categorical distribution, clipped-surrogate/value/entropy losses,
train state) was implemented entirely against synthetic/toy inputs
with no real `MergeEnvironment` dependency and no continuous
Gaussian/Beta action head anywhere. P4's rollout/GAE/batch-building
against the real `MergeEnvironment` (`downstream_mode="frenet_mpc"`)
required zero changes to any frozen Phase 1-3 file (`git diff --stat`
against `src/environment/`, `src/planning/`, `src/control/`,
`src/scenarios/` confirmed empty) — confirming the PPO -> Environment
dependency direction stays strictly one-directional throughout P0-P4.
No blocker, no deviation from PPO_PLAN.md §1/§2/§3/§5.1/§6/§7.1/§7.2/
§11 required.

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

## Small follow-up correction landed after Pre-P6 hardening

A narrow, additive bug fix landed on `feat/ppo-pre-p6` immediately
after the Pre-P6 hardening pass's own final commit (`d75b593`): Fix
7's `reward/terminal`/`reward/decision_cost` W&B breakdown was still
slightly wrong whenever a terminal/truncated outcome landed on a real
policy-decision step (e.g. SUCCESS `+1.0` combined with a `-0.01`
decision cost got fully counted as `reward/terminal` instead of split
`+1.0`/`-0.01`). Fixed by propagating `MergeRewardWrapper`'s own
already-correct per-step `last_terminal_component`/
`last_decision_cost_component` onto two new additive `Transition`
fields (`reward_terminal_component`/`reward_decision_cost_component`)
and summing those directly in `trainer.py`, instead of guessing from
`terminated`/`truncated`. Reward V0's values and PPO hyperparameters
are unchanged; 12 new tests added; full regression 641 passed / 0
failed (629 baseline + 12 new). See
[PRE_P6_REPORT.md §10](PRE_P6_REPORT.md#10-follow-up-fix-reward-component-logging-correctness-post-report)
for the full writeup. This does **not** change the end-state below —
it is a correction within the same phase.

## Next Exact Action

**None — the entire P0-P5 effort, AND the follow-up Pre-P6
correctness/instrumentation hardening pass (including the small
reward-component-logging correction above), are both COMPLETE.** All
items in [PPO_PLAN.md §12](PPO_PLAN.md#12-completion-checklist) hold
(see [SMOKE_TRAINING_REPORT.md](SMOKE_TRAINING_REPORT.md) for the P0-P5
itemized evidence and [PRE_P6_REPORT.md](PRE_P6_REPORT.md) for the
Pre-P6 hardening evidence) and the state is **PRE-P6 HARDENING
COMPLETE — WAITING FOR USER TUNING**. There is no further automated
PPO work queued in either effort. P6+ (reward/hyperparameter tuning, a
real TRAIN/TUNE split, longer training, canonical VAL evaluation,
FSM-vs-PPO comparison) is reserved for the user to decide on and run
manually after reviewing the real W&B diagnostics (now including the
Pre-P6 fixes' full diagnostic set) themselves — see
[HANDOFF.md](HANDOFF.md)'s NEXT OWNER ACTION.
