# P5 Smoke Training Report

Status: **P5 COMPLETE — WAITING FOR USER TUNING**. This is the final
report of the entire P0-P5 PPO effort on branch `feat/ppo-phase0-5`.
It documents what was built, exactly how it was verified, and the real
(unimpressive-by-design) smoke-training results. See
[PPO_PLAN.md](PPO_PLAN.md) for the durable plan this implements and
[PROGRESS.md](PROGRESS.md)/[HANDOFF.md](HANDOFF.md) for the live-state
trackers.

This report is committed at the P5 completion commit — the sixth and
final commit of the P0-P5 effort, immediately following `bcf2b4c`
(P0), `2f865b6` (P1), `565a7fe` (P2), `334e4c6` (P3), `fe8edc9` (P4) on
`feat/ppo-phase0-5`. "This commit" below refers to that commit.

---

## 1. Final file/directory structure (P1-P5)

```
src/
├── rewards/
│   ├── __init__.py
│   ├── merge_reward.py       # P2: Reward V0 fixed-table lookup
│   └── reward_wrapper.py     # P2: per-episode reward accumulation wrapper
├── policies/
│   ├── __init__.py
│   └── ppo/
│       ├── __init__.py
│       ├── networks.py       # P3: PolicyNetwork / ValueNetwork (flax.linen)
│       ├── distribution.py   # P3: categorical sampling/log_prob/entropy + action-index mapping
│       ├── policy.py         # P3: PPOPolicy (act / act_deterministic)
│       ├── loss.py           # P3: PPO ratio, clipped surrogate, value loss, entropy bonus
│       └── state.py          # P3: PPOTrainState / PPOTrainingState / create_train_state
├── training/
│   ├── __init__.py
│   ├── config.py             # P1: load_ppo_config / load_reward_config
│   ├── seeding.py            # P1: make_seed_state / split_key
│   ├── checkpoint.py         # P1 skeleton, P5 real: CheckpointPayload + save/load
│   ├── rollout.py            # P4: collect_episode_rollout / collect_rollout
│   ├── gae.py                # P4: compute_gae / normalize_advantages_masked / masked_mean_std
│   └── trainer.py            # P4 build_training_batch, P5 run_update / run_training
└── tracking/
    ├── __init__.py
    └── wandb_logger.py       # P2: WandbLogger (online/offline)

configs/
├── reward/
│   └── merge_reward_v0.yaml  # Reward V0 values (P1, unchanged since)
└── ppo/
    ├── ppo_base.yaml         # baseline network/hyperparameters (P1, unchanged since)
    └── ppo_smoke.yaml        # inherits ppo_base.yaml + smoke.{max_maneuvers,num_updates,max_episode_steps}

scripts/
├── train_ppo.py              # P1 skeleton, P5 real: general PPO training entry point (+ --resume)
└── smoke_train_ppo.py        # P1 skeleton, P5 real: P5 smoke-training entry point (+ --resume)

tests/
├── rewards/           (test_imports.py, test_merge_reward.py)
├── policies/          (test_imports.py, test_ppo_core.py)
├── tracking/          (test_wandb_logger.py)
└── training/          (test_imports.py, test_config.py, test_gae.py, test_rollout.py,
                         test_trainer.py, test_checkpoint.py, test_run_training.py)
```

No file under the frozen `src/environment/`, `src/planning/`,
`src/control/`, `src/scenarios/` directories was ever touched across
P0-P5 (verified again for this commit — see §7).

## 2. What was ported from V-Max vs. implemented fresh (PPO_PLAN.md §3)

**Ported (pattern/structure only, re-implemented in this repo's own
code, never copied verbatim or imported as a dependency):**

- PPO clipped surrogate objective, GAE, advantage normalization,
  entropy regularization, value loss, minibatch update, gradient
  clipping, and the optimizer/train-state pattern — adapted from
  `vmax/agents/learning/reinforcement/ppo/ppo_factory.py` and
  `vmax/config/algorithm/ppo.yaml` to this project's discrete 4-way
  categorical action head.
- The general Actor/Critic-separate-train-state and
  `optax.chain(clip_by_global_norm, adam)` optimizer pattern
  (`vmax/agents/pipeline/`).

**Explicitly NOT ported (per §3):**

- V-Max's continuous Gaussian/Beta action head — this project's PPO
  outputs a 4-way categorical distribution over
  `BehaviorAction.{KEEP,FOLLOW,MERGE,STOP}` only.
- V-Max's observation extractor — this project's PPO consumes the
  frozen 14D `MergeEnvironment` observation as-is
  (`src/environment/observation_builder.py`).
- V-Max's own reward design — Reward V0 (§5) is this project's own
  fixed terminal + decision-cost table, sourced from the frozen
  `MergeEnvironment`'s own termination signal, never V-Max's reward.
- V-Max's vectorized Waymax environment structure — this project's
  rollout runs one `MergeEnvironment` episode at a time
  (`src/training/rollout.py`), matching the existing
  `full_split_evaluator.py` pattern rather than a vectorized-env
  design.

## 3. Reward V0 as implemented

```
R = terminal_outcome + decision_cost
```

| `TerminationReason` | Terminal value |
|---|---|
| `SUCCESS` | +1.0 |
| `FAILURE_COLLISION` | -1.0 |
| `FAILURE_OFFROAD` | -1.0 |
| `TRUNCATION_HORIZON` (="TIMEOUT" in §5's table) | -0.5 |
| `NONE` / no termination yet | 0.0 |

Decision cost: `-0.01` on a real PPO decision step
(`is_policy_step=True`), `0.0` on an auto-executed MERGE-commitment
step (`is_policy_step=False`). Both components come purely from a
fixed-table lookup in `src/rewards/merge_reward.py::compute_reward`,
keyed by the caller-supplied `termination_reason` string (the
environment's own `info["termination_reason"]`, never re-derived) and
`is_policy_step` flag (decided pre-step per §7.1, never post-step).
`compute_reward` contains zero imports from `src.environment` or
`waymax` (verified by a code-level test in
`tests/rewards/test_merge_reward.py`), satisfying §5.1's
source-of-truth rule. `src/rewards/reward_wrapper.py::MergeRewardWrapper`
is the rollout-loop-friendly wrapper used by `rollout.py`, accumulating
per-episode `reward/terminal`/`reward/decision_cost`/`reward/total`
sums. Reward weights were never tuned during P0-P5 (unchanged from the
values fixed at P1).

## 4. PPO architecture as implemented

Network (from `configs/ppo/ppo_base.yaml`, verbatim, not tuned):

- Policy network: `14 → 256 → 64 → 32 → 4 logits`, `tanh` activation
  between every hidden Dense layer, no shared trunk with the value
  network (`src/policies/ppo/networks.py::PolicyNetwork`).
- Value network: `14 → 256 → 64 → 32 → 1`, same `tanh` pattern,
  squeezed to a scalar (unbatched) / `(batch,)` (batched)
  (`ValueNetwork`).
- Categorical action distribution over 4 discrete actions
  (`src/policies/ppo/distribution.py`): `sample_action`
  (`jax.random.categorical`), `deterministic_action` (`jnp.argmax`),
  `log_prob` (`log_softmax` + `take_along_axis`), `entropy`
  (`-sum(p log p)`).

Hyperparameters (from `ppo_base.yaml`, unchanged through P5):

```
learning_rate  = 3.0e-4
gamma          = 0.99
gae_lambda     = 0.95
clip_epsilon   = 0.2
value_coef     = 0.5
entropy_coef   = 0.01
max_grad_norm  = 0.5
ppo_epochs     = 4
num_minibatches = 4   (ppo_base.yaml) / 1 (ppo_smoke.yaml — too few
                        smoke transitions to usefully split further)
```

Two independent `flax.training.train_state.TrainState`s (policy,
value), each with its own `optax.chain(optax.clip_by_global_norm(max_grad_norm),
optax.adam(learning_rate))` optimizer — no shared parameters between
Actor and Critic (`src/policies/ppo/state.py::create_train_state`).

## 5. Rollout structure

`src/training/rollout.py::collect_episode_rollout` runs one
`MergeEnvironment` episode (`downstream_mode="frenet_mpc"`, per §2) to
completion (`terminated` or `truncated`, or `max_steps`), mirroring the
exact pre-step pattern already used by
`src/environment/full_split_evaluator.py::run_episode`:

```python
is_policy_step = not info_before.get("merge_committed", False)
```

computed from the **previous** `reset()`/`step()` call's `info`,
never the current step's post-step `info` (§7.1). On an
auto-execution step, the PPO policy network is not called for action
selection — `BehaviorAction.MERGE` is submitted directly (a `log_prob`
is still computed, purely so `Transition.log_prob` stays a
well-defined finite field; it is never consumed downstream since
`policy_mask==0` rows are filtered out of every Actor-side
computation). Every physical frame becomes one `Transition`
(`observation`, `action`, `reward`, `next_observation`, `terminated`,
`truncated`, `value`, `next_value`, `log_prob`, `policy_mask`, plus
diagnostics `episode_id`/`maneuver_id`/`step_index`) — no frame is
ever dropped at collection time (§7.2's Critic scope requires the full
trajectory). `collect_rollout` concatenates `collect_episode_rollout`
across multiple maneuvers into one flat transition list.

## 6. GAE semantics

`src/training/gae.py::compute_gae` implements standard backward-
recursion GAE (Schulman et al. 2015, eq. 16) over the **full physical
trajectory** (every frame, regardless of `policy_mask` — §7.2),
`gamma=0.99`/`gae_lambda=0.95`, correctly distinguishing:

- true termination (`terminated=True`) → bootstrap mask `0`, no value
  flows past a terminal state
- truncation (`truncated=True`) → bootstrap mask `1`, DOES bootstrap
  from `next_value`

`normalize_advantages_masked` computes the Actor-side normalization
mean/std using **only** `policy_mask == 1` positions, via boolean-mask
indexing *before* the reduction runs — masked-out values are
structurally excluded, not merely down-weighted (§7.2 test A holds by
construction). `src/training/trainer.py::build_training_batch` wires
rollout → GAE → masked normalization into one flat batch dict
(`observation`, `action`, `reward`, `next_observation`, `terminated`,
`truncated`, `value`, `next_value`, `log_prob`, `policy_mask`,
`advantages_raw` [unmasked, full-trajectory, diagnostics only],
`advantages` [masked-normalized, Actor scope], `returns`
[full-trajectory GAE, Critic target]) and raises `ValueError` on any
non-finite value or empty transition list.

## 7. policy_mask semantics — implemented and verified (§7.1/§7.2)

**§7.1 (decision timing):** guarded by
`tests/training/test_rollout.py::test_prestep_merge_policy_mask_regression`
— a scripted always-MERGE policy against the real `MergeEnvironment`
on maneuver `MAN_CAUSALITY` (a single-transition `lane_chain`, so MERGE
at decision frame 0 commits immediately). The test asserts the VERY
FIRST transition (where MERGE is selected) has `policy_mask == 1`, and
every transition strictly after it has `policy_mask == 0` — directly
guarding against the reversed post-step bug §7.1 describes. **PASSED.**

**§7.2 (Actor vs. Critic scope):**

- Test A (masked-frame invariance on the advantage-normalization
  statistic) — `tests/training/test_gae.py`. **PASSED.**
- Test B (masked-frame invariance on the real PPO loss/entropy/
  approx-KL/clip-fraction, not just the raw arrays) —
  `tests/training/test_trainer.py::test_masked_frames_do_not_affect_real_ppo_loss_statistics`.
  Corrupts only `policy_mask==0` rows' observation/log_prob/advantage
  values and confirms the `policy_mask==1`-filtered PPO loss/entropy/
  KL/clip-fraction stay bit-identical. **PASSED.**
- Test C (Critic uses the full trajectory) — implicit in
  `build_training_batch`'s construction (GAE/returns always computed
  over every transition, never filtered) and in
  `run_update`'s value-loss pass, which trains on `batch["observation"]`/
  `batch["returns"]` unfiltered (full length), never the
  `policy_mask==1`-filtered Actor subset.
- Test D (terminal reward propagation across masked frames) —
  `tests/training/test_rollout.py::test_terminal_reward_propagates_to_merge_decision_frame`.
  Confirms a real terminal SUCCESS/COLLISION reward's GAE credit
  propagates backward through intervening `policy_mask==0`
  auto-execution frames to the `policy_mask==1` MERGE decision frame.
  **PASSED.**
- P5's `run_update` (`src/training/trainer.py`) implements this scope
  operationally: `_filter_actor_rows` filters `observation`/`action`/
  `log_prob`/`advantages` to `policy_mask==1` rows before any policy
  loss/entropy/gradient computation; the value-loss pass uses the
  unfiltered full-length `batch["observation"]`/`batch["returns"]`.
  `tests/training/test_run_training.py::test_run_training_action_stats_only_count_policy_mask_one_rows`
  additionally confirms the `action/*_ratio` metrics reflect only
  `policy_mask==1` rows (not smeared across MERGE-dominated
  auto-execution frames). **PASSED.**

## 8. W&B logging fields actually emitted

`src/tracking/wandb_logger.py::WandbLogger` wraps `wandb.init`/
`wandb.config.update`/`wandb.log`. `log_config` enforces (raises
`ValueError` if missing) every one of §8's `REQUIRED_CONFIG_KEYS`:
`git_sha`, `reward_version`, `seed`, `learning_rate`, `gamma`,
`gae_lambda`, `clip_epsilon`, `entropy_coef`, `value_coef`,
`batch_size`, `ppo_epochs`, `network_layers` — all logged by both
`scripts/train_ppo.py` and `scripts/smoke_train_ppo.py`, plus extra
run-identifying fields (`maneuver_ids`, `num_updates`, and, for
`train_ppo.py`, `resumed_from`).

`run_training` (`src/training/trainer.py`) logs, per PPO update, via
`wandb_logger.log_metrics(metrics, step=ppo_update_step)`:

```
train/episode_return, train/episode_length,
reward/terminal, reward/decision_cost, reward/total,
ppo/policy_loss, ppo/value_loss, ppo/entropy, ppo/approx_kl,
ppo/clip_fraction, ppo/grad_norm,
action/keep_ratio, action/follow_ratio, action/merge_ratio, action/stop_ratio
```

**Known limitation (honest, not a bug):** `log_metrics` deliberately
accepts *any* metric name rather than enforcing all of §8's
`MINIMUM_METRICS` be present on every call (that list — e.g.
`train/success_rate`, `downstream/intervention_rate`,
`ppo/explained_variance`, `runtime/env_steps_per_sec` — becomes
available incrementally across phases, per `wandb_logger.py`'s own
docstring). `run_training` as implemented in P5 does **not** currently
emit `train/success_rate`/`train/collision_rate`/`train/offroad_rate`/
`train/timeout_rate`, `ppo/explained_variance`,
`downstream/intervention_rate` and related downstream-diagnostic
metrics, or `runtime/env_steps_per_sec`. Pipeline correctness (the P5
goal) does not depend on these; a future phase wiring the full §8
metric list would be a straightforward, low-risk addition (all the
underlying data — `info["intervention_rate"]` etc., episode outcome
counts — is already available from the rollout `Transition`s and
environment `info` dict), left for the user to decide whether/when to
add.

Real offline W&B run directories from the actual smoke runs (see §9)
exist on disk at (not committed to git — see §11):

```
wandb/offline-run-20260919_024059-dodmdn2z/   (Stage 1)
wandb/offline-run-20260919_024226-bsp9erzm/   (Stage 2, --resume)
```

each containing `run-<id>.wandb`, `run-<id>.wandb.syncstate`,
`logs/debug.log`, `logs/debug-internal.log`, `logs/debug-core.log`,
`files/requirements.txt`, `files/wandb-summary.json`, `files/config.yaml`.

## 9. Smoke training: exact conditions and real results

Both stages used `scripts/smoke_train_ppo.py`, config
`configs/ppo/ppo_smoke.yaml`, seed `0`, `downstream_mode="frenet_mpc"`,
`WANDB_MODE=offline`, maneuver subset drawn from the deterministic
sorted canonical-TRAIN split (`data/manifests/phase2_dataset_split.csv`,
never a PPO-FIT/TUNE split).

**Stage 1** (`--max-maneuvers 2 --num-updates 2 --max-episode-steps 60`,
fresh init, no `--resume`):

```
Smoke maneuver subset (2 maneuvers): ['MAN_0001', 'MAN_0002']
num_updates=2, max_episode_steps=60
Smoke training finished in 61.1s
global_env_step=70, ppo_update_step=2
update 0: episode_return=0.885, episode_length=17.5, reward/terminal=2.0,
          reward/decision_cost=-0.230, reward/total=1.77,
          policy_loss=-0.0917, value_loss=0.1455, entropy=1.2610,
          approx_kl=0.0894, clip_fraction=0.565, grad_norm=8.656,
          action ratios: keep=0.304, follow=0.261, merge=0.087, stop=0.348
update 1: episode_return=0.950, episode_length=17.5, reward/terminal=2.0,
          reward/decision_cost=-0.100, reward/total=1.90,
          policy_loss=-0.0722, value_loss=0.0218, entropy=1.2470,
          approx_kl=0.0612, clip_fraction=0.200, grad_norm=2.644,
          action ratios: keep=0.300, follow=0.000, merge=0.200, stop=0.500
```

Checkpoint saved: `outputs/ppo_checkpoints/smoke_stage1_final.pkl`
(542,445 bytes). W&B run:
`wandb/offline-run-20260919_024059-dodmdn2z/`.

**Stage 2** (`--max-maneuvers 4 --num-updates 3 --max-episode-steps 60
--resume outputs/ppo_checkpoints/smoke_stage1_final.pkl`, a
**separate process invocation**, genuinely resuming Stage 1's saved
state):

```
Smoke maneuver subset (4 maneuvers): ['MAN_0001', 'MAN_0002', 'MAN_0003', 'MAN_0004']
num_updates=3, max_episode_steps=60
Resuming from checkpoint: outputs/ppo_checkpoints/smoke_stage1_final.pkl
Resumed: global_env_step=70, ppo_update_step=2
Smoke training finished in 211.0s
global_env_step=403, ppo_update_step=5
update 0: episode_return=0.725, episode_length=27.5, reward/terminal=3.0,
          reward/decision_cost=-0.100, reward/total=2.90,
          policy_loss=-0.0517, value_loss=0.0104, entropy=1.2361,
          approx_kl=-0.0120, clip_fraction=0.200, grad_norm=1.561,
          action ratios: keep=0.500, follow=0.100, merge=0.400, stop=0.000
update 1: episode_return=0.7325, episode_length=27.75, reward/terminal=3.0,
          reward/decision_cost=-0.070, reward/total=2.93,
          policy_loss=-0.0222, value_loss=0.0093, entropy=1.2267,
          approx_kl=-0.0382, clip_fraction=0.000, grad_norm=1.815,
          action ratios: keep=0.143, follow=0.000, merge=0.571, stop=0.286
update 2: episode_return=0.695, episode_length=28.0, reward/terminal=3.0,
          reward/decision_cost=-0.220, reward/total=2.78,
          policy_loss=-0.0229, value_loss=0.0049, entropy=1.1752,
          approx_kl=0.0446, clip_fraction=0.136, grad_norm=1.086,
          action ratios: keep=0.409, follow=0.000, merge=0.182, stop=0.409
```

`ppo_update_step` continued 2→5 (never reset to 0) and
`global_env_step` continued 70→403 across the resume boundary,
confirming resume is a real continuation, not a restart. Checkpoint
saved: `outputs/ppo_checkpoints/smoke_stage2_final.pkl` (542,468
bytes; md5 `87d66baa4080b409ca6c790254479a28`, differs from Stage 1's
`eadae104a5362aa78e7f02da60168dc4` — genuinely different parameters).
W&B run: `wandb/offline-run-20260919_024226-bsp9erzm/`.

**Honest interpretation (not a performance claim — P5 is not a
performance phase per §0.1/§12):** across both stages, episode returns
stayed in the `0.7-0.95` range (dominated by the `reward/terminal`
component, i.e. episodes mostly ended in SUCCESS or a positive
terminal outcome on these particular maneuvers/short horizons — a
property of the specific TRAIN maneuvers picked and the small step
budget, not a claim about general performance), `ppo/entropy` stayed
around `1.18-1.26` (near `log(4) ≈ 1.386`, i.e. the policy had not yet
collapsed to a near-deterministic distribution over only 5 updates —
expected for this few gradient steps), and action ratios varied update
to update with no fixed pattern — consistent with a freshly-initialized
policy taking a handful of real gradient steps, not with any
convergence claim. No collision/offroad/timeout outcome happened to
occur in this particular 5-update, 4-maneuver smoke run; that is a
report of what was observed on this specific tiny sample, not evidence
about collision/offroad rates in general.

## 10. Full regression suite results

**587 passed, 0 failed, 80 warnings (all `optax.global_norm` deprecation
warnings — `optax.global_norm is deprecated in favor of optax.tree.norm`
— not test failures), in 1948.37s (0:32:28)**, run as a solo background
`pytest tests/ -q` process (PID 495630, no other `pytest` process
running concurrently), the real process exit waited for and the final
summary line read directly from the completed log. Independently
cross-checked via `pytest tests/ --collect-only -q` → **"587 tests
collected"**. Per-file P5 delta: `tests/training/test_checkpoint.py`
(new file, 3 collected), `tests/training/test_run_training.py` (new
file, 5 collected), `tests/training/test_trainer.py` (4 collected, net
0 — one stub-check test replaced by
`test_run_training_is_real_not_a_stub`), `tests/training/test_imports.py`
(5 collected, -1 — the now-obsolete `run_training`-still-a-stub check
was removed). Net: 3 + 5 + 0 + (-1) = **7 new P5 tests**. (Directly
checking out the P4 completion commit `fe8edc9`'s exact `tests/` tree
against the current environment collects 580, not the 579 figure
`PROGRESS.md`/`HANDOFF.md` recorded for that commit at the time — a
pre-existing one-test off-by-one in that prior session's own count,
not something this P5 session introduced or is correcting
retroactively; 580 + 7 = 587 reconciles exactly against the number
actually measured in this session.)

`git diff --stat -- src/environment/ src/planning/ src/control/ src/scenarios/`
is confirmed **empty** — zero frozen Phase 1-3 files touched across
the entire P0-P5 effort, including this final P5 commit.

## 11. Checkpoint location and verified resume command

Checkpoints live under `outputs/ppo_checkpoints/` (gitignored — see
§12), a plain-`pickle` dump of `src/training/checkpoint.py::CheckpointPayload`
(policy params, value params, optimizer state for both networks, JAX
PRNG key, `global_env_step`, `ppo_update_step`, seed, config snapshot,
reward version, git SHA — full §10 contract, not weights-only).

Verified resume command (Stage 2, run as reported above):

```bash
PYTHONPATH=. WANDB_MODE=offline python scripts/smoke_train_ppo.py \
  --max-maneuvers 4 --num-updates 3 --max-episode-steps 60 \
  --resume outputs/ppo_checkpoints/smoke_stage1_final.pkl
```

Also verified at the unit-test level, independent of any script, in
`tests/training/test_checkpoint.py::test_full_save_load_resume_additional_update_cycle`
(save → load in a fresh `CheckpointPayload`/training state → one more
real `run_training` update, asserting: an additional parameter change
occurred; `ppo_update_step` continues rather than resets;
`global_env_step` increases past its checkpointed value).

## 12. `wandb/` run directories — not committed

`wandb/offline-run-*/` directories are local W&B run artifacts (binary
`.wandb` files, debug logs), not source. They are referenced by path
above (§8, §9) for traceability but are **not** tracked in git — a
`wandb/` entry was added to `.gitignore` in this commit. Anyone
wanting to inspect a run can `wandb sync <run dir>` locally, or re-run
the exact commands in §9/§11 to regenerate equivalent runs.

## 13. Known limitations

- W&B metric coverage is a strict subset of §8's `MINIMUM_METRICS` (see
  §8) — success/collision/offroad/timeout rates and downstream
  intervention diagnostics are not yet logged by `run_training`,
  though the underlying data already exists in the rollout output.
- Checkpointing uses plain `pickle`, not a JAX-native format like
  `orbax` (already installed in this env). This was flagged as an
  acceptable choice back in P1 and re-confirmed sufficient in P5 —
  every SS10 field round-trips exactly (verified against real Flax/
  optax pytrees in `tests/training/test_checkpoint.py`), but a future
  phase moving to very large models/distributed training might prefer
  `orbax` for its sharding/async-save support.
- The smoke runs are, by design, tiny (2-4 maneuvers, 2-5 total
  updates) — no statement about convergence, generalization, or
  success/collision rate at scale can be drawn from them; that is
  explicitly reserved for the user's own tuning work starting at P6.
- `run_update`'s per-update `ppo/grad_norm` metric is the mean of all
  per-minibatch global-norm values across both the policy and value
  optimizer steps combined (`np.mean(policy_grad_norms + value_grad_norms)`),
  not separate `ppo/policy_grad_norm`/`ppo/value_grad_norm` fields — a
  minor granularity simplification, not a correctness issue (both
  norms are individually finite, as required).

## 14. Reproduction commands

Full regression suite (from repo root, inside the `its-merge` conda
env):

```bash
source /home/autonav/miniconda3/etc/profile.d/conda.sh && conda activate its-merge
PYTHONPATH=. pytest tests/ -q
```

Stage-1-equivalent smoke run (fresh, no resume):

```bash
PYTHONPATH=. WANDB_MODE=offline python scripts/smoke_train_ppo.py \
  --max-maneuvers 2 --num-updates 2 --max-episode-steps 60 \
  --checkpoint-path outputs/ppo_checkpoints/my_stage1.pkl
```

Stage-2-equivalent resumed run:

```bash
PYTHONPATH=. WANDB_MODE=offline python scripts/smoke_train_ppo.py \
  --max-maneuvers 4 --num-updates 3 --max-episode-steps 60 \
  --resume outputs/ppo_checkpoints/my_stage1.pkl \
  --checkpoint-path outputs/ppo_checkpoints/my_stage2.pkl
```

General entry point (`scripts/train_ppo.py`, same pipeline, defaults
to `configs/ppo/ppo_base.yaml` — still constrained through P5 to the
small deterministic canonical-TRAIN subset per its own module
docstring):

```bash
PYTHONPATH=. WANDB_MODE=offline python scripts/train_ppo.py \
  --max-maneuvers 2 --num-updates 2 --max-episode-steps 30
```

## 15. Final Git SHA

This report is committed at the P5 completion commit on
`feat/ppo-phase0-5` — the sixth and final commit of the P0-P5 effort
(`test(ppo): validate smoke training pipeline (P5)`), immediately
following `bcf2b4c`/`2f865b6`/`565a7fe`/`334e4c6`/`fe8edc9`. See
`git log` on this branch for the exact SHA.
