# PPO Phase P0–P5 Plan (Reference Copy)

Status: **Not started.** This document is a durable reference for the
scope, architecture, and guardrails of the PPO work. It is written once
at the start and should rarely need edits — live state goes in
[PROGRESS.md](PROGRESS.md), session handoff goes in [HANDOFF.md](HANDOFF.md),
run-by-run records go in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

Base SHA at plan creation: `c5d2c1d2ea197cd247b3bc2e4f127b3c36d1a990` (branch `main`).

## 0. Scope of this work

This effort implements PPO Phases **P0 through P5 only**:

- P0. Baseline Audit
- P1. PPO Foundation
- P2. Reward V0 + W&B Tracking
- P3. Discrete PPO Core
- P4. Rollout + GAE Integration
- P5. Smoke Training

When P5 succeeds, work stops. The following are explicitly **out of scope**
and must not be started automatically:

- Reading W&B learning curves and revising the reward function
- Hyperparameter tuning or W&B Sweeps
- Reward V1 design
- Full training runs / multi-seed training
- Canonical VAL evaluation
- FSM vs PPO final performance comparison

P6+ is reserved for the user to run manually. This work only needs to leave
behind a config/script/logging/checkpoint/resume structure that lets the
user start P6 without further scaffolding.

## 0.1 Per-Phase execution plan (durable Source of Truth)

This section is the detailed execution and acceptance-criteria reference
for each Phase. A future session recovering from context loss should be
able to read this section alone (plus [PROGRESS.md](PROGRESS.md) for
current position) and know exactly what remains to be done and how each
Phase will be judged complete. It is written once and should not need to
be re-derived from conversation history.

### P0 — Baseline Audit

**Goal:** Confirm the current state of the frozen Phase 1–3 research
foundation before any PPO code is added.

**Tasks:**

- `git status` / `git branch` / `git log`
- Record Python / JAX / jaxlib / NumPy / Waymax versions
- Confirm GPU visibility via `jax.devices()`
- Review `requirements.txt`
- Review dataset manifests (`data/manifests/`)
- Confirm 14D observation (`OBSERVATION_DIM`, field order)
- Confirm `BehaviorAction` semantics
- Confirm termination semantics
- Confirm MERGE commitment behavior
- Confirm `downstream_mode="frenet_mpc"` is the path in use
- Confirm intervention diagnostics are intact
- Run the full existing `pytest` suite
- Reproduce a short deterministic rollout (reproducibility sanity check)
- Take a short runtime/throughput measurement

**Required tests:** the existing regression suite only (no new PPO tests
in P0).

**Completion criteria:**

- Existing regression tests PASS
- GPU/JAX/Waymax stack confirmed working
- Baseline SHA, versions, and throughput recorded in PROGRESS.md

**Explicit non-goals:** no modification or tuning of any existing Phase
1–3 behavior. P0 is read-only investigation.

### P1 — PPO Foundation

**Goal:** Add PPO infrastructure scaffolding without touching or
breaking the existing environment.

**Tasks:**

- Create `src/rewards/`, `src/policies/ppo/`, `src/training/`,
  `src/tracking/` directories (see §4 file structure)
- Create `configs/reward/`, `configs/ppo/`
- Create `scripts/` entry-point skeletons
- Implement config loader (PPO config + reward config)
- Implement seed handling
- Implement checkpoint skeleton (structure only; full save/load lands
  in P5, contract defined in §10)

**Dependency rule:** do not upgrade the existing JAX/jaxlib/CUDA/Waymax
stack. In particular, never run `pip install --upgrade jax`. Only add
`flax` / `optax` / `wandb` if required, chosen to be compatible with the
JAX version recorded in P0. After any install: re-run `jax.devices()`
and the existing regression suite to confirm nothing broke.

**Important constraint carried through P0–P5:** no PPO-FIT / PPO-TUNE
split of the dataset is created. The Smoke Training entry point (P5)
only supports selecting a small deterministic subset of the canonical
TRAIN split via flags such as `--max-maneuvers`, `--maneuver-ids`,
`--seed`. This subset is for pipeline verification only — it is never
used for tuning or performance evaluation.

**Required tests:** import checks, config-load checks; existing
regression suite still PASS.

**Completion criteria:**

- Imports, config loading, seed handling, and checkpoint skeleton work
- Existing regression tests PASS

**Explicit non-goals:** no TRAIN/TUNE split; no training loop yet; no
reward implementation yet (that's P2).

### P2 — Reward V0 + W&B Foundation

**Goal:** Implement the minimal Reward V0 and the experiment-tracking
foundation.

**Reward V0** (values unchanged from §5), with the exact frozen-enum
mapping made explicit:

```
TerminationReason.SUCCESS            -> +1.0
TerminationReason.FAILURE_COLLISION  -> -1.0
TerminationReason.FAILURE_OFFROAD    -> -1.0
TerminationReason.TRUNCATION_HORIZON -> -0.5
TerminationReason.NONE               -> 0.0
```

(`TerminationReason` is defined in
[termination.py:48-53](../../src/environment/termination.py#L48-L53) —
these are the only five values it defines; no new "TIMEOUT" enum value
is introduced anywhere. `TRUNCATION_HORIZON` is what the reward table in
§5 calls "TIMEOUT".)

Reward code consumes `terminated`, `truncated`, and
`info["termination_reason"]` as the sole source of truth (§5.1) — it
does not re-derive termination itself.

**Tasks:**

- Implement reward module mapping the enum table above, plus decision
  cost (`-0.01` real decision step / `0.0` auto-execution step, decided
  pre-step per §7.1)
- Implement W&B logger supporting both online and offline
  (`WANDB_MODE=offline`) modes
- Wire minimum config logging (git_sha, reward_version, seed,
  hyperparameters — see §8)

**Required tests:**

- SUCCESS = +1, COLLISION = -1, OFFROAD = -1, TIMEOUT/TRUNCATION_HORIZON
  = -0.5, NONE = 0
- real decision step = -0.01, auto-committed step = 0 decision cost
- total reward = terminal + decision cost, correctly summed
- no NaN/inf in any reward path
- W&B smoke run produces logged config + at least one metric point

**Completion criteria:**

- Reward unit tests PASS
- W&B online-or-offline smoke run succeeds
- Existing regression tests PASS

**Explicit non-goals:** no reward weight tuning, no new reward terms
(TTC/gap/comfort/etc.), no W&B sweep, no curve-based reward revision.

### P3 — Discrete PPO Core

**Goal:** Validate the PPO algorithm itself, independent of the
environment.

**Implement:** policy network, value network, categorical distribution,
stochastic action sampling, deterministic inference, log_prob, entropy,
PPO ratio, clipped surrogate objective, value loss, entropy
regularization, gradient clipping, optimizer / train state.

**Required tests:**

- 14D input handled correctly
- output logits shape == 4
- finite logits
- probability sum == 1
- action range 0..3
- deterministic inference == argmax
- stochastic sampling works
- finite log_prob
- finite entropy
- correct PPO ratio computation
- correct clipping behavior
- value output is scalar
- finite PPO loss
- finite gradients
- optimizer update changes parameters
- same-seed reproducibility sanity check
- action-mapping regression (§11):
  `0->KEEP, 1->FOLLOW, 2->MERGE, 3->STOP`

**Completion criteria:**

- All PPO core unit tests PASS
- JAX GPU confirmed working
- Existing regression tests PASS

**Explicit non-goals:** no real environment training in P3 — this Phase
validates the algorithm in isolation only.

### P4 — Rollout + GAE Integration

**Goal:** Connect PPO to the real `MergeEnvironment`.

**Rollout transition fields (minimum):** `observation`, `action`,
`reward`, `next_observation`, `terminated`, `truncated`, `value`,
`next_value`, `log_prob`, `policy_mask`. Optional diagnostics:
`episode_id`, `maneuver_id`, `step_index`.

**GAE:** computed over the full physical trajectory (all frames,
regardless of `policy_mask`) — see §7.2. Truncation and true termination
are distinguished for bootstrapping.

**policy_mask:** decided before `env.step()`, per §7.1. The frame where
PPO directly selects MERGE is `policy_mask=1`; only the subsequent
auto-executed MERGE frames are `policy_mask=0`. Actor computations use
`policy_mask==1` only; Critic/GAE computations use all frames — per §7.2.

**Required tests:**

- handcrafted trajectory GAE matches hand-computed values
- terminal bootstrap handled correctly
- truncation handled correctly
- advantage normalization (Actor-side, masked-frame-excluded)
- rollout shape consistency
- no NaN/inf
- an actual 1-episode rollout against the real `MergeEnvironment`
- pre-step MERGE `policy_mask` regression test (§7.1): the decision
  frame that selects MERGE is `policy_mask=1`, not `0`
- Actor-mask invariance tests A and B from §7.2
- terminal reward propagation test D from §7.2 (MERGE decision →
  auto-execution frames → terminal SUCCESS/COLLISION → GAE credit flows
  back to the MERGE decision)
- action-mapping regression (§11)

**Completion criteria:**

- A real `MergeEnvironment` rollout → GAE → PPO-compatible training
  batch is produced end-to-end
- All required tests PASS
- Existing regression tests PASS

**Explicit non-goals:** no actual multi-update training loop yet (that's
P5); no dataset TUNE split.

### P5 — Smoke Training

**Goal:** Verify the end-to-end training pipeline runs correctly on the
real `MergeEnvironment` — **not** to obtain good PPO performance.

**Stage 1:** deterministic 1–3 maneuvers from canonical TRAIN, minimal
number of updates, just to exercise every stage of the pipeline once.

**Stage 2:** only if Stage 1 succeeds — expand to a somewhat larger
deterministic maneuver subset and a few more updates, still bounded by
what's needed to demonstrate correctness, not performance. Exact numbers
may be chosen pragmatically based on runtime; the point is pipeline
correctness, not convergence.

**Explicitly forbidden in P5:**

- long full-TRAIN training
- millions of environment steps
- W&B-curve-based reward revision
- hyperparameter tuning
- a TRAIN/TUNE split
- W&B sweeps
- canonical VAL evaluation
- FSM vs PPO final comparison/conclusions

**Verify:**

- env reset
- action sampling
- action mapping
- rollout
- reward computation
- GAE
- PPO loss
- parameter update (backprop)
- finite loss / finite gradients
- checkpoint save
- checkpoint load
- resume (save → load → resume → additional update, per §10)
- W&B logging (metrics + config)
- no NaN/inf, no crash, no OOM/leak across the smoke run

The checkpoint contract in §10 applies unchanged — full state, not just
weights.

**Completion criteria:** all of the P5 completion items in §12 hold, and
the state is left as **P5 COMPLETE — WAITING FOR USER TUNING**.

**Explicit non-goals:** performance tuning of any kind. A low success
rate, odd action distribution, or low entropy at the end of P5 is not a
failure by itself — a pipeline that does not run end-to-end, produces
NaN/inf, or has a demonstrable implementation bug is the failure mode
this Phase actually guards against.

## 1. Target architecture

```
WOMD Merge Scenario
        ↓
MergeEnvironment
        ↓
14D Observation
        ↓
PPO Policy
        ↓
Categorical Distribution
        ↓
KEEP / FOLLOW / MERGE / STOP
        ↓
BehaviorExecutor
        ↓
Frenet Planner
        ↓
LTV-MPC
        ↓
Waymax
        ↓
Reward / Next State
        ↓
Rollout
        ↓
GAE
        ↓
PPO Update
        ↓
W&B Logging
```

PPO **never** outputs acceleration/steering directly — only a 4-way
categorical `BehaviorAction`. Dependency direction is strictly
`PPO -> Environment`; the environment must not know whether the policy
in control is FSM or PPO.

All PPO runs must use `downstream_mode="frenet_mpc"`.

## 2. Frozen invariants (must not change)

These are inherited from Phase 1–3 and protected by regression tests
throughout P0–P5:

- `OBSERVATION_DIM == 14` and observation field order
  ([observation_builder.py:68](../../src/environment/observation_builder.py#L68))
- `BehaviorAction` semantics: `KEEP=0, FOLLOW=1, MERGE=2, STOP=3`
  ([behavior_action.py:64-67](../../src/environment/behavior_action.py#L64-L67))
- decision window / `decision_start_frame` semantics
- MERGE commitment behavior
- merge success / collision / offroad / timeout termination semantics
- episode horizon
- Frenet Planner and LTV-MPC internals
- common downstream pipeline and intervention diagnostics
- FSM baseline has zero PPO-specific dependency
- PPO has no privileged information beyond the 14D observation

## 3. V-Max reference scope

Reference `vmax/agents/learning/reinforcement/ppo/`,
`vmax/config/algorithm/ppo.yaml`, `vmax/config/network/base.yaml`,
`vmax/simulator/wrappers/reward.py`, `vmax/agents/pipeline/` for:
clipped surrogate objective, actor/critic structure, GAE, advantage
normalization, entropy regularization, value loss, minibatch update,
gradient clipping, optimizer setup, checkpoint concept, training metrics.

Do **not** port: Gaussian/Beta continuous action heads, direct
acceleration/steering output, V-Max's observation extractor, V-Max's full
reward design, or V-Max's vectorized Waymax environment structure. This
research stays on 14D observation → 4-class categorical PPO →
KEEP/FOLLOW/MERGE/STOP.

## 4. File structure convention

New PPO code only — no relocation of existing Phase 1–3 files.

```
src/
├── rewards/
│   ├── __init__.py
│   ├── merge_reward.py
│   └── reward_wrapper.py
├── policies/
│   ├── __init__.py
│   └── ppo/
│       ├── __init__.py
│       ├── networks.py
│       ├── distribution.py
│       ├── policy.py
│       ├── loss.py
│       └── state.py
├── training/
│   ├── __init__.py
│   ├── rollout.py
│   ├── gae.py
│   ├── trainer.py
│   └── checkpoint.py
└── tracking/
    ├── __init__.py
    └── wandb_logger.py

configs/
├── reward/
│   └── merge_reward_v0.yaml
└── ppo/
    ├── ppo_base.yaml
    └── ppo_smoke.yaml

scripts/
├── train_ppo.py
└── smoke_train_ppo.py

tests/
├── rewards/
├── policies/
└── training/
```

## 5. Reward V0 (fixed through P5)

```
R = terminal_outcome + decision_cost
```

Terminal:

| Outcome     | Value |
|-------------|-------|
| SUCCESS     | +1.0  |
| COLLISION   | -1.0  |
| OFFROAD     | -1.0  |
| TIMEOUT     | -0.5  |
| NONTERMINAL | 0.0   |

Decision cost: `-0.01` on a real PPO decision step, `0.0` on an
auto-executed MERGE-commitment step.

No TTC/gap/comfort/progression/intervention/lane-deviation/imitation/
action-bonus terms in V0. Reward weights are not tuned in P0–P5.

### 5.1 Reward source of truth (must not be re-derived)

The reward module MUST NOT recompute `SUCCESS` / `COLLISION` / `OFFROAD` /
`TIMEOUT` itself. It consumes the frozen `MergeEnvironment`'s own
`terminated`, `truncated`, and `info["termination_reason"]` as the sole
source of truth, and simply maps that outcome to the fixed values above.
The existing environment's termination priority and semantics
(inherited from Phase 1–3) are consumed as-is, never re-implemented.

Explicitly forbidden inside reward code:

- a merge-success detector
- an overlap/collision detector
- an offroad detector
- an episode-timeout detector

Re-implementing any of these risks silent semantic divergence from the
frozen Phase 1–3 environment — the reward would then be scoring a
different notion of "success" than what the environment actually
enforces.

Decision cost (`-0.01` / `0.0`) is likewise applied based on the
**pre-step** `is_policy_step` / `policy_mask` value for that step (see
§7.1), never derived from post-step state.

## 6. PPO baseline architecture & hyperparameters

Policy network: `14 → 256 → 64 → 32 → 4 logits` (tanh activations).
Value network: `14 → 256 → 64 → 32 → 1` (tanh activations).

Baseline hyperparameters (V-Max-derived, not tuned in P0–P5):

```
learning_rate  = 3e-4
gamma          = 0.99
gae_lambda     = 0.95
clip_epsilon   = 0.2
value_coef     = 0.5
entropy_coef   = 0.01
```

## 7. MERGE commitment / policy_mask

Rollout transitions carry a `policy_mask`: `1` on frames where PPO
actually chose an action, `0` on auto-executed MERGE-commitment frames.

### 7.1 Decision timing (critical)

`policy_mask` MUST be decided **before** `env.step()` is called, based on
whether PPO actually chose the action for the *current* step — never from
the `info` dict returned *after* `env.step()`. Reading
`info_after["merge_committed"]` to decide the current step's mask is a
bug: it would cause the very frame where PPO selects `MERGE` to be
mislabeled `policy_mask = 0`, because `merge_committed` only flips to
`True` as a result of that same step.

Recommended pattern:

```python
info_before = current_info  # info from *before* this step

is_policy_step = not info_before["merge_committed"]
policy_mask = 1 if is_policy_step else 0

if is_policy_step:
    action = ppo_policy(observation)
else:
    action = BehaviorAction.MERGE

next_obs, reward, terminated, truncated, info_after = env.step(action)
```

The frame where PPO directly selects `MERGE` is `policy_mask = 1`. Only
the *subsequent* physical frames, where the environment is auto-executing
the MERGE commitment (`info_before["merge_committed"] == True` already,
before this step even runs), are `policy_mask = 0`.

A regression test must guard against the reversed (post-step) bug: given
a scripted episode where PPO selects `MERGE` at some decision frame, the
mask at that exact frame must be `1`, and only frames strictly after it
(while auto-execution is in progress) may be `0`.

### 7.2 Scope of policy_mask across Actor vs. Critic computations

`policy_mask == 0` (auto MERGE-execution) frames are excluded from every
**Actor**-side computation:

- PPO policy loss
- entropy (and entropy loss)
- approximate KL
- clip fraction
- action-distribution statistics
- action counts
- Actor-side advantage normalization statistics — the mean/std used to
  normalize advantages must be computed **only** over `policy_mask == 1`
  frames, not the full physical rollout

`policy_mask == 0` frames remain fully included in every **Critic /
temporal** computation:

- reward
- value prediction
- return
- GAE propagation
- terminal signal
- value loss

So a terminal SUCCESS/COLLISION reward still flows back through GAE to
the decision frame that triggered MERGE commitment, even though the
auto-execution frames contribute nothing to the policy/entropy/KL/advantage-
normalization statistics. In short:

```
Physical rollout (all frames)
    ├── Critic / GAE: uses ALL frames
    └── Actor / PPO loss, entropy, KL, clip-frac,
        action stats, advantage-norm mean/std: policy_mask == 1 ONLY
```

Required unit tests (P4):

A. **Actor normalization invariance to masked frames.** Holding the set
   of `policy_mask == 1` decision advantages fixed, changing the
   advantage *values* of `policy_mask == 0` auto-execution frames must
   not change the Actor-side normalization mean/std, nor the resulting
   normalized decision advantages. (Masked frames must never leak into
   the statistic — not "the statistic changes when masked frames
   change," which would itself be the bug.)
B. **Actor-loss invariance to masked frames.** `policy_mask == 0` frames
   must not affect policy loss, entropy, approximate KL, clip fraction,
   or action-distribution statistics.
C. **Critic uses the full trajectory.** GAE/return computation uses the
   full physical trajectory (all frames, masked or not) — this is not
   optional and is not itself something to vary in a test; removing a
   real auto-execution frame changes the temporal trajectory and is
   expected to change GAE/return, so no test asserts GAE/return staying
   the same after deleting a physical frame.
D. **Terminal reward propagation across masked frames.** A terminal
   SUCCESS/COLLISION reward reached after one or more `policy_mask == 0`
   auto-execution frames following a `policy_mask == 1` MERGE decision
   must propagate back through GAE to that MERGE decision frame's
   advantage.

## 8. W&B logging (P0–P5 usage only: config + metrics + smoke verification — no sweeps, no tuning from curves)

Minimum metrics: `train/episode_return`, `train/success_rate`,
`train/collision_rate`, `train/offroad_rate`, `train/timeout_rate`,
`train/episode_length`, `reward/terminal`, `reward/decision_cost`,
`reward/total`, `ppo/policy_loss`, `ppo/value_loss`, `ppo/entropy`,
`ppo/approx_kl`, `ppo/clip_fraction`, `ppo/explained_variance`,
`ppo/grad_norm`, `action/keep_ratio`, `action/follow_ratio`,
`action/merge_ratio`, `action/stop_ratio`,
`downstream/intervention_rate`, `downstream/planner_infeasible_rate`,
`downstream/collision_blocked_rate`, `downstream/controller_failure_rate`,
`runtime/env_steps_per_sec`, optionally `safety/min_ttc`,
`safety/mean_ttc` (diagnostic only — not a reward term).

Each run config logs: `git_sha`, `reward_version`, `seed`,
`learning_rate`, `gamma`, `gae_lambda`, `clip_epsilon`, `entropy_coef`,
`value_coef`, `batch_size`, `ppo_epochs`, `network_layers`.

If W&B auth is unavailable, use `WANDB_MODE=offline` rather than skipping
logging entirely.

## 9. Git conventions

Startup order (unified across PROGRESS.md and HANDOFF.md):

1. Check `git status` / `git branch` / `git log`
2. Create `feat/ppo-phase0-5` (if it does not already exist)
3. Begin P0 Baseline Audit
4. All subsequent PPO-related changes and documentation commits happen
   on `feat/ppo-phase0-5`

No PPO implementation changes are made on `main`. Any currently
uncommitted PPO documentation is carried into the feature branch rather
than committed to `main`. One meaningful commit per completed Phase. No
merges to `main`, no destructive git operations, no history rewrites or
edits to existing frozen commits without explicit user permission.

## 10. Checkpoint / resume contract

A checkpoint is not considered "resume-capable" if it only saves model
weights. Every checkpoint must save, at minimum:

- policy parameters
- value parameters
- optimizer state
- JAX PRNG key / RNG state
- global environment step count
- PPO update step count
- seed
- the complete PPO config (or a config snapshot)
- reward version
- Git SHA

Additional metadata may be stored alongside as needed.

P5 must verify the full cycle actually works end-to-end:

```
save → load → resume → additional PPO update
```

Where feasible, add a sanity test that reloading the same checkpoint with
the same RNG/config and performing the next update produces consistent
(reproducible) results — this is the strongest signal that resume is
implemented correctly rather than merely present.

## 11. Action-mapping regression test

P3/P4 test plans include a fixed regression test asserting the
categorical action index → `BehaviorAction` mapping never drifts:

```
0 -> BehaviorAction.KEEP
1 -> BehaviorAction.FOLLOW
2 -> BehaviorAction.MERGE
3 -> BehaviorAction.STOP
```

This guards against the PPO categorical output index silently disagreeing
with the frozen `BehaviorAction` `IntEnum` semantics.

## 12. Completion checklist

Work in this effort is done only when all of the following hold, and the
state is left as **P5 COMPLETE — WAITING FOR USER TUNING**:

- [ ] P0 Baseline Audit
- [ ] P1 PPO Foundation
- [ ] P2 Reward V0
- [ ] P2 W&B logger
- [ ] P3 Categorical PPO
- [ ] P3 Actor/Critic
- [ ] P3 PPO clipped loss
- [ ] P4 rollout buffer
- [ ] P4 GAE
- [ ] P4 policy_mask / MERGE commitment (pre-step decision timing, §7.1)
- [ ] P4 Actor-vs-Critic policy_mask scope test (§7.2)
- [ ] P4 action-mapping regression test (§11)
- [ ] P5 real environment smoke training
- [ ] PPO parameter update verified
- [ ] checkpoint save/load verified (full contract, §10)
- [ ] resume verified (save → load → resume → additional update)
- [ ] W&B metric logging verified
- [ ] no NaN/inf
- [ ] regression tests PASS
- [ ] documentation complete
- [ ] Git commits organized

See [docs/ppo/PROGRESS.md](PROGRESS.md) for live status against this list.
