# P6 Experiment Policy

Baseline config: [`configs/ppo/ppo_p6_baseline.yaml`](../../configs/ppo/ppo_p6_baseline.yaml) —
byte-identical to the Pre-P6-validated `configs/ppo/ppo_base.yaml`, no
tuning applied. Branch: `exp/ppo-p6-tuning` (created from `main` at
`066b123`, the Pre-P6 hardening merge). See
[`docs/ppo/PRE_P6_REPORT.md`](PRE_P6_REPORT.md) for what was verified
before this baseline was frozen, and
[`docs/ppo/PPO_PLAN.md`](PPO_PLAN.md) for the original P0-P5 design.

## Rule 1 — The baseline config does not change

`ppo_p6_baseline.yaml` is a fixed reference point. Do not edit it once
a baseline run has been logged against it — every subsequent experiment
needs a stable thing to compare against.

## Rule 2 — One axis per experiment

Each experiment config changes exactly one thing relative to the
baseline (one hyperparameter, or the reward config, but not both).
Keep experiment configs under `configs/ppo/experiments/`, e.g.:

```
configs/ppo/experiments/
  p6_lr_1e4.yaml
  p6_lr_5e4.yaml
  p6_entropy_005.yaml
  p6_entropy_02.yaml
```

These files are created only when you've decided to run that specific
experiment — none exist yet as of this workspace setup.

## Rule 3 — Don't change Reward and PPO hyperparameters together

If a run's results suggest the reward shaping might be off, isolate
that from hyperparameter changes. Change one axis, rerun, compare.

## Rule 4 — What to look at in W&B

At minimum, check these before drawing any conclusion from a run:

```
train/episode_return

train/success_rate
train/collision_rate
train/offroad_rate
train/timeout_rate

reward/terminal
reward/decision_cost
reward/total

ppo/policy_loss_mean
ppo/value_loss_mean
ppo/entropy_mean

ppo/approx_kl_mean
ppo/exact_kl_mean
ppo/exact_kl_max

ppo/clip_fraction_mean
ppo/explained_variance

action/keep_ratio
action/follow_ratio
action/merge_ratio
action/stop_ratio

train/policy_decision_count
train/physical_step_count

downstream/intervention_rate
downstream/planner_infeasible_rate
downstream/collision_blocked_rate
downstream/controller_failure_rate

runtime/env_steps_per_sec
```

`reward/terminal` and `reward/decision_cost` are now correctly
separated (see `PRE_P6_REPORT.md` §10) — a SUCCESS episode's
`reward/terminal` should read exactly `+1.0` per occurrence, not a
value contaminated by decision cost.

## Rule 5 — Canonical VAL is not for tuning

Canonical VAL evaluation is reserved for a final, already-decided
model — never used to pick between experiments. No TRAIN-internal
tune-split protocol exists yet; that is a decision for you to make
later, once baseline results are in, not something pre-built here.

## Baseline run command (example only — not executed by this workspace setup)

```
PYTHONPATH=. python scripts/train_ppo.py \
  --ppo-config configs/ppo/ppo_p6_baseline.yaml \
  --max-maneuvers <YOUR_CHOSEN_COUNT> \
  --num-updates <YOUR_CHOSEN_COUNT> \
  --max-episode-steps <YOUR_CHOSEN_COUNT> \
  --checkpoint-path outputs/ppo_checkpoints/p6_baseline_seed0.pkl
```

The training budget (`--max-maneuvers`/`--num-updates`/
`--max-episode-steps`) is intentionally left for you to choose — this
setup does not pick numbers on your behalf. For a quick pipeline
sanity check (not a real baseline run), something like
`--max-maneuvers 2 --num-updates 1 --max-episode-steps 60` mirrors the
scale used in prior Pre-P6 smoke runs. Set `WANDB_MODE=offline` if you
don't want to log online yet.
