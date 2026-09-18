# PPO Session Handoff

Read this whenever resuming PPO work after a context loss. See
[PROGRESS.md](PROGRESS.md) for the detailed live-state tracker and
[PPO_PLAN.md](PPO_PLAN.md) for the durable scope/architecture reference.

## Resume protocol

When the user says **"계속 시작해"** (or equivalent), do this first,
without asking questions:

1. `git status`
2. `git branch`
3. `git log -n 10`
4. Read [docs/ppo/PROGRESS.md](PROGRESS.md)
5. Read this file (`docs/ppo/HANDOFF.md`)

Then resume exactly from the point recorded in "Next Owner Action" below.
Do **not** restart a Phase that PROGRESS.md already marks complete.

Unified branch-creation order (see also
[PPO_PLAN.md § 9](PPO_PLAN.md#9-git-conventions)):

1. Check `git status` / `git branch` / `git log`
2. Create `feat/ppo-phase0-5` (if absent)
3. Begin P0 Baseline Audit
4. All further PPO changes/doc commits happen on that branch, never on
   `main`

---

## What's completed so far

Nothing beyond documentation scaffolding. `docs/ppo/PPO_PLAN.md`,
`PROGRESS.md`, `HANDOFF.md`, `EXPERIMENT_LOG.md` have been created
(originally under `docs/phase4/`, relocated to `docs/ppo/` to avoid
confusion between internal PPO P0–P5 stage numbering and the project's
overall Phase numbering). No PPO source code, configs, or tests exist
yet. No branch has been created yet — repo is still on `main` at
`c5d2c1d2ea197cd247b3bc2e4f127b3c36d1a990`.

This session's edits also strengthened the plan itself (still
documentation-only, no implementation started): precise pre-step timing
for `policy_mask` (§7.1 in PPO_PLAN.md), explicit Actor-vs-Critic scope
for `policy_mask` (§7.2), reward-source-of-truth rules tying reward
strictly to the frozen environment's own termination signals (§5.1), a
full checkpoint/resume contract (§10), a unified branch-creation order
(§9), and an action-mapping regression test requirement (§11).

A subsequent revision pass:

- Corrected the §7.2 unit-test description. It previously implied that
  changing masked (`policy_mask == 0`) auto-execution frames' advantages
  *should* change the Actor normalization statistics and that GAE/return
  should stay unchanged if such frames are removed — both wrong. The
  corrected rule: masked-frame advantage values/counts must **never**
  affect Actor-side normalization mean/std or the normalized decision
  advantages (invariance, not sensitivity); and GAE/return legitimately
  changes if a real physical auto-execution frame is deleted, since that
  alters the actual temporal trajectory — so no test asserts GAE/return
  staying fixed under frame deletion. See §7.2 in PPO_PLAN.md for the
  corrected tests A–D.
- Added §0.1 to PPO_PLAN.md: a full per-Phase (P0–P5) execution plan
  with Goal / Tasks / Required tests / Completion criteria / Explicit
  non-goals for each Phase. **PPO_PLAN.md is now the complete durable
  Source of Truth for the whole P0–P5 effort** — a session recovering
  from context loss can read PPO_PLAN.md §0.1 plus PROGRESS.md's current
  position and know exactly what remains, without needing prior
  conversation history.
- Made the `TerminationReason` enum → Reward V0 mapping explicit in the
  P2 section of §0.1 (`SUCCESS`, `FAILURE_COLLISION`, `FAILURE_OFFROAD`,
  `TRUNCATION_HORIZON`, `NONE` — confirmed against
  [termination.py:48-53](../../src/environment/termination.py#L48-L53)),
  and reconfirmed no new "TIMEOUT" enum value is introduced; the reward
  table's "TIMEOUT" row corresponds to `TRUNCATION_HORIZON`.
- Reconfirmed no PPO-FIT/PPO-TUNE dataset split exists through P5; the
  Smoke Training subset selection (`--max-maneuvers`,
  `--maneuver-ids`, `--seed`) is for pipeline verification only.
- No P0 audit, branch creation, dependency install, source/config/test
  implementation, or `pytest`/training execution was performed in this
  revision pass — documentation only.

## Last successful test

None run yet as part of this PPO effort.

## Failed tests

None run yet.

## Problems found

None yet.

## Recent commits (PPO-related)

None yet.

## Running processes

None.

## Latest checkpoint

None.

## Next command to run

```
git checkout -b feat/ppo-phase0-5
```
then begin the P0 Baseline Audit steps listed in
[PROGRESS.md § Next Exact Action](PROGRESS.md#next-exact-action).

## Next file to modify

None yet — P0 is an audit phase (read-only investigation), no source
files are modified until P1.

---

## NEXT OWNER ACTION

Wait for the user's final approval of the plan documents
(`docs/ppo/PPO_PLAN.md`, `PROGRESS.md`, `HANDOFF.md`,
`EXPERIMENT_LOG.md`). Do **not** start P0, do not install dependencies,
do not create source/config/test files, and do not create the
`feat/ppo-phase0-5` branch until that approval is given.

Once approved, start **Phase P0 — Baseline Audit** per
[PPO_PLAN.md § 0](PPO_PLAN.md#0-scope-of-this-work) and the detailed
steps in [PROGRESS.md](PROGRESS.md), following the branch-creation order
in [PPO_PLAN.md § 9](PPO_PLAN.md#9-git-conventions). Do not skip ahead to
P1–P5. Do not perform any reward/hyperparameter tuning, W&B sweeps, full
training, or FSM-vs-PPO comparisons — those remain out of scope through
P5 and are reserved for the user to do manually afterward.
