# Capstone Midterm Figures

Four figures for the "04. 현재까지 진행 내용" section of the capstone
midterm report. All figures are generated from real repository
data/code only -- no synthetic numbers or placeholder trajectories.

Run individually (`python scripts/figures/capstone_midterm/fig01_....py`,
each prefixed with `PYTHONPATH=.` from the repo root and, for fig04,
`WANDB_MODE=offline` to avoid any network call) or all at once via
`generate_all.py`. Output goes to `outputs/capstone_midterm_figures/`.

## fig01_dataset_scenario

**Source:**
- `configs/dataset.yaml` (real WOMD TRAIN shards) via `MergeEnvironment.reset`
- `data/manifests/` canonical maneuver manifest (via
  `src.environment.full_split_evaluator.load_decision_dataset_maneuver_specs`)
- `outputs/diagnostics/train_diag64_seed20260928.txt` (the project's own
  64 canonical Fixed-TRAIN64 maneuver IDs, reused here to pick a
  known-good case)
- Front/Rear vehicle identity: `src.scenarios.scenario_features.find_target_lane_front_rear`
  -- the SAME function the production 14D observation builder calls,
  never a manual reselection.

**Scenario ID:** `training_tfexample.tfrecord-00000-of-01000#378__t18__237_456`

Shows the main (source) lane, merge (target) lane, Ego, and the real
target-lane Front vehicle at the maneuver's `merge_start_frame`. No
Rear vehicle exists at this frame for this maneuver (confirmed by
scanning all 64 canonical maneuvers: `rear_id` is `None` for every one
of them at reset time in this dataset -- not a bug, simply absent at
that frame) -- no Rear box is drawn.

## fig02_rule_based_baseline

**Source:**
- `src/environment/fsm_policy.py` (`FsmPolicy`, the project's real,
  implemented rule-based baseline -- confirmed via repository audit
  that no standalone FSM rollout script previously existed; this
  figure script is the first to actually run one end-to-end)
- `src/planning/frenet_planner.py` (`plan()`, the real Frenet
  candidate-generation + feasibility-evaluation function)
- Same scenario as fig01: `training_tfexample.tfrecord-00000-of-01000#378__t18__237_456`

Shows the REAL closed-loop executed trajectory (FsmPolicy driving a
real `MergeEnvironment` rollout through `BehaviorExecutor` +
`CommonDownstream`, i.e. the same Frenet planner + LTV-MPC path PPO
also uses) plus the 4 real per-action candidate paths
(KEEP/FOLLOW/MERGE/STOP) computed at the episode's first decision
frame.

**Important caveat:** this project's planner generates exactly ONE
candidate trajectory per `BehaviorAction` -- there is no multi
-candidate search in the codebase. "Candidate paths" in this figure
means these 4 real alternative-action candidates, not a synthesized
path family. For this particular (gentle highway-merge) scenario the
4 candidates nearly overlap near the decision point, which is an
accurate reflection of the real geometry, not a rendering artifact.
Figure aspect ratio is wider than the recommended 4:3/1:1 because the
real trajectory itself is long and shallow; forcing a squarer aspect
would have required either distorting the geometry or mostly-empty
padding, both judged worse than the current proportions.

## fig03_ppo_training

**Source:**
`outputs/diagnostics/retrain_freeze3213b1b/wandb_per_update_raw.csv`
-- 36 rows (PPO updates 1-36), parsed from the real Full PPO
Retraining run's own W&B offline log
(`wandb/run-20260930_203415-zid8h94i/files/output.log`, run ID
`zid8h94i`) by `scripts/diagnostics/analyze_wandb_training_history.py`
in a prior session. This figure script only re-plots that existing CSV
at document-ready size/style; it does not regenerate or modify the
underlying metrics.

Left panel: `train/episode_return` per update. Right panel:
`train/success_rate` and `train/collision_rate` per update.

## fig04_evaluation_framework

**Source:**
- `outputs/checkpoints/full_seed0_step000036.pkl` (the real, final PPO
  checkpoint from the same retraining run)
- `outputs/visualizations/train_diag64_step000036/selected_episodes.json`
  (the real outcome-scan output of `scripts/visualize_ppo.py`,
  deterministic policy, over the 64 Fixed-TRAIN64 maneuvers)

Re-runs the first listed Success/Collision/Timeout maneuver_id through
the same checkpoint/policy/downstream to re-render each at document
-ready size. Maneuver IDs used:
- Success: `training_tfexample.tfrecord-00000-of-01000#157__t28__290_340`
- Collision: `training_tfexample.tfrecord-00000-of-01000#188__t75__498_500`
- Timeout: `training_tfexample.tfrecord-00000-of-01000#117__t75__132_394`

**Offroad is omitted, not fabricated:** `selected_episodes.json`'s own
`"offroad"` list is empty -- offroad never occurred across the 64
evaluated maneuvers at this checkpoint. No FSM-vs-PPO comparison bar
chart was made (Option B in the task brief) because no FSM baseline
evaluation results exist anywhere in the repository (confirmed via
audit: `docs/HANDOFF.md`/`README.md` explicitly state the FSM baseline
comparison is NOT implemented) -- inventing FSM numbers was not an
option.

## Design notes

- White background, no gradients/shadows/3D, Figure-caption-free (to
  be captioned separately in the report document).
- Color convention reused from `src/visualization/style.py` (the
  project's own existing paper/PPT figure style module): ego near
  -black, source lane blue-gray, target lane orange, target-front red.
- Line style (solid vs dashed/dash-dot) differentiates series in
  addition to color for colorblind-safe reading.
- 300 DPI PNG + vector PDF for every figure.
