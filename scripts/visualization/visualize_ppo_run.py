#!/usr/bin/env python3
"""PPO Rollout Visualization Pipeline (docs/ppo/PPO_VISUALIZATION_GUIDE.md).

VISUALIZATION / DIAGNOSTIC ONLY. Loads a trained PPO checkpoint,
reconstructs its exact policy/value networks, runs real closed-loop
``MergeEnvironment(downstream_mode="frenet_mpc")`` rollouts over the
checkpoint's own training maneuver scope (or an explicit
override), finds one representative Success/Collision/Timeout episode,
and renders a GIF + trace.csv + trace.png + rollout_4panel.png + a run
manifest for each.

This is a POST-TRAINING FINAL-CHECKPOINT EVALUATION -- not a replay of
any specific training-time rollout, and not evidence of generalization
by itself (see docs/ppo/PPO_VISUALIZATION_GUIDE.md).

Basic usage:

    PYTHONPATH=. python scripts/visualization/visualize_ppo_run.py \\
        --checkpoint outputs/ppo_checkpoints/p6_baseline_50m_50u_100s_seed0.pkl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import jax

from src.environment.dataset_split import load_split_manifest
from src.environment.full_split_evaluator import load_maneuver_specs
from src.environment.merge_environment import MergeEnvironment
from src.scenarios.merge_v2 import DATASET_SCHEMA_V2
from src.training.config import load_reward_config
from src.visualization.episode_summary import write_episode_summary
from src.visualization.manifest import build_manifest, write_manifest
from src.visualization.outcome_scan import (
    DEFAULT_MAX_PER_OUTCOME,
    build_outcome_index,
    select_representative_episodes,
    write_outcome_index_csv,
)
from src.visualization.ppo_checkpoint_policy import restore_ppo_checkpoint
from src.visualization.ppo_rollout import run_ppo_episode
from src.visualization.render import (
    render_rollout_4panel,
    render_rollout_gif,
    render_trace_png,
)
from src.visualization.trace_writer import write_trace_csv

DEFAULT_DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"
DEFAULT_DOWNSTREAM_MODE = "frenet_mpc"
DEFAULT_MAX_EPISODE_STEPS = 100
OUTPUT_ROOT = "outputs/ppo_visualizations"
DEFAULT_V2_ROOT = "data/manifests/v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a PPO checkpoint .pkl (see src/training/checkpoint.py).")
    parser.add_argument("--run-id", default=None, help="Output run id; defaults to the checkpoint filename stem.")
    parser.add_argument(
        "--scope", choices=("checkpoint", "split", "explicit"), default="checkpoint",
        help="Which maneuvers to evaluate: 'checkpoint' (default) = the exact "
        "maneuver_ids the checkpoint's config_snapshot recorded; 'split' = "
        "the full canonical split named by --split; 'explicit' = --maneuver-ids.",
    )
    parser.add_argument("--split", choices=("train", "tune", "validation"), default=None,
                         help="Used only with --scope split. Never used by default (Rule 5, P6_EXPERIMENT_GUIDE.md: "
                         "canonical VALIDATION must be explicitly requested, never auto-used).")
    parser.add_argument("--maneuver-ids", default=None, help="Comma-separated explicit maneuver_id list (used only with --scope explicit).")
    parser.add_argument("--policy-mode", choices=("deterministic", "stochastic"), default="deterministic",
                         help="Default deterministic (argmax) for reproducibility.")
    parser.add_argument("--policy-seed", type=int, default=0, help="RNG seed for --policy-mode stochastic.")
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--dataset-config-path", default=DEFAULT_DATASET_CONFIG_PATH)
    parser.add_argument("--downstream-mode", default=DEFAULT_DOWNSTREAM_MODE,
                         help="Must stay 'frenet_mpc' for real PPO research evaluation (see MergeEnvironment's "
                         "Stage 3-H freeze notice). Overridable only for explicit debugging.")
    parser.add_argument("--reward-config-path", default=None,
                         help="Defaults to the checkpoint's own recorded reward_config_path.")
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--maneuver-table", default=f"{DEFAULT_V2_ROOT}/merge_maneuvers_v2.csv")
    parser.add_argument("--candidate-manifest", default=f"{DEFAULT_V2_ROOT}/merge_manifest_v2.csv")
    parser.add_argument("--split-manifest", default=f"{DEFAULT_V2_ROOT}/dataset_split_v2.csv")
    parser.add_argument(
        "--allow-legacy-dataset", action="store_true",
        help="Permit historical v1 checkpoints for diagnosis only; never combine them with v2 results.",
    )
    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument("--scan-only", action="store_true",
                               help="Only run the outcome scan (outcome_index.csv + selected_episodes.json); skip rendering.")
    render_group.add_argument("--render-all", action="store_true",
                               help="Render EVERY evaluated episode per outcome category, not just the default "
                               f"top {DEFAULT_MAX_PER_OUTCOME}. outcome classification itself is unaffected -- "
                               "this only changes how many episodes get rendered.")
    parser.add_argument("--max-per-outcome", type=int, default=DEFAULT_MAX_PER_OUTCOME,
                         help=f"Max episodes to render per outcome category in the default mode (default {DEFAULT_MAX_PER_OUTCOME}). "
                         "Ignored with --render-all/--scan-only.")
    parser.add_argument("--select-success", default=None, help="Comma-separated maneuver_id(s) to force into the success selection (in addition to the default/--render-all picks).")
    parser.add_argument("--select-collision", default=None, help="Comma-separated maneuver_id(s) to force into the collision selection.")
    parser.add_argument("--select-timeout", default=None, help="Comma-separated maneuver_id(s) to force into the timeout selection.")
    parser.add_argument("--select-offroad", default=None, help="Comma-separated maneuver_id(s) to force into the offroad selection.")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """``argv=None`` (default) reads ``sys.argv`` as usual; a caller
    (e.g. a test) may pass an explicit argument list instead."""

    return build_parser().parse_args(argv)


# Test-friendly alias -- identical to parse_args, kept as a separate
# name only so tests can call it without shadowing the module-level
# parse_args() any differently than production code does.
parse_args_from = parse_args


def resolve_render_policy(render_all: bool, scan_only: bool, scope: str, max_per_outcome: int):
    """Resolves the effective (``max_per_outcome`` for
    ``select_representative_episodes``, ``render_policy`` manifest dict)
    pair from the CLI flags. Extracted as its own function so both
    ``main()`` and tests exercise the exact same decision logic (task
    spec Section 6: an explicit --maneuver-ids scope must never lose a
    user-named maneuver to the default per-outcome cap)."""

    if render_all:
        return None, {"mode": "all"}
    if scan_only:
        return max_per_outcome, {"mode": "scan_only"}
    if scope == "explicit":
        return None, {"mode": "all", "reason": "scope=explicit always renders every requested maneuver"}
    return max_per_outcome, {"mode": "representative_per_outcome", "max_per_outcome": max_per_outcome}


def _resolve_maneuvers(args, restored):
    if args.scope == "checkpoint":
        maneuver_ids = sorted(restored.maneuver_ids)
        if not maneuver_ids:
            raise ValueError(
                "--scope checkpoint requested but this checkpoint's config_snapshot "
                "has no maneuver_ids recorded (older checkpoint?). Use --scope split "
                "or --scope explicit instead."
            )
        if args.allow_legacy_dataset:
            specs = load_maneuver_specs("train")
        else:
            specs = load_maneuver_specs(
                "train", split_manifest_path=args.split_manifest,
                maneuver_table_path=args.maneuver_table,
                candidate_manifest_path=args.candidate_manifest,
                required_schema_version=DATASET_SCHEMA_V2,
            )
        specs_by_id = {s.maneuver_id: s for s in specs}
        missing = [m for m in maneuver_ids if m not in specs_by_id]
        if missing:
            raise ValueError(f"Checkpoint maneuver_ids not found in TRAIN specs: {missing}")
        return [specs_by_id[m] for m in maneuver_ids]

    if args.scope == "split":
        split = args.split or "train"
        kwargs = {} if args.allow_legacy_dataset else dict(
            split_manifest_path=args.split_manifest,
            maneuver_table_path=args.maneuver_table,
            candidate_manifest_path=args.candidate_manifest,
            required_schema_version=DATASET_SCHEMA_V2,
        )
        return sorted(load_maneuver_specs(split, **kwargs), key=lambda s: s.maneuver_id)

    if args.scope == "explicit":
        if not args.maneuver_ids:
            raise ValueError("--scope explicit requires --maneuver-ids")
        requested = [m.strip() for m in args.maneuver_ids.split(",") if m.strip()]
        kwargs = {} if args.allow_legacy_dataset else dict(
            split_manifest_path=args.split_manifest,
            maneuver_table_path=args.maneuver_table,
            candidate_manifest_path=args.candidate_manifest,
            required_schema_version=DATASET_SCHEMA_V2,
        )
        all_specs = {s.maneuver_id: s for split in ("train", "tune", "validation") for s in load_maneuver_specs(split, **kwargs)}
        missing = [m for m in requested if m not in all_specs]
        if missing:
            raise ValueError(f"--maneuver-ids requested maneuver(s) not found in any split: {missing}")
        return [all_specs[m] for m in requested]

    raise ValueError(f"Unknown --scope: {args.scope}")


def main() -> None:
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    restored = restore_ppo_checkpoint(
        args.checkpoint,
        expected_dataset_schema_version=(
            None if args.allow_legacy_dataset else DATASET_SCHEMA_V2
        ),
    )
    print(f"  seed={restored.seed} reward_version={restored.reward_version} "
          f"ppo_update_step={restored.ppo_update_step} global_env_step={restored.global_env_step}")
    print(f"  network_hidden_sizes={restored.network_hidden_sizes}")
    print(f"  checkpoint git_sha={restored.checkpoint_git_sha}")

    reward_config_path = args.reward_config_path or restored.reward_config_path or "configs/reward/merge_reward_v0.yaml"
    reward_config = load_reward_config(reward_config_path)
    print(f"Reward config: {reward_config_path} (reward_version={reward_config.reward_version})")

    maneuvers = _resolve_maneuvers(args, restored)
    maneuver_ids = [m.maneuver_id for m in maneuvers]
    print(f"Evaluating {len(maneuvers)} maneuver(s) under scope={args.scope!r}: {maneuver_ids[:10]}"
          + (" ..." if len(maneuver_ids) > 10 else ""))

    env = MergeEnvironment(
        dataset_config_path=args.dataset_config_path,
        downstream_mode=args.downstream_mode,
        required_dataset_schema_version=(
            None if args.allow_legacy_dataset else DATASET_SCHEMA_V2
        ),
    )

    rng_key = jax.random.PRNGKey(args.policy_seed) if args.policy_mode == "stochastic" else None

    run_id = args.run_id or os.path.splitext(os.path.basename(args.checkpoint))[0]
    run_dir = os.path.join(args.output_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Running deterministic outcome scan (policy_mode={args.policy_mode}) ...")
    episode_results = []
    for maneuver in maneuvers:
        if rng_key is not None:
            rng_key, episode_key = jax.random.split(rng_key)
        else:
            episode_key = None
        result = run_ppo_episode(
            env=env,
            maneuver=maneuver,
            policy=restored.policy,
            value_network=restored.value_network,
            value_params=restored.value_params,
            reward_config=reward_config,
            max_steps=args.max_episode_steps,
            policy_mode=args.policy_mode,
            rng_key=episode_key,
        )
        episode_results.append(result)
        print(f"  {maneuver.maneuver_id}: outcome={result.outcome} "
              f"steps={result.physical_step_count} decisions={result.policy_decision_count} "
              f"merge_commit_step={result.merge_commit_step}")

    outcome_rows = build_outcome_index(episode_results)
    outcome_index_path = os.path.join(run_dir, "outcome_index.csv")
    write_outcome_index_csv(outcome_rows, outcome_index_path)
    print(f"Wrote {outcome_index_path}")

    max_per_outcome, render_policy = resolve_render_policy(
        render_all=args.render_all, scan_only=args.scan_only,
        scope=args.scope, max_per_outcome=args.max_per_outcome,
    )

    selected = select_representative_episodes(episode_results, max_per_outcome=max_per_outcome)

    overrides = {
        "success": args.select_success,
        "collision": args.select_collision,
        "timeout": args.select_timeout,
        "offroad": args.select_offroad,
    }
    results_by_id = {r.maneuver_id: r for r in episode_results}
    for outcome, override_ids in overrides.items():
        if override_ids is None:
            continue
        requested = [m.strip() for m in override_ids.split(",") if m.strip()]
        missing = [m for m in requested if m not in results_by_id]
        if missing:
            raise ValueError(f"--select-{outcome} requested maneuver(s) not in evaluated scope: {missing}")
        # Explicit user selection is ADDITIVE to (never a silent
        # replacement of) the default/--render-all pick, de-duplicated,
        # stable-sorted so the merged list stays deterministic.
        merged = sorted(set(selected[outcome]["maneuver_ids"]) | set(requested))
        selected[outcome] = {"maneuver_ids": merged, "reason": selected[outcome]["reason"] + "; plus user-specified --select-* override(s)"}

    import json
    selected_episodes_path = os.path.join(run_dir, "selected_episodes.json")
    selected_ids_only = {k: v["maneuver_ids"] for k, v in selected.items()}
    with open(selected_episodes_path, "w") as f:
        json.dump(selected_ids_only, f, indent=2)
    print(f"Wrote {selected_episodes_path}: {selected_ids_only}")
    for outcome, info in selected.items():
        if not info["maneuver_ids"]:
            print(f"  NOTE: {info['reason']}")

    manifest = build_manifest(
        restored=restored,
        run_id=run_id,
        evaluated_maneuver_ids=maneuver_ids,
        policy_mode=args.policy_mode,
        policy_seed=args.policy_seed if args.policy_mode == "stochastic" else None,
        downstream_mode=args.downstream_mode,
        max_episode_steps=args.max_episode_steps,
        selected_episodes=selected_ids_only,
        scope=args.scope,
        render_policy=render_policy,
    )
    write_manifest(manifest, os.path.join(run_dir, "manifest.json"))
    print(f"Wrote {os.path.join(run_dir, 'manifest.json')}")

    if args.scan_only:
        print("--scan-only requested; skipping rendering.")
        return

    for outcome, info in selected.items():
        for maneuver_id in info["maneuver_ids"]:
            episode = results_by_id[maneuver_id]
            episode_dir = os.path.join(run_dir, outcome, maneuver_id)
            os.makedirs(episode_dir, exist_ok=True)

            print(f"Rendering {outcome}/{maneuver_id} ...")
            write_trace_csv(episode, os.path.join(episode_dir, "trace.csv"))
            write_episode_summary(episode, os.path.join(episode_dir, "summary.json"))

            # Re-run the episode once more to get a live `env` positioned
            # at each frame in sequence for rendering (the GIF/4-panel
            # renderer needs env._polylines_by_id, which is fixed per-
            # episode/scene, so a fresh deterministic re-run is safe and
            # byte-identical to the scan above -- same policy_mode, same
            # maneuver, same seed).
            if args.policy_mode == "stochastic":
                render_rng_key = jax.random.PRNGKey(args.policy_seed)
            else:
                render_rng_key = None
            render_episode = run_ppo_episode(
                env=env,
                maneuver=next(m for m in maneuvers if m.maneuver_id == maneuver_id),
                policy=restored.policy,
                value_network=restored.value_network,
                value_params=restored.value_params,
                reward_config=reward_config,
                max_steps=args.max_episode_steps,
                policy_mode=args.policy_mode,
                rng_key=render_rng_key,
            )

            render_rollout_gif(render_episode, env, os.path.join(episode_dir, "rollout.gif"))
            render_trace_png(render_episode, os.path.join(episode_dir, "trace.png"))
            render_rollout_4panel(render_episode, env, os.path.join(episode_dir, "rollout_4panel.png"))
            print(f"  wrote {episode_dir}/{{rollout.gif, trace.png, rollout_4panel.png}}")

    print("Done.")
    print(f"\nOutput root: {run_dir}")


if __name__ == "__main__":
    main()
