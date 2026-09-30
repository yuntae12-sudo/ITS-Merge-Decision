#!/usr/bin/env python3
"""PPO training entry point.

Loads the PPO + reward configs, builds/resumes a training state, runs
``src.training.trainer.run_training`` against the real
``MergeEnvironment`` on the final canonical MERGE Dataset TRAIN split,
and saves a checkpoint.

``--resume`` performs a REAL resume: it loads the checkpoint's policy/
value params + optimizer state + JAX RNG key + global_env_step/
ppo_update_step and continues training from there -- not a restart
from zero.
"""

import argparse
import dataclasses
import hashlib
import time
from pathlib import Path

import jax
import numpy as np

from src.environment.full_split_evaluator import load_decision_dataset_maneuver_specs
from src.environment.merge_environment import MergeEnvironment
from src.scenarios.merge_v2 import MERGE_DATASET_SCHEMA
from src.policies.ppo.state import create_train_state
from src.training.checkpoint import (
    CheckpointPayload,
    get_git_sha,
    load_checkpoint,
    restore_numpy_rng,
    save_checkpoint,
)
from src.training.config import load_ppo_config, load_reward_config
from src.training.provenance import load_dataset_provenance
from src.training.progress_reporter import ProgressReporter
from src.training.run_manifest import write_run_manifest
from src.training.seeding import make_seed_state
from src.training.trainer import run_training
from src.tracking.wandb_logger import WandbLogger

DEFAULT_DATASET_CONFIG_PATH = "configs/dataset.yaml"
EXPECTED_TRAIN_POOL_SIZE = 1097
FULL_TRAIN_SAMPLING_STRATEGY = "deterministic_seeded_shuffle_cycle_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ppo-config",
        default="configs/ppo/train.yaml",
        help="Path to a PPO run config YAML (see configs/ppo/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed recorded in --ppo-config.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume from (see "
        "src/training/checkpoint.py for the save/load contract). "
        "Continues global_env_step/ppo_update_step from the checkpoint "
        "rather than restarting from zero.",
    )
    parser.add_argument(
        "--max-maneuvers",
        type=int,
        default=1,
        help="Number of canonical-TRAIN maneuvers to use, taken in "
        "fixed sorted maneuver_id order.",
    )
    parser.add_argument(
        "--maneuver-ids",
        default=None,
        help="Comma-separated explicit maneuver_id list. Overrides "
        "--max-maneuvers when given.",
    )
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Use the complete canonical TRAIN pool with a deterministic "
        "per-update shuffle/cycle schedule.",
    )
    parser.add_argument(
        "--maneuvers-per-update",
        type=int,
        default=None,
        help="Required with --full-train: number of TRAIN maneuvers rolled "
        "out per PPO update.",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=1,
        help="Number of PPO updates to run.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=30,
        help="Max physical steps per rollout episode.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Where to save the final checkpoint. Defaults to "
        "outputs/checkpoints/train_<timestamp>.pkl.",
    )
    parser.add_argument(
        "--checkpoint-every-updates",
        type=int,
        default=0,
        help="Save a periodic checkpoint every N completed PPO updates; "
        "0 disables periodic saves.",
    )
    parser.add_argument(
        "--dataset-config-path",
        default=DEFAULT_DATASET_CONFIG_PATH,
        help="Waymax dataset config path (see MergeEnvironment).",
    )
    return parser.parse_args()


def _resolve_maneuver_ids(max_maneuvers: int, explicit_ids, train_ids) -> list:
    if explicit_ids is not None:
        requested = [m.strip() for m in explicit_ids.split(",") if m.strip()]
        missing = [m for m in requested if m not in train_ids]
        if missing:
            raise ValueError(
                f"--maneuver-ids requested maneuver(s) not in canonical "
                f"TRAIN: {missing}"
            )
        return requested
    return train_ids[:max_maneuvers]


def deterministic_full_train_batch_ids(
    train_ids: list,
    seed: int,
    absolute_ppo_update_step: int,
    maneuvers_per_update: int,
) -> list:
    """Returns one stateless deterministic shuffle/cycle TRAIN batch."""

    pool = sorted(train_ids)
    if not pool:
        raise ValueError("full-train pool must not be empty")
    if len(pool) != len(set(pool)):
        raise ValueError("full-train pool contains duplicate maneuver_ids")
    if absolute_ppo_update_step < 0:
        raise ValueError("absolute_ppo_update_step must be >= 0")
    if maneuvers_per_update <= 0:
        raise ValueError("maneuvers_per_update must be > 0")

    pool_size = len(pool)
    position = absolute_ppo_update_step * maneuvers_per_update
    remaining = maneuvers_per_update
    selected = []
    while remaining:
        sweep_index, offset = divmod(position, pool_size)
        digest = hashlib.sha256(f"{seed}:{sweep_index}".encode("utf-8")).digest()
        sweep_seed = int.from_bytes(digest[:4], "big")
        shuffled = list(pool)
        np.random.RandomState(sweep_seed).shuffle(shuffled)
        take = min(remaining, pool_size - offset)
        selected.extend(shuffled[offset:offset + take])
        position += take
        remaining -= take
    return selected


def _checkpoint_config_snapshot(
    ppo_config,
    reward_config,
    maneuver_ids: list,
    full_train: bool,
    maneuvers_per_update,
) -> dict:
    return {
        "ppo_config_path": ppo_config.source_path,
        "reward_config_path": reward_config.source_path,
        "hyperparameters": dataclasses.asdict(ppo_config.hyperparameters),
        "network_hidden_sizes": list(ppo_config.network.hidden_sizes),
        "maneuver_ids": list(maneuver_ids),
        "full_train": full_train,
        "train_pool_size": len(maneuver_ids),
        "maneuvers_per_update": maneuvers_per_update,
        "sampling_strategy": (
            FULL_TRAIN_SAMPLING_STRATEGY if full_train else "fixed_subset"
        ),
    }


def _build_checkpoint_payload(
    training_state,
    rng_key,
    numpy_rng,
    global_env_step: int,
    ppo_update_step: int,
    seed: int,
    config_snapshot: dict,
    reward_version: str,
) -> CheckpointPayload:
    return CheckpointPayload(
        policy_params=training_state.policy_state.params,
        value_params=training_state.value_state.params,
        optimizer_state={
            "policy": training_state.policy_state.opt_state,
            "value": training_state.value_state.opt_state,
        },
        jax_rng_key=rng_key,
        global_env_step=global_env_step,
        ppo_update_step=ppo_update_step,
        seed=seed,
        config_snapshot=config_snapshot,
        reward_version=reward_version,
        git_sha=get_git_sha(),
        numpy_rng_state=numpy_rng.get_state(),
        dataset_schema_version=MERGE_DATASET_SCHEMA,
    )


def make_periodic_checkpoint_callback(
    interval: int,
    checkpoint_dir,
    filename_prefix: str,
    seed: int,
    reward_version: str,
    config_snapshot: dict,
    numpy_rng,
):
    """Returns a function ``(training_state, rng_key, global_env_step,
    ppo_update_step) -> Optional[str]`` that saves a periodic checkpoint
    when due (``ppo_update_step % interval == 0``) and returns the
    saved path, or ``None`` on an off-interval update / when periodic
    checkpointing is disabled. Unchanged save format/naming/cadence --
    only the print statement moved to the caller so it can be merged
    into the single progress line (display-only)."""

    if interval is None or interval == 0:
        return None
    if interval < 0:
        raise ValueError("periodic checkpoint interval must be >= 0")
    checkpoint_dir = Path(checkpoint_dir)

    def maybe_save(training_state, rng_key, global_env_step, ppo_update_step):
        if ppo_update_step % interval != 0:
            return None
        path = checkpoint_dir / f"{filename_prefix}_step{ppo_update_step:06d}.pkl"
        payload = _build_checkpoint_payload(
            training_state=training_state,
            rng_key=rng_key,
            numpy_rng=numpy_rng,
            global_env_step=global_env_step,
            ppo_update_step=ppo_update_step,
            seed=seed,
            config_snapshot=config_snapshot,
            reward_version=reward_version,
        )
        save_checkpoint(payload, str(path))
        return str(path)

    return maybe_save


def make_combined_update_callback(periodic_checkpoint_fn, progress_reporter):
    """Wires the (optional) periodic-checkpoint save and the (always
    -on) display-only progress reporter into the single ``on_update``
    callback ``trainer.run_training`` accepts. Pure display/IO
    composition -- never touches training_state/rng_key beyond passing
    them through to ``periodic_checkpoint_fn`` unchanged, never raises
    (a progress-line formatting bug must not abort training)."""

    def on_update(update_index, training_state, rng_key, global_env_step, ppo_update_step):
        checkpoint_path = None
        if periodic_checkpoint_fn is not None:
            checkpoint_path = periodic_checkpoint_fn(
                training_state, rng_key, global_env_step, ppo_update_step
            )
        try:
            progress_reporter.report(
                update_index=update_index,
                global_env_step=global_env_step,
                ppo_update_step=ppo_update_step,
                checkpoint_path=checkpoint_path,
            )
        except Exception as exc:  # noqa: BLE001 -- display-only, must never abort training.
            print(f"WARNING: progress display failed ({exc}); training continues unaffected.", flush=True)

    return on_update


def main() -> None:
    args = parse_args()

    if args.full_train and args.maneuver_ids is not None:
        raise ValueError("--full-train cannot be combined with --maneuver-ids")
    if args.full_train and args.maneuvers_per_update is None:
        raise ValueError("--maneuvers-per-update is required with --full-train")
    if args.maneuvers_per_update is not None and args.maneuvers_per_update <= 0:
        raise ValueError("--maneuvers-per-update must be > 0")
    if not args.full_train and args.maneuvers_per_update is not None:
        raise ValueError("--maneuvers-per-update requires --full-train")
    if args.checkpoint_every_updates < 0:
        raise ValueError("--checkpoint-every-updates must be >= 0")

    ppo_config = load_ppo_config(args.ppo_config)
    reward_config = load_reward_config(ppo_config.reward_config_path)
    seed = args.seed if args.seed is not None else ppo_config.seed
    seed_state = make_seed_state(seed)

    all_train_specs = load_decision_dataset_maneuver_specs("train")
    train_ids = sorted(s.maneuver_id for s in all_train_specs)
    specs_by_id = {s.maneuver_id: s for s in all_train_specs}
    if len(specs_by_id) != len(all_train_specs):
        raise ValueError("Canonical TRAIN contains duplicate maneuver_ids")

    if args.full_train:
        if len(train_ids) != EXPECTED_TRAIN_POOL_SIZE:
            raise ValueError(
                f"--full-train requires exactly {EXPECTED_TRAIN_POOL_SIZE} canonical "
                f"TRAIN maneuvers, found {len(train_ids)}"
            )
        maneuver_ids = train_ids
        maneuvers = [specs_by_id[maneuver_id] for maneuver_id in maneuver_ids]
        maneuvers_per_update = args.maneuvers_per_update

        def maneuver_selector(absolute_ppo_update_step):
            batch_ids = deterministic_full_train_batch_ids(
                train_ids=maneuver_ids,
                seed=seed,
                absolute_ppo_update_step=absolute_ppo_update_step,
                maneuvers_per_update=maneuvers_per_update,
            )
            return [specs_by_id[maneuver_id] for maneuver_id in batch_ids]
    else:
        maneuver_ids = _resolve_maneuver_ids(
            args.max_maneuvers, args.maneuver_ids, train_ids=train_ids
        )
        maneuvers = [specs_by_id[maneuver_id] for maneuver_id in maneuver_ids]
        maneuvers_per_update = len(maneuvers)
        maneuver_selector = None

    checkpoint_config_snapshot = _checkpoint_config_snapshot(
        ppo_config=ppo_config,
        reward_config=reward_config,
        maneuver_ids=maneuver_ids,
        full_train=args.full_train,
        maneuvers_per_update=maneuvers_per_update,
    )

    print(f"Loaded PPO config from {ppo_config.source_path}")
    print(f"  network: {ppo_config.network}")
    print(f"  hyperparameters: {ppo_config.hyperparameters}")
    print(f"  rollout.downstream_mode: {ppo_config.rollout.downstream_mode}")
    print(f"Loaded reward config from {reward_config.source_path} "
          f"(reward_version={reward_config.reward_version})")
    print(f"Seed: {seed}")
    if args.full_train:
        print(
            f"Full TRAIN pool: {len(maneuver_ids)} maneuvers; "
            f"maneuvers_per_update={maneuvers_per_update}; "
            f"sampling={FULL_TRAIN_SAMPLING_STRATEGY}"
        )
    else:
        print(f"Maneuver subset ({len(maneuver_ids)}): {maneuver_ids}")

    env = MergeEnvironment(
        dataset_config_path=args.dataset_config_path,
        downstream_mode=ppo_config.rollout.downstream_mode,
        required_dataset_schema_version=MERGE_DATASET_SCHEMA,
    )

    global_env_step = 0
    ppo_update_step = 0

    if args.resume is not None:
        print(f"--resume={args.resume}: loading checkpoint and continuing training "
              "(not restarting from scratch).")
        payload = load_checkpoint(args.resume)
        if payload.dataset_schema_version != MERGE_DATASET_SCHEMA:
            raise ValueError(
                f"Cannot resume {payload.dataset_schema_version!r} checkpoint "
                f"with {MERGE_DATASET_SCHEMA!r} dataset"
            )
        if args.full_train:
            expected_sampling = {
                "full_train": True,
                "train_pool_size": EXPECTED_TRAIN_POOL_SIZE,
                "maneuvers_per_update": maneuvers_per_update,
                "sampling_strategy": FULL_TRAIN_SAMPLING_STRATEGY,
            }
            actual_sampling = {
                key: payload.config_snapshot.get(key) for key in expected_sampling
            }
            if payload.seed != seed or actual_sampling != expected_sampling:
                raise ValueError(
                    "Full-train resume sampling contract mismatch: "
                    f"checkpoint_seed={payload.seed}, requested_seed={seed}, "
                    f"checkpoint_sampling={actual_sampling}, "
                    f"requested_sampling={expected_sampling}"
                )
        training_state = create_train_state(
            seed_state.jax_key,
            learning_rate=ppo_config.hyperparameters.learning_rate,
            max_grad_norm=ppo_config.hyperparameters.max_grad_norm,
            policy_hidden_sizes=ppo_config.network.hidden_sizes,
            value_hidden_sizes=ppo_config.network.hidden_sizes,
        )
        training_state = dataclasses.replace(
            training_state,
            policy_state=training_state.policy_state.replace(
                params=payload.policy_params,
                opt_state=payload.optimizer_state["policy"],
            ),
            value_state=training_state.value_state.replace(
                params=payload.value_params,
                opt_state=payload.optimizer_state["value"],
            ),
        )
        run_key = payload.jax_rng_key
        global_env_step = payload.global_env_step
        ppo_update_step = payload.ppo_update_step
        # Restore the NumPy RNG (minibatch-shuffle) stream too -- not
        # just the JAX key -- so resumed minibatch ordering matches
        # what an uninterrupted run would have produced.
        numpy_rng = restore_numpy_rng(payload.numpy_rng_state, fallback_seed=seed)
        print(f"Resumed: global_env_step={global_env_step}, "
              f"ppo_update_step={ppo_update_step}")
    else:
        init_key, run_key = jax.random.split(seed_state.jax_key)
        training_state = create_train_state(
            init_key,
            learning_rate=ppo_config.hyperparameters.learning_rate,
            max_grad_norm=ppo_config.hyperparameters.max_grad_norm,
            policy_hidden_sizes=ppo_config.network.hidden_sizes,
            value_hidden_sizes=ppo_config.network.hidden_sizes,
        )
        numpy_rng = seed_state.numpy_rng

    wandb_logger = WandbLogger(
        project=ppo_config.tracking.wandb_project,
        mode=ppo_config.tracking.wandb_mode,
        run_name=f"train_seed{seed}_{int(time.time())}",
    )
    wandb_logger.log_config({
        "git_sha": get_git_sha(),
        "reward_version": reward_config.reward_version,
        "seed": seed,
        "learning_rate": ppo_config.hyperparameters.learning_rate,
        "gamma": ppo_config.hyperparameters.gamma,
        "gae_lambda": ppo_config.hyperparameters.gae_lambda,
        "clip_epsilon": ppo_config.hyperparameters.clip_epsilon,
        "entropy_coef": ppo_config.hyperparameters.entropy_coef,
        "value_coef": ppo_config.hyperparameters.value_coef,
        "batch_size": maneuvers_per_update * args.max_episode_steps,
        "ppo_epochs": ppo_config.hyperparameters.ppo_epochs,
        "network_layers": ppo_config.network.hidden_sizes,
        "maneuver_ids": maneuver_ids,
        "num_updates": args.num_updates,
        "resumed_from": args.resume,
        "train_pool_size": len(maneuver_ids),
        "maneuvers_per_update": maneuvers_per_update,
        "full_train": args.full_train,
        "sampling_strategy": checkpoint_config_snapshot["sampling_strategy"],
    })

    checkpoint_path = args.checkpoint_path or (
        f"outputs/checkpoints/train_seed{seed}_{int(time.time())}.pkl"
    )
    periodic_checkpoint_fn = make_periodic_checkpoint_callback(
        interval=args.checkpoint_every_updates,
        checkpoint_dir="outputs/checkpoints",
        filename_prefix=(f"full_seed{seed}" if args.full_train else f"train_seed{seed}"),
        seed=seed,
        reward_version=reward_config.reward_version,
        config_snapshot=checkpoint_config_snapshot,
        numpy_rng=numpy_rng,
    )
    progress_reporter = ProgressReporter(
        total_updates=args.num_updates, resumed_from_step=ppo_update_step,
    )
    combined_callback = make_combined_update_callback(periodic_checkpoint_fn, progress_reporter)

    t0 = time.time()
    result = run_training(
        ppo_config=ppo_config,
        reward_config=reward_config,
        env=env,
        maneuvers=maneuvers,
        training_state=training_state,
        rng_key=run_key,
        numpy_rng=numpy_rng,
        num_updates=args.num_updates,
        max_steps_per_episode=args.max_episode_steps,
        global_env_step=global_env_step,
        ppo_update_step=ppo_update_step,
        wandb_logger=wandb_logger,
        on_update=combined_callback,
        maneuver_selector=maneuver_selector,
    )
    elapsed = time.time() - t0

    print(f"Training finished in {elapsed:.1f}s")
    print(f"global_env_step={result['global_env_step']}, "
          f"ppo_update_step={result['ppo_update_step']}")
    for i, m in enumerate(result["updates"]):
        print(f"  update {i}: {m}")

    dataset_provenance = load_dataset_provenance()

    payload = _build_checkpoint_payload(
        training_state=result["training_state"],
        rng_key=result["rng_key"],
        numpy_rng=numpy_rng,
        global_env_step=result["global_env_step"],
        ppo_update_step=result["ppo_update_step"],
        seed=seed,
        config_snapshot=checkpoint_config_snapshot,
        reward_version=reward_config.reward_version,
    )
    save_checkpoint(payload, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    try:
        run_manifest_path = write_run_manifest(
            checkpoint_path=checkpoint_path,
            ppo_config_path=ppo_config.source_path,
            reward_config_path=reward_config.source_path,
            dataset_config_path=args.dataset_config_path,
            maneuver_ids=maneuver_ids,
            seed=seed,
            num_updates=args.num_updates,
            max_episode_steps=args.max_episode_steps,
            reward_version=reward_config.reward_version,
            git_sha=get_git_sha(),
            global_env_step=result["global_env_step"],
            ppo_update_step=result["ppo_update_step"],
            resumed_from=args.resume,
            dataset_schema_version=MERGE_DATASET_SCHEMA,
            dataset_version=dataset_provenance.version,
            dataset_freeze_sha=dataset_provenance.git_sha,
        )
        print(f"Saved run manifest to {run_manifest_path}")
    except Exception as exc:  # noqa: BLE001 -- additive metadata only;
        # must never fail a completed training run.
        print(f"WARNING: failed to write run manifest ({exc}); "
              "checkpoint itself is unaffected.")

    wandb_logger.finish()

    print("\nTraining finished.")
    print("\nTo visualize this checkpoint:")
    print(
        f"\nPYTHONPATH=. python scripts/visualize_ppo.py \\\n"
        f"  --checkpoint {checkpoint_path}"
    )


if __name__ == "__main__":
    main()
