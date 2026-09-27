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
import time

import jax

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
from src.training.run_manifest import write_run_manifest
from src.training.seeding import make_seed_state
from src.training.trainer import run_training
from src.tracking.wandb_logger import WandbLogger

DEFAULT_DATASET_CONFIG_PATH = "configs/dataset.yaml"


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


def main() -> None:
    args = parse_args()

    ppo_config = load_ppo_config(args.ppo_config)
    reward_config = load_reward_config(ppo_config.reward_config_path)
    seed = args.seed if args.seed is not None else ppo_config.seed
    seed_state = make_seed_state(seed)

    all_train_specs = load_decision_dataset_maneuver_specs("train")
    maneuver_ids = _resolve_maneuver_ids(
        args.max_maneuvers, args.maneuver_ids,
        train_ids=sorted(s.maneuver_id for s in all_train_specs),
    )

    print(f"Loaded PPO config from {ppo_config.source_path}")
    print(f"  network: {ppo_config.network}")
    print(f"  hyperparameters: {ppo_config.hyperparameters}")
    print(f"  rollout.downstream_mode: {ppo_config.rollout.downstream_mode}")
    print(f"Loaded reward config from {reward_config.source_path} "
          f"(reward_version={reward_config.reward_version})")
    print(f"Seed: {seed}")
    print(f"Maneuver subset ({len(maneuver_ids)}): {maneuver_ids}")

    specs_by_id = {s.maneuver_id: s for s in all_train_specs}
    missing = [m for m in maneuver_ids if m not in specs_by_id]
    if missing:
        raise ValueError(f"Selected maneuver_ids not found in TRAIN specs: {missing}")
    maneuvers = [specs_by_id[m] for m in maneuver_ids]

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
        "batch_size": len(maneuvers) * args.max_episode_steps,
        "ppo_epochs": ppo_config.hyperparameters.ppo_epochs,
        "network_layers": ppo_config.network.hidden_sizes,
        "maneuver_ids": maneuver_ids,
        "num_updates": args.num_updates,
        "resumed_from": args.resume,
    })

    checkpoint_path = args.checkpoint_path or (
        f"outputs/checkpoints/train_seed{seed}_{int(time.time())}.pkl"
    )

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
    )
    elapsed = time.time() - t0

    print(f"Training finished in {elapsed:.1f}s")
    print(f"global_env_step={result['global_env_step']}, "
          f"ppo_update_step={result['ppo_update_step']}")
    for i, m in enumerate(result["updates"]):
        print(f"  update {i}: {m}")

    dataset_provenance = load_dataset_provenance()

    payload = CheckpointPayload(
        policy_params=result["training_state"].policy_state.params,
        value_params=result["training_state"].value_state.params,
        optimizer_state={
            "policy": result["training_state"].policy_state.opt_state,
            "value": result["training_state"].value_state.opt_state,
        },
        jax_rng_key=result["rng_key"],
        global_env_step=result["global_env_step"],
        ppo_update_step=result["ppo_update_step"],
        seed=seed,
        config_snapshot={
            "ppo_config_path": ppo_config.source_path,
            "reward_config_path": reward_config.source_path,
            "hyperparameters": dataclasses.asdict(ppo_config.hyperparameters),
            "network_hidden_sizes": list(ppo_config.network.hidden_sizes),
            "maneuver_ids": maneuver_ids,
        },
        reward_version=reward_config.reward_version,
        git_sha=get_git_sha(),
        numpy_rng_state=numpy_rng.get_state(),
        dataset_schema_version=MERGE_DATASET_SCHEMA,
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
