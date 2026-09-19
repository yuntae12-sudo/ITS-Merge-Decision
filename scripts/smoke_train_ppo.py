#!/usr/bin/env python3
"""P5 Smoke Training entry point (docs/ppo/PPO_PLAN.md SS0.1/P5).

Runs the real, end-to-end PPO smoke-training loop against the real
``MergeEnvironment`` (``downstream_mode="frenet_mpc"``) on a small,
deterministic subset of the canonical TRAIN split
(``data/manifests/phase2_dataset_split.csv``), selected via
``--max-maneuvers``/``--maneuver-ids``/``--seed`` -- explicitly NOT a
PPO-FIT/PPO-TUNE split (PPO_PLAN.md SS0/SS9): the subset exists only to
exercise every pipeline stage once, never for tuning/evaluation.

This is pipeline-VERIFICATION only -- not a performance run. Explicitly
forbidden here (per PPO_PLAN.md SS0.1/P5): long/full-TRAIN training,
millions of steps, hyperparameter tuning, W&B sweeps, a TRAIN/TUNE
split, canonical VAL evaluation, FSM-vs-PPO comparison.

Saves a checkpoint at the end of the run (and, with ``--checkpoint-every``,
after intermediate updates too) so ``--resume`` on
``scripts/train_ppo.py`` (or this script) can be exercised end-to-end.
"""

import argparse
import dataclasses
import time

import jax

from src.environment.full_split_evaluator import load_maneuver_specs
from src.environment.merge_environment import MergeEnvironment
from src.policies.ppo.state import create_train_state
from src.training.checkpoint import (
    CheckpointPayload,
    get_git_sha,
    load_checkpoint,
    restore_numpy_rng,
    save_checkpoint,
)
from src.training.config import load_ppo_config, load_reward_config
from src.training.seeding import make_seed_state
from src.training.trainer import run_training
from src.tracking.wandb_logger import WandbLogger

DEFAULT_DATASET_CONFIG_PATH = "outputs/phase1/training_10shard_pilot/dataset_training_10shard.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ppo-config",
        default="configs/ppo/ppo_smoke.yaml",
        help="Path to a PPO run config YAML (see configs/ppo/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed recorded in --ppo-config.",
    )
    parser.add_argument(
        "--max-maneuvers",
        type=int,
        default=None,
        help="Cap the number of canonical-TRAIN maneuvers used this "
        "run (overrides the config's smoke.max_maneuvers). Pipeline "
        "verification only -- never used for tuning/evaluation.",
    )
    parser.add_argument(
        "--maneuver-ids",
        default=None,
        help="Comma-separated explicit maneuver_id list, e.g. "
        "'MAN_0001,MAN_0002'. Overrides --max-maneuvers when given.",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=None,
        help="Override the config's smoke.num_updates.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Override the config's smoke.max_episode_steps.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume from (src/training/"
        "checkpoint.py's save/load contract). Continues global_env_step/"
        "ppo_update_step from where the checkpoint left off, rather than "
        "restarting from zero.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Where to save the final checkpoint. Defaults to "
        "outputs/ppo_checkpoints/smoke_<timestamp>.pkl.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="If > 0, also save an intermediate checkpoint every N "
        "updates (in addition to the final one), at "
        "<checkpoint-path>.step<N>.pkl.",
    )
    parser.add_argument(
        "--dataset-config-path",
        default=DEFAULT_DATASET_CONFIG_PATH,
        help="Waymax dataset config path (see MergeEnvironment).",
    )
    return parser.parse_args()


def select_smoke_maneuver_ids(
    seed: int, max_maneuvers: int, explicit_ids=None
) -> list:
    """Deterministically selects a small subset of canonical-TRAIN
    maneuver_ids for Smoke Training.

    If ``explicit_ids`` is given, uses exactly those (still validated
    against the TRAIN split below). Otherwise takes the first
    ``max_maneuvers`` TRAIN maneuver_ids in a fixed, seed-independent
    sorted order -- deterministic and reproducible run-over-run, which
    matters more here than seed-dependent sampling since Smoke
    Training's whole point is pipeline verification, not performance
    variance.
    """

    from src.environment.dataset_split import load_split_manifest

    train_rows = [
        row for row in load_split_manifest() if row.split == "train"
    ]
    train_ids = sorted(row.maneuver_id for row in train_rows)

    if explicit_ids is not None:
        requested = [m.strip() for m in explicit_ids.split(",") if m.strip()]
        missing = [m for m in requested if m not in train_ids]
        if missing:
            raise ValueError(
                f"--maneuver-ids requested maneuver(s) not in canonical "
                f"TRAIN: {missing}"
            )
        return requested

    del seed  # selection is deterministic/sorted, not seed-sampled.
    return train_ids[:max_maneuvers]


def main() -> None:
    args = parse_args()

    ppo_config = load_ppo_config(args.ppo_config)
    reward_config = load_reward_config(ppo_config.reward_config_path)
    seed = args.seed if args.seed is not None else ppo_config.seed
    seed_state = make_seed_state(seed)

    max_maneuvers = (
        args.max_maneuvers
        if args.max_maneuvers is not None
        else (ppo_config.smoke.max_maneuvers if ppo_config.smoke else 1)
    )
    num_updates = (
        args.num_updates
        if args.num_updates is not None
        else (ppo_config.smoke.num_updates if ppo_config.smoke else 1)
    )
    max_episode_steps = (
        args.max_episode_steps
        if args.max_episode_steps is not None
        else (ppo_config.smoke.max_episode_steps if ppo_config.smoke else 30)
    )

    maneuver_ids = select_smoke_maneuver_ids(
        seed=seed, max_maneuvers=max_maneuvers, explicit_ids=args.maneuver_ids
    )

    print(f"Loaded PPO config from {ppo_config.source_path}")
    print(f"Loaded reward config from {reward_config.source_path} "
          f"(reward_version={reward_config.reward_version})")
    print(f"Seed: {seed}")
    print(f"Smoke maneuver subset (canonical TRAIN only, {len(maneuver_ids)} "
          f"maneuvers): {maneuver_ids}")
    print(f"num_updates={num_updates}, max_episode_steps={max_episode_steps}")

    all_train_specs = load_maneuver_specs("train")
    specs_by_id = {s.maneuver_id: s for s in all_train_specs}
    missing = [m for m in maneuver_ids if m not in specs_by_id]
    if missing:
        raise ValueError(f"Selected maneuver_ids not found in TRAIN specs: {missing}")
    maneuvers = [specs_by_id[m] for m in maneuver_ids]

    env = MergeEnvironment(
        dataset_config_path=args.dataset_config_path,
        downstream_mode=ppo_config.rollout.downstream_mode,
    )

    global_env_step = 0
    ppo_update_step = 0

    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        payload = load_checkpoint(args.resume)
        # Network init still needs SOME key to build the param pytree
        # shape/structure, but the values are immediately overwritten
        # below by the checkpoint's own params -- the init key itself
        # is discarded, never used for actual training randomness.
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
        # Continue the RNG stream from exactly where the checkpoint left
        # off, per PPO_PLAN.md SS10 -- never re-derive it from --seed.
        run_key = payload.jax_rng_key
        global_env_step = payload.global_env_step
        ppo_update_step = payload.ppo_update_step
        # Fix 5: restore the NumPy RNG (minibatch-shuffle) stream too --
        # not just the JAX key -- so resumed minibatch ordering matches
        # what an uninterrupted run would have produced.
        numpy_rng = restore_numpy_rng(payload.numpy_rng_state, fallback_seed=seed)
        print(
            f"Resumed: global_env_step={global_env_step}, "
            f"ppo_update_step={ppo_update_step}"
        )
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
        run_name=f"smoke_seed{seed}_{int(time.time())}",
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
        "batch_size": len(maneuvers) * max_episode_steps,
        "ppo_epochs": ppo_config.hyperparameters.ppo_epochs,
        "network_layers": ppo_config.network.hidden_sizes,
        "maneuver_ids": maneuver_ids,
        "num_updates": num_updates,
    })

    checkpoint_path = args.checkpoint_path or (
        f"outputs/ppo_checkpoints/smoke_seed{seed}_{int(time.time())}.pkl"
    )

    def _save(training_state_, global_env_step_, ppo_update_step_, rng_key_, path):
        payload = CheckpointPayload(
            policy_params=training_state_.policy_state.params,
            value_params=training_state_.value_state.params,
            optimizer_state={
                "policy": training_state_.policy_state.opt_state,
                "value": training_state_.value_state.opt_state,
            },
            jax_rng_key=rng_key_,
            global_env_step=global_env_step_,
            ppo_update_step=ppo_update_step_,
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
        )
        save_checkpoint(payload, path)
        print(f"Saved checkpoint to {path}")

    def on_update(update_index, training_state_, rng_key_, global_env_step_, ppo_update_step_):
        if args.checkpoint_every > 0 and (update_index + 1) % args.checkpoint_every == 0:
            intermediate_path = f"{checkpoint_path}.step{ppo_update_step_}.pkl"
            _save(training_state_, global_env_step_, ppo_update_step_, rng_key_, intermediate_path)

    t0 = time.time()
    result = run_training(
        ppo_config=ppo_config,
        reward_config=reward_config,
        env=env,
        maneuvers=maneuvers,
        training_state=training_state,
        rng_key=run_key,
        numpy_rng=numpy_rng,
        num_updates=num_updates,
        max_steps_per_episode=max_episode_steps,
        global_env_step=global_env_step,
        ppo_update_step=ppo_update_step,
        wandb_logger=wandb_logger,
        on_update=on_update,
    )
    elapsed = time.time() - t0

    print(f"Smoke training finished in {elapsed:.1f}s")
    print(f"global_env_step={result['global_env_step']}, "
          f"ppo_update_step={result['ppo_update_step']}")
    for i, m in enumerate(result["updates"]):
        print(f"  update {i}: {m}")

    _save(
        result["training_state"],
        result["global_env_step"],
        result["ppo_update_step"],
        result["rng_key"],
        checkpoint_path,
    )

    print(f"W&B run dir: {wandb_logger.run_dir}")
    wandb_logger.finish()
    print("W&B run finished (offline files flushed to disk if wandb_mode=offline).")


if __name__ == "__main__":
    main()
