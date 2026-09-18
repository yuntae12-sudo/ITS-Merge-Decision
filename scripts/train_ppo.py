#!/usr/bin/env python3
"""PPO full-training entry point (docs/ppo/PPO_PLAN.md).

P1 scope: skeleton only. Loads and validates the PPO + reward configs
and reports the resolved settings; does not yet build a rollout,
network, or run any environment steps (that lands across P2-P4). This
script is intentionally NOT wired to run a real training loop until
P5's Smoke Training explicitly verifies the full pipeline end-to-end --
running this script today only proves config loading + seeding work.

IMPORTANT (PPO_PLAN.md SS0/SS5.9): this entry point must never be
pointed at anything beyond the small, deterministic canonical-TRAIN
subset Smoke Training uses through P5. Full/long training runs are
explicitly out of scope for this effort and are reserved for the user
to run manually starting at P6 -- see
docs/ppo/SMOKE_TRAINING_REPORT.md (written at P5 completion) for the
exact reproduction commands this script is meant to support then.
"""

import argparse

from src.training.config import load_ppo_config, load_reward_config
from src.training.seeding import make_seed_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ppo-config",
        default="configs/ppo/ppo_base.yaml",
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
        "src/training/checkpoint.py for the save/load contract). P1 "
        "skeleton: accepted and echoed only -- real "
        "save/load/resume/additional-update wiring lands in P5 per "
        "PPO_PLAN.md SS0.1/P5 and SS10.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ppo_config = load_ppo_config(args.ppo_config)
    reward_config = load_reward_config(ppo_config.reward_config_path)
    seed = args.seed if args.seed is not None else ppo_config.seed
    seed_state = make_seed_state(seed)

    print(f"Loaded PPO config from {ppo_config.source_path}")
    print(f"  network: {ppo_config.network}")
    print(f"  hyperparameters: {ppo_config.hyperparameters}")
    print(f"  rollout.downstream_mode: {ppo_config.rollout.downstream_mode}")
    print(f"Loaded reward config from {reward_config.source_path} "
          f"(reward_version={reward_config.reward_version})")
    print(f"Seed: {seed} (jax_key={seed_state.jax_key})")
    if args.resume is not None:
        print(
            f"--resume={args.resume} was given, but checkpoint "
            "save/load/resume is a P5 feature (docs/ppo/PPO_PLAN.md SS10) "
            "and is not wired up yet -- this run will not actually resume."
        )
    print(
        "P1 skeleton: config + seed resolved successfully. "
        "Rollout/GAE/PPO-update wiring lands in P2-P4; a real training "
        "loop is intentionally not invoked by this script yet."
    )


if __name__ == "__main__":
    main()
