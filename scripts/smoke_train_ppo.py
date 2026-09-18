#!/usr/bin/env python3
"""P5 Smoke Training entry point (docs/ppo/PPO_PLAN.md SS0.1/P5).

P1 scope: skeleton only, wired to config loading + seed handling +
deterministic canonical-TRAIN maneuver-subset selection. The real
rollout/GAE/PPO-update/checkpoint loop lands in P5 once P2-P4 exist.

--max-maneuvers/--maneuver-ids/--seed select a SMALL, deterministic
subset of the canonical TRAIN split (data/manifests/phase2_dataset_split.csv)
for pipeline verification only. This is explicitly NOT a PPO-FIT/
PPO-TUNE split (PPO_PLAN.md SS0/SS9): the subset is never used for
tuning or performance evaluation, only to exercise every stage of the
pipeline at small scale.
"""

import argparse

from src.environment.dataset_split import load_split_manifest
from src.training.config import load_ppo_config, load_reward_config
from src.training.seeding import make_seed_state


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

    del seed  # selection is deterministic/sorted, not seed-sampled;
    # kept as a parameter for API symmetry with make_seed_state and in
    # case a future stage wants seeded sampling instead.
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
    maneuver_ids = select_smoke_maneuver_ids(
        seed=seed,
        max_maneuvers=max_maneuvers,
        explicit_ids=args.maneuver_ids,
    )

    print(f"Loaded PPO config from {ppo_config.source_path}")
    print(f"Loaded reward config from {reward_config.source_path} "
          f"(reward_version={reward_config.reward_version})")
    print(f"Seed: {seed} (jax_key={seed_state.jax_key})")
    print(f"Smoke maneuver subset (canonical TRAIN only, {len(maneuver_ids)} "
          f"maneuvers): {maneuver_ids}")
    print(
        "P1 skeleton: config + seed + deterministic maneuver-subset "
        "selection resolved successfully. The real rollout/GAE/PPO-update/"
        "checkpoint smoke-training loop is implemented in P5, once P2-P4 "
        "exist."
    )


if __name__ == "__main__":
    main()
