"""PPO P1: seed handling.

A single deterministic entry point for turning one integer seed into
everything PPO training needs to be reproducible: a JAX PRNG key (for
network init / action sampling) and a NumPy RandomState (for any
plain-Python/NumPy-side sampling, e.g. maneuver-subset selection in
Smoke Training). Kept separate from ``config.py`` since seeding is a
runtime concern, not a config-parsing one.

This does not seed Python's global ``random`` module or any Waymax
internal state -- ``MergeEnvironment`` has no stochastic component
(confirmed in P0; see ``MergeEnvironment.reset``'s own docstring), so
there is nothing environment-side to seed.
"""

import dataclasses

import jax
import numpy as np


@dataclasses.dataclass(frozen=True)
class SeedState:
    seed: int
    jax_key: jax.Array
    numpy_rng: np.random.RandomState


def make_seed_state(seed: int) -> SeedState:
    """Builds one ``SeedState`` from an integer seed. Deterministic:
    the same ``seed`` always yields the same ``jax_key`` bits and the
    same ``numpy_rng`` stream."""

    return SeedState(
        seed=seed,
        jax_key=jax.random.PRNGKey(seed),
        numpy_rng=np.random.RandomState(seed),
    )


def split_key(key: jax.Array, num: int = 2):
    """Thin wrapper around ``jax.random.split`` so call sites import
    seeding utilities from one place rather than reaching into
    ``jax.random`` directly and risking an inconsistent splitting
    convention across PPO modules."""

    return jax.random.split(key, num)
