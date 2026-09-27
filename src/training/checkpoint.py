"""PPO checkpoint / resume contract (docs/ppo/PPO_PLAN.md SS10).

P1 scope: structure only -- the ``CheckpointPayload`` shape and the
save/load function signatures are fixed here so later phases (P3's
train state, P4's rollout/GAE, P5's smoke training loop) all target
the same contract from the start. The actual save/load bodies are
implemented in P5, where the full save -> load -> resume -> additional
update cycle is verified end-to-end against real PPO train state.

A checkpoint is NOT considered resume-capable if it only saves model
weights (PPO_PLAN.md SS10). Every checkpoint must carry, at minimum:

- policy parameters
- value parameters
- optimizer state
- JAX PRNG key / RNG state
- global environment step count
- PPO update step count
- seed
- the complete PPO config (or a config snapshot)
- reward version
- Git SHA

``CheckpointPayload`` below is the single object that carries exactly
this set (plus whatever additional metadata a later phase finds useful
-- PPO_PLAN.md SS10 explicitly allows that).
"""

import dataclasses
import pickle
import subprocess
from pathlib import Path
from typing import Any, Optional

from src.scenarios.merge_v2 import LEGACY_DATASET_SCHEMA


def get_git_sha(repo_root: str = ".") -> str:
    """Best-effort current Git SHA of the repo containing this code.
    Used to stamp every checkpoint per PPO_PLAN.md SS10, so a resumed
    run can always be traced back to the exact code that produced it.
    Returns "unknown" rather than raising if git is unavailable (e.g. a
    stripped deployment environment) -- a checkpoint should never fail
    to save purely because ``git`` isn't on PATH."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


@dataclasses.dataclass(frozen=True)
class CheckpointPayload:
    """The full, resume-capable checkpoint contract (PPO_PLAN.md SS10).

    ``policy_params``/``value_params``/``optimizer_state`` are left
    typed as ``Any`` here deliberately: P1 has no PPO train-state
    definition yet (that lands in P3's ``src/policies/ppo/state.py`).
    This dataclass is the stable envelope; P3+ decide what concrete
    pytree structures go inside each field.

    ``numpy_rng_state`` (pre-P6 hardening Fix 5, additive): the tuple
    returned by ``numpy.random.RandomState.get_state()`` for the
    ``numpy_rng`` used to shuffle PPO minibatches
    (``src.training.trainer._minibatch_indices``). P1/P5 already saved
    the JAX PRNG key (governing rollout action sampling) but never this
    NumPy RNG's state, so a resumed run's minibatch SHUFFLE ORDER for
    the very next update would silently diverge from what an
    uninterrupted run would have produced, even though every other
    piece of state (params, optimizer, JAX key, step counters) resumed
    correctly -- this field closes that gap. Defaults to ``None`` so an
    OLDER checkpoint (saved before this field existed) still loads
    without error; ``None`` means "the NumPy RNG state was never
    recorded for this checkpoint" and a caller resuming from one must
    fall back to re-seeding from ``seed`` for the NumPy stream (a real,
    if not bit-identical-mid-stream, degradation -- not a crash).
    """

    policy_params: Any
    value_params: Any
    optimizer_state: Any
    jax_rng_key: Any
    global_env_step: int
    ppo_update_step: int
    seed: int
    config_snapshot: dict
    reward_version: str
    git_sha: str
    extra: Optional[dict] = None
    numpy_rng_state: Optional[tuple] = None
    dataset_schema_version: str = LEGACY_DATASET_SCHEMA


def save_checkpoint(payload: CheckpointPayload, path: str) -> None:
    """Saves ``payload`` to ``path``.

    P1 skeleton: a plain ``pickle`` of the dataclass is sufficient to
    fix the on-disk contract and let P1's import/structure tests pass
    without depending on a PPO train-state pytree that doesn't exist
    yet. P5 verifies the real save -> load -> resume -> additional
    update cycle against actual JAX pytrees (policy/value params,
    optax optimizer state) produced by P3/P4's training code; if a
    JAX-native format (e.g. orbax, already installed in this repo's
    env per the P0 audit) proves preferable for real pytrees at that
    point, P5 may swap the implementation here without changing this
    function's signature or ``CheckpointPayload``'s contract.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataclasses.asdict(payload), f)


def load_checkpoint(path: str) -> CheckpointPayload:
    """Loads a ``CheckpointPayload`` previously written by
    ``save_checkpoint``. Inverse of ``save_checkpoint`` -- see that
    function's docstring for the P1-vs-P5 implementation note."""

    with open(path, "rb") as f:
        raw = pickle.load(f)
    raw.setdefault("numpy_rng_state", None)  # Fix 5: older checkpoints
    raw.setdefault("dataset_schema_version", LEGACY_DATASET_SCHEMA)
    return CheckpointPayload(**raw)


def require_checkpoint_dataset_schema(
    payload: CheckpointPayload, expected_schema_version: str
) -> None:
    """Fail closed instead of mixing legacy and v2 research results."""

    if payload.dataset_schema_version != expected_schema_version:
        raise ValueError(
            "Checkpoint/dataset schema mismatch: checkpoint uses "
            f"{payload.dataset_schema_version!r}, evaluation requires "
            f"{expected_schema_version!r}. Retrain from scratch; schema "
            "migration of policy weights is intentionally unsupported."
        )


def require_single_dataset_schema(payloads) -> str:
    """Reject aggregate reports containing both legacy and v2 checkpoints."""

    schemas = {payload.dataset_schema_version for payload in payloads}
    if not schemas:
        raise ValueError("Cannot aggregate an empty checkpoint collection")
    if len(schemas) != 1:
        raise ValueError(f"Mixed dataset schemas are not aggregatable: {sorted(schemas)}")
    return next(iter(schemas))


def restore_numpy_rng(numpy_rng_state: Optional[tuple], fallback_seed: int) -> "np.random.RandomState":
    """Fix 5 (pre-P6 hardening): builds a ``numpy.random.RandomState``
    from a checkpointed ``numpy_rng_state`` if present, otherwise falls
    back to re-seeding from ``fallback_seed`` (matching the pre-Fix-5
    behavior for a checkpoint saved before this field existed).

    Args:
        numpy_rng_state: ``CheckpointPayload.numpy_rng_state`` -- either
            a tuple as returned by ``RandomState.get_state()``, or
            ``None`` (older checkpoint / never captured).
        fallback_seed: the run's ``seed`` (``CheckpointPayload.seed``),
            used only if ``numpy_rng_state`` is ``None``.

    Returns:
        A ``numpy.random.RandomState`` whose internal state exactly
        matches what the checkpointed run's ``numpy_rng`` held at
        checkpoint-save time (or, in the fallback case, a fresh stream
        seeded from ``fallback_seed``).
    """

    import numpy as np

    rng = np.random.RandomState(fallback_seed)
    if numpy_rng_state is not None:
        rng.set_state(numpy_rng_state)
    return rng
