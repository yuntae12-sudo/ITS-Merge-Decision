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
    return CheckpointPayload(**raw)
