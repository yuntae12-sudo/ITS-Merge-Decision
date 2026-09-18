"""PPO trainer loop: rollout -> GAE -> PPO update -> logging (docs/ppo/
PPO_PLAN.md SS0.1 P5).

P4 scope (this module): wires enough of the pipeline to produce ONE
complete PPO-ready training batch end-to-end -- rollout (P4:
``src.training.rollout``) -> Reward V0 (P2: ``src.rewards``) -> GAE
(P4: ``src.training.gae``) -> masked Actor-side advantage normalization
(SS7.2) -> a flat batch dict ready for a PPO update. The actual
multi-epoch parameter-UPDATE loop (gradient steps, minibatching across
epochs, checkpointing, W&B run orchestration) is explicit P5 scope
(SS0.1/P4's completion criteria: "the actual multi-update training loop
belongs to P5") and is NOT implemented here.

``build_training_batch`` is this module's P4 deliverable: given a
rollout's ``Transition`` list plus the PPO hyperparameters, it computes
GAE over the FULL physical trajectory (SS7.2 Critic scope) and returns
a dict whose ``policy_mask``-relevant fields (advantages used for the
policy loss) are ALREADY the masked-normalized ones, while ``returns``/
``values`` remain full-trajectory (Critic scope) -- exactly what a P5
PPO update loop will consume directly.
"""

from typing import Any, Dict, List

import numpy as np

from src.training.config import PPOConfig, RewardConfig
from src.training.gae import compute_gae, normalize_advantages_masked
from src.training.rollout import Transition


def build_training_batch(
    transitions: List[Transition], ppo_config: PPOConfig
) -> Dict[str, np.ndarray]:
    """Turns one rollout's flat ``Transition`` list into a PPO-ready
    training batch dict.

    Critic-side quantities (``returns``, ``values``, GAE's own
    ``advantages_raw``) are computed over the FULL physical trajectory,
    regardless of ``policy_mask`` (SS7.2). The batch's ``advantages``
    field is the Actor-side MASKED-NORMALIZED version (SS7.2: mean/std
    computed only over ``policy_mask == 1`` frames via
    ``normalize_advantages_masked``) -- a P5 update loop must further
    filter every Actor-side quantity (this batch's ``policy_mask``
    field, ``log_prob``, ``action``, ``observation``) down to
    ``policy_mask == 1`` rows before computing policy loss/entropy/KL/
    clip-fraction (SS7.2); this function does not do that filtering
    itself since the Critic-side arrays in the SAME batch must stay
    full-length.

    Returns a dict with keys: ``observation``, ``action``, ``reward``,
    ``next_observation``, ``terminated``, ``truncated``, ``value``,
    ``next_value``, ``log_prob``, ``policy_mask``, ``advantages_raw``
    (unmasked, full-trajectory GAE output -- kept for diagnostics/
    tests), ``advantages`` (masked-normalized, SS7.2), ``returns``
    (full-trajectory GAE returns, Critic target).
    """

    if len(transitions) == 0:
        raise ValueError("build_training_batch: got an empty transition list.")

    observations = np.stack([t.observation for t in transitions])
    next_observations = np.stack([t.next_observation for t in transitions])
    actions = np.asarray([t.action for t in transitions], dtype=np.int32)
    rewards = np.asarray([t.reward for t in transitions], dtype=np.float64)
    terminated = np.asarray([t.terminated for t in transitions], dtype=bool)
    truncated = np.asarray([t.truncated for t in transitions], dtype=bool)
    values = np.asarray([t.value for t in transitions], dtype=np.float64)
    next_values = np.asarray([t.next_value for t in transitions], dtype=np.float64)
    log_probs = np.asarray([t.log_prob for t in transitions], dtype=np.float64)
    policy_mask = np.asarray([t.policy_mask for t in transitions], dtype=np.int32)

    hp = ppo_config.hyperparameters
    gae_result = compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminated=terminated,
        truncated=truncated,
        gamma=hp.gamma,
        gae_lambda=hp.gae_lambda,
    )

    advantages_normalized = normalize_advantages_masked(
        gae_result.advantages, policy_mask
    )

    batch = {
        "observation": observations,
        "action": actions,
        "reward": rewards,
        "next_observation": next_observations,
        "terminated": terminated,
        "truncated": truncated,
        "value": values,
        "next_value": next_values,
        "log_prob": log_probs,
        "policy_mask": policy_mask,
        "advantages_raw": gae_result.advantages,
        "advantages": advantages_normalized,
        "returns": gae_result.returns,
    }

    for key, array in batch.items():
        if not np.all(np.isfinite(np.asarray(array, dtype=np.float64))):
            raise ValueError(
                f"build_training_batch: batch field {key!r} contains "
                "non-finite (NaN/inf) values."
            )

    return batch


def run_training(ppo_config: PPOConfig, reward_config: RewardConfig, *args, **kwargs):
    """Runs the full PPO training loop (rollout -> GAE -> update ->
    checkpoint -> W&B logging) for one config.

    P4 scope note: ``build_training_batch`` above (rollout -> GAE ->
    masked-normalized batch) is implemented and tested as of P4. The
    actual multi-epoch PARAMETER-UPDATE loop this function is meant to
    drive (minibatching across ``ppo_epochs``, applying gradients,
    checkpoint save/load/resume, W&B run orchestration) lands in P5 per
    PPO_PLAN.md SS0.1/P5 -- P4's explicit non-goal is "no actual
    multi-update training loop yet."
    """

    raise NotImplementedError(
        "The full PPO training loop (multi-epoch parameter updates, "
        "checkpointing, W&B run orchestration) lands in P5 (docs/ppo/"
        "PPO_PLAN.md SS0.1/P5). P4 implements and tests "
        "src.training.trainer.build_training_batch (rollout -> GAE -> "
        "masked-normalized PPO-ready batch), which this function will "
        "call once P5 wires in the update loop."
    )
