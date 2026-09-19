"""Generalized Advantage Estimation (docs/ppo/PPO_PLAN.md SS0.1 P4,
SS6, SS7.2; pre-P6 hardening Fix 1).

P4 implementation. GAE is computed over the FULL physical trajectory
(all frames, regardless of ``policy_mask`` -- SS7.2), with truncation
and true termination distinguished for bootstrapping, using the
baseline (not-tuned) ``gamma=0.99`` / ``gae_lambda=0.95`` (SS6).

Actor-side advantage normalization (mean/std) is computed only over
``policy_mask == 1`` frames -- that masking happens here via
``normalize_advantages_masked`` (kept in this module, not
``loss.py``, per P1's original placement decision, since GAE tests
SS7.2 A/B need to exercise this exact masking behavior directly), but
the GAE/return computation itself (``compute_gae``) always uses the
whole trajectory -- that is Critic-side and is never masked.

Standard backward-recursion GAE (Schulman et al. 2015, "High-
Dimensional Continuous Control Using Generalized Advantage
Estimation", eq. 16), matching the V-Max reference pattern
(PPO_PLAN.md SS3):

    delta_t   = r_t + gamma * V(s_{t+1}) * mask_t - V(s_t)
    A_t       = delta_t + gamma * lambda * mask_t * A_{t+1}
    return_t  = A_t + V(s_t)

where ``mask_t`` is the BOOTSTRAP mask for step t: ``0`` if step t
TERMINATED (true termination -- no value flows from beyond a terminal
state), ``1`` otherwise (including a TRUNCATED step, which still
bootstraps from ``next_value`` -- SS0.1/P4's required truncation
test). This bootstrap mask is unrelated to ``policy_mask``.
"""

from typing import Any, NamedTuple, Optional, Sequence

import numpy as np


class GAEResult(NamedTuple):
    advantages: np.ndarray
    returns: np.ndarray


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    next_values: Sequence[float],
    terminated: Sequence[bool],
    truncated: Sequence[bool],
    gamma: float,
    gae_lambda: float,
    rollout_cutoff: Optional[Sequence[bool]] = None,
) -> GAEResult:
    """Computes GAE advantages and returns over one full physical
    trajectory (all frames, regardless of ``policy_mask`` -- SS7.2).

    This function assumes its input is ONE contiguous episode segment
    (or, for backward compatibility, a single trajectory the caller has
    already verified never crosses an episode boundary). Callers with a
    multi-episode, flat-concatenated transition list (e.g.
    ``collect_rollout``'s output) MUST use ``compute_gae_segmented``
    instead, which calls this function independently per episode
    segment and concatenates results back in order (Fix 1, pre-P6
    hardening) -- calling this function directly over a multi-episode
    flat list risks a later episode's advantage leaking backward into
    an earlier truncated/cutoff episode's last steps, since the
    backward recursion below has no notion of an episode boundary
    other than the ``terminated``/``truncated``/``rollout_cutoff``
    flags it is given.

    Args:
        rewards: per-step reward, shape ``(T,)``.
        values: ``V(s_t)`` -- the value network's prediction for the
            observation BEFORE step t was taken, shape ``(T,)``.
        next_values: ``V(s_{t+1})`` -- the value network's prediction
            for the observation AFTER step t, shape ``(T,)``. For a
            terminated step this is not used for bootstrapping (masked
            to 0 internally) but must still be a finite number (e.g.
            whatever the value network happened to predict for the
            terminal observation, or 0.0) -- callers are not required
            to special-case it.
        terminated: per-step ``terminated`` flag from ``env.step``,
            shape ``(T,)``. A True at step t means step t reached a
            true terminal state (SUCCESS/COLLISION/OFFROAD) -- no
            bootstrap value flows past it.
        truncated: per-step ``truncated`` flag from ``env.step``,
            shape ``(T,)``. A True at step t means step t hit the
            episode horizon -- UNLIKE ``terminated``, this DOES
            bootstrap from ``next_values[t]`` (the trajectory was cut
            off, not actually over).
        gamma: discount factor (PPO_PLAN.md SS6: ``0.99``).
        gae_lambda: GAE lambda (PPO_PLAN.md SS6: ``0.95``).
        rollout_cutoff: optional per-step flag (Fix 1), shape ``(T,)``.
            A True at step t means step t is the LAST step of this
            trajectory because the rollout-COLLECTION loop ran out of
            steps, not because the environment itself reported
            terminated/truncated. Treated exactly like ``truncated``
            for bootstrapping purposes (DOES bootstrap from
            ``next_values[t]``) -- an artificial collection cutoff is
            never grounds to withhold the value estimate the Critic
            already has for the next observation. Defaults to all-False
            (no cutoff) for backward compatibility with existing
            single-episode callers/tests that never pass it.

    Returns:
        ``GAEResult(advantages, returns)``, each shape ``(T,)``.
    """

    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    if rollout_cutoff is None:
        rollout_cutoff = np.zeros_like(terminated, dtype=bool)
    else:
        rollout_cutoff = np.asarray(rollout_cutoff, dtype=bool)

    num_steps = rewards.shape[0]
    if not (
        values.shape[0] == num_steps
        and next_values.shape[0] == num_steps
        and terminated.shape[0] == num_steps
        and truncated.shape[0] == num_steps
        and rollout_cutoff.shape[0] == num_steps
    ):
        raise ValueError(
            "compute_gae: rewards/values/next_values/terminated/truncated/"
            "rollout_cutoff must all have the same length (one full "
            "physical trajectory/episode segment). Got lengths "
            f"{rewards.shape[0]}, {values.shape[0]}, "
            f"{next_values.shape[0]}, {terminated.shape[0]}, "
            f"{truncated.shape[0]}, {rollout_cutoff.shape[0]}."
        )

    # Bootstrap mask: 0.0 on true termination (no value beyond a
    # terminal state), 1.0 otherwise -- including truncation AND an
    # artificial rollout_cutoff (Fix 1), both of which DO bootstrap
    # from next_values (SS0.1/P4 required test; Fix 1 extends the same
    # rule to a trainer-side max_steps cutoff).
    bootstrap_mask = np.where(terminated, 0.0, 1.0)

    # Trace-continuation mask: whether the backward recursion may carry
    # gae_running from step t+1 into step t's advantage. This is ALWAYS
    # 1.0 within one call to compute_gae, because compute_gae now
    # assumes (per its docstring / compute_gae_segmented) that it is
    # only ever given ONE contiguous episode segment -- there is no
    # "next episode" inside a single call for a boundary to cut. The
    # cross-episode leakage this fix targets is instead prevented by
    # compute_gae_segmented calling this function separately per
    # segment (gae_running starts fresh at 0.0 for each), never by an
    # in-loop mask here.
    del truncated, rollout_cutoff  # already folded into bootstrap_mask via terminated only

    advantages = np.zeros(num_steps, dtype=np.float64)
    gae_running = 0.0
    for t in reversed(range(num_steps)):
        delta = (
            rewards[t]
            + gamma * next_values[t] * bootstrap_mask[t]
            - values[t]
        )
        gae_running = delta + gamma * gae_lambda * bootstrap_mask[t] * gae_running
        advantages[t] = gae_running

    returns = advantages + values

    if not (np.all(np.isfinite(advantages)) and np.all(np.isfinite(returns))):
        raise ValueError(
            "compute_gae produced non-finite advantages/returns -- check "
            "input rewards/values for NaN/inf."
        )

    return GAEResult(advantages=advantages, returns=returns)


def compute_gae_segmented(
    rewards: Sequence[float],
    values: Sequence[float],
    next_values: Sequence[float],
    terminated: Sequence[bool],
    truncated: Sequence[bool],
    episode_ids: Sequence[Any],
    gamma: float,
    gae_lambda: float,
    rollout_cutoff: Optional[Sequence[bool]] = None,
) -> GAEResult:
    """Computes GAE independently PER EPISODE SEGMENT of a
    flat, multi-episode-concatenated transition list (Fix 1, pre-P6
    hardening), then concatenates the per-episode results back in
    their ORIGINAL order.

    This is the correctness fix for ``collect_rollout``'s output: that
    function concatenates one or more independent episodes'
    transitions into a single flat list (each tagged with its own
    ``episode_id``), and calling ``compute_gae`` once over that whole
    flat list would let a truncated (or rollout_cutoff'd) episode's
    backward-recursion ``gae_running`` leak into -- i.e. get
    contaminated by -- whatever unrelated episode happens to follow it
    in the flat list, purely because the old backward recursion never
    reset at an episode boundary (it only ever looked at
    ``terminated``, never episode identity). Segmenting by
    ``episode_id`` and running ``compute_gae`` independently per
    segment structurally makes that leakage impossible: each segment's
    backward recursion starts its own ``gae_running = 0.0``, with no
    way to reference a value from outside its own segment.

    Args:
        episode_ids: per-step episode identifier, shape ``(T,)``.
            Transitions are grouped into contiguous runs of equal
            ``episode_ids`` value (matching ``collect_rollout``'s own
            construction, which never interleaves episodes) -- each run
            is GAE'd independently.
        rollout_cutoff: see ``compute_gae``. Also treated as an episode
            SEGMENT boundary here (Fix 1): even though a rollout_cutoff
            step keeps the same ``episode_id`` as the steps before it
            (the environment's own episode was never actually
            terminated/truncated), it is still the LAST step this
            function will ever see for that episode in this batch --
            since it's already the last transition collect_rollout
            produced for it, this has no additional segmenting effect
            beyond what ``episode_id`` grouping already does, but is
            documented here for clarity of intent.
        (remaining args: see ``compute_gae``.)

    Returns:
        ``GAEResult(advantages, returns)`` over the full flat input,
        each shape ``(T,)``, in the SAME order as the input.
    """

    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    episode_ids = np.asarray(episode_ids, dtype=object)
    num_steps = rewards.shape[0]
    if rollout_cutoff is None:
        rollout_cutoff_arr = np.zeros(num_steps, dtype=bool)
    else:
        rollout_cutoff_arr = np.asarray(rollout_cutoff, dtype=bool)

    if not (
        values.shape[0] == num_steps
        and next_values.shape[0] == num_steps
        and terminated.shape[0] == num_steps
        and truncated.shape[0] == num_steps
        and episode_ids.shape[0] == num_steps
        and rollout_cutoff_arr.shape[0] == num_steps
    ):
        raise ValueError(
            "compute_gae_segmented: all per-step arrays (rewards, values, "
            "next_values, terminated, truncated, episode_ids, "
            "rollout_cutoff) must have the same length. Got lengths "
            f"{rewards.shape[0]}, {values.shape[0]}, {next_values.shape[0]}, "
            f"{terminated.shape[0]}, {truncated.shape[0]}, "
            f"{episode_ids.shape[0]}, {rollout_cutoff_arr.shape[0]}."
        )
    if num_steps == 0:
        return GAEResult(
            advantages=np.zeros(0, dtype=np.float64),
            returns=np.zeros(0, dtype=np.float64),
        )

    advantages = np.zeros(num_steps, dtype=np.float64)
    returns = np.zeros(num_steps, dtype=np.float64)

    # Group into contiguous runs of equal episode_ids, preserving order
    # (collect_rollout never interleaves episodes, but this does not
    # assume that -- it only assumes each episode's own transitions are
    # contiguous, which is all collect_rollout ever produces).
    segment_start = 0
    for t in range(1, num_steps + 1):
        at_boundary = t == num_steps or episode_ids[t] != episode_ids[segment_start]
        if not at_boundary:
            continue
        segment_slice = slice(segment_start, t)
        segment_result = compute_gae(
            rewards=rewards[segment_slice],
            values=values[segment_slice],
            next_values=next_values[segment_slice],
            terminated=terminated[segment_slice],
            truncated=truncated[segment_slice],
            gamma=gamma,
            gae_lambda=gae_lambda,
            rollout_cutoff=rollout_cutoff_arr[segment_slice],
        )
        advantages[segment_slice] = segment_result.advantages
        returns[segment_slice] = segment_result.returns
        segment_start = t

    return GAEResult(advantages=advantages, returns=returns)


def normalize_advantages_masked(
    advantages: Sequence[float], policy_mask: Sequence[int], eps: float = 1e-8
) -> np.ndarray:
    """Normalizes advantages using only ``policy_mask == 1`` frames'
    mean/std (Actor-side normalization, SS7.2).

    Per SS7.2 test A: holding the set of ``policy_mask == 1``
    advantages fixed, changing ``policy_mask == 0`` frames' advantage
    VALUES must not change the computed mean/std, nor the resulting
    normalized values at ``policy_mask == 1`` positions. This is
    achieved by computing mean/std ONLY over the ``policy_mask == 1``
    subset -- masked-out entries never enter the reduction.

    Args:
        advantages: full-trajectory advantages, shape ``(T,)`` (as
            produced by ``compute_gae`` over the WHOLE trajectory).
        policy_mask: per-step ``policy_mask`` (1 = real decision step,
            0 = auto-executed MERGE-commitment step), shape ``(T,)``.
        eps: numerical-stability epsilon added to the std before
            dividing.

    Returns:
        An array shape ``(T,)`` where ``policy_mask == 1`` positions
        hold the normalized advantage (using the masked mean/std) and
        ``policy_mask == 0`` positions hold the ORIGINAL (unnormalized)
        advantage value unchanged (they are never consumed by any
        Actor-side computation anyway, per SS7.2, so their value here
        is a don't-care that we leave untouched rather than zeroing,
        to avoid implying any semantic meaning for a masked slot).
    """

    advantages = np.asarray(advantages, dtype=np.float64)
    policy_mask = np.asarray(policy_mask)

    if advantages.shape != policy_mask.shape:
        raise ValueError(
            "normalize_advantages_masked: advantages and policy_mask must "
            f"have the same shape, got {advantages.shape} vs "
            f"{policy_mask.shape}."
        )

    mask_bool = policy_mask.astype(bool)
    if not np.any(mask_bool):
        raise ValueError(
            "normalize_advantages_masked: policy_mask has no True entries "
            "-- cannot compute Actor-side normalization statistics over an "
            "empty set of decision frames."
        )

    decision_advantages = advantages[mask_bool]
    mean = float(np.mean(decision_advantages))
    std = float(np.std(decision_advantages))

    normalized = advantages.copy()
    normalized[mask_bool] = (decision_advantages - mean) / (std + eps)

    if not np.all(np.isfinite(normalized[mask_bool])):
        raise ValueError(
            "normalize_advantages_masked produced non-finite normalized "
            "advantages at a policy_mask==1 position."
        )

    return normalized


def masked_mean_std(advantages: Sequence[float], policy_mask: Sequence[int]):
    """Returns ``(mean, std)`` of ``advantages`` computed only over
    ``policy_mask == 1`` positions -- exposed standalone (in addition
    to being used internally by ``normalize_advantages_masked``) so
    SS7.2 test A can assert the statistic itself is unaffected by
    changes to masked-out frames, not just the final normalized
    output."""

    advantages = np.asarray(advantages, dtype=np.float64)
    policy_mask = np.asarray(policy_mask).astype(bool)
    decision_advantages = advantages[policy_mask]
    return float(np.mean(decision_advantages)), float(np.std(decision_advantages))
