"""Reward V0 computation (docs/ppo/PPO_PLAN.md SS5, SS5.1, SS0.1 P2).

P2 implementation. This module maps the frozen ``MergeEnvironment``'s
own ``terminated``/``truncated``/``info["termination_reason"]`` to the
fixed Reward V0 terminal-outcome table, and adds a pre-step decision
cost. It never re-derives SUCCESS/COLLISION/OFFROAD/TIMEOUT itself --
that detection is exclusively the frozen ``MergeEnvironment``'s job
(``src/environment/termination.py``). This module only consumes the
already-computed ``info["termination_reason"]`` string value (see
``MergeEnvironment._build_info``, which sets it to
``termination_reason.value if termination_reason else None`` -- i.e.
one of ``"success"``, ``"failure_collision"``, ``"failure_offroad"``,
``"truncation_horizon"``, ``"none"``, or ``None``).

Reward V0::

    R = terminal_outcome + decision_cost

Terminal-outcome table (``TerminationReason`` values, from
``src/environment/termination.py:48-53``, the only five values it
defines)::

    TerminationReason.SUCCESS            -> +1.0
    TerminationReason.FAILURE_COLLISION  -> -1.0
    TerminationReason.FAILURE_OFFROAD    -> -1.0
    TerminationReason.TRUNCATION_HORIZON -> -0.5
    TerminationReason.NONE               -> 0.0

Decision cost: ``-0.01`` on a real PPO decision step
(pre-step ``is_policy_step is True``), ``0.0`` on an auto-executed
MERGE-commitment step (``is_policy_step is False``). This flag MUST be
computed by the caller BEFORE calling ``env.step()`` (PPO_PLAN.md
SS7.1) and passed in here -- this module never infers it from
post-step state (e.g. it never looks at ``info["merge_committed"]``
itself to decide the cost).

Explicitly forbidden inside this module (SS5.1): a merge-success
detector, an overlap/collision detector, an offroad detector, an
episode-timeout detector. This module only performs a fixed dict
lookup against the ``termination_reason`` string the environment
already computed.

Reward V1 (outputs/reward_v1_spec/REWARD_V1_SPEC_FINAL.md, approved)
adds three dense per-step components on top of V0's unchanged terminal +
decision_cost::

    R = terminal_outcome + decision_cost
        + w_safety   * r_safety
        + w_progress * r_progress
        + w_decision * r_decision

All three are computed ONLY from information already present in the 14D
observation vector (``src.environment.observation_builder.
OBSERVATION_FIELD_NAMES``, indices mirrored locally below as named
constants -- this module intentionally never imports
``src.environment``, per this file's own regression test
``test_reward_module_does_not_reimplement_termination_detection``) plus
the current/previous discrete action -- never future/logged-outcome
data. A ``RewardConfig`` with ``safety``/``progress``/``decision`` all
``None`` (i.e. a V0 config) makes every one of these three components
contribute exactly 0.0, so V0's ``R = terminal + decision_cost`` behavior
is completely unchanged when a V1-only argument is omitted.

r_safety (target-front TTC/Gap ONLY -- final approved scope, see
REWARD_V1_SPEC_FINAL.md Section 5): the gap<=0 "already overlapping"
sentinel (``_compute_ttc`` in ``src.scenarios.scenario_features``) is
NEVER treated as an automatic maximum penalty -- a targeted geometry
audit found 0/20 confirmed real physical overlaps among sustained
negative-gap candidates. Both the TTC and the Gap term are masked to 0.0
whenever ``target_front_present`` is 0.0, or whenever the gap<=0 sentinel
precondition holds.

r_progress (potential-difference form, NOT gamma-discounted -- a
deliberate simplification, not a claim of exact RL-theory
policy-invariant shaping): ``Phi(s_next) - Phi(s_current)``, where
``Phi = clip(1 - d_m / d_m_initial, 0, 1)`` reuses the observation's own
``d_m`` field (index 1) -- ``d_m_initial`` is a per-episode constant the
caller must track and pass in (this module is a pure function of its
arguments; it does not itself hold per-episode state -- see
``src.rewards.reward_wrapper.MergeRewardWrapper`` for the stateful
lifecycle).

r_decision (action-switching/chattering regularizer, NOT the same as the
unchanged V0 ``decision_cost``): ``-1.0`` if ``is_policy_step`` is True
AND a previous real policy action exists AND it differs from the current
action; ``0.0`` otherwise (including every ``is_policy_step=False``
auto-execution step, and the first real policy decision of an episode).
"""

import math
from typing import Any, Dict, Optional

from src.training.config import RewardConfig

# Mirrors src.environment.observation_builder.OBSERVATION_FIELD_NAMES'
# indices exactly (NOT imported -- see module docstring/regression test
# above for why this module must never import src.environment).
_OBS_D_M = 1
_OBS_TARGET_FRONT_PRESENT = 2
_OBS_TARGET_FRONT_GAP = 3
_OBS_TARGET_FRONT_TTC = 5


def _r_safety_breakdown(reward_config: RewardConfig, observation) -> Dict[str, float]:
    """Returns ``{"r_ttc": ..., "r_gap": ..., "r_safety": ...}`` (raw,
    unweighted) -- target-front-only TTC/Gap safety shaping (final
    approved scope). All three are 0.0 if ``reward_config.safety`` is
    None (V0) or ``observation`` is None."""

    safety = reward_config.safety
    if safety is None or observation is None:
        return {"r_ttc": 0.0, "r_gap": 0.0, "r_safety": 0.0}
    if safety.use_target_rear or safety.use_source_front:
        raise NotImplementedError(
            "merge_reward._r_safety_breakdown: use_target_rear/"
            "use_source_front are reserved config fields, not yet "
            "implemented -- the approved Reward V1 scope "
            "(REWARD_V1_SPEC_FINAL.md Section 5) is target-front only."
        )
    if not safety.use_target_front:
        return {"r_ttc": 0.0, "r_gap": 0.0, "r_safety": 0.0}

    present = float(observation[_OBS_TARGET_FRONT_PRESENT])
    gap = float(observation[_OBS_TARGET_FRONT_GAP])
    ttc = float(observation[_OBS_TARGET_FRONT_TTC])

    if present == 0.0 or gap <= 0.0:
        # Absent target-front vehicle, OR the gap<=0 "already overlapping"
        # sentinel (see module docstring) -- masked for BOTH the TTC and
        # the Gap term, never a penalty.
        return {"r_ttc": 0.0, "r_gap": 0.0, "r_safety": 0.0}

    r_ttc = _ttc_shaping(ttc, safety.ttc_danger_s, safety.ttc_safe_s)
    r_gap = _gap_shaping(gap, safety.gap_sufficient_m)
    return {"r_ttc": r_ttc, "r_gap": r_gap, "r_safety": min(r_ttc, r_gap)}


def _r_safety(reward_config: RewardConfig, observation) -> float:
    """Convenience wrapper returning only the aggregated ``r_safety``
    value -- see ``_r_safety_breakdown`` for the raw TTC/Gap components."""

    return _r_safety_breakdown(reward_config, observation)["r_safety"]


def _ttc_shaping(ttc: float, danger_s: float, safe_s: float) -> float:
    """Piecewise-linear: 0 at/above danger_s..safe_s's safe end, -1 at/
    below danger_s, linear between. TTC is the observation's own
    TTC_CAP_S-capped value (see observation_builder.py) -- a "no closing
    vehicle"/very-safe reading saturates at the observation's own cap,
    which is far above ``safe_s``, so it naturally returns 0.0 here
    without any special-cased +Inf handling."""

    if ttc >= safe_s:
        return 0.0
    if ttc <= danger_s:
        return -1.0
    return -(safe_s - ttc) / (safe_s - danger_s)


def _gap_shaping(gap: float, sufficient_m: float) -> float:
    """Piecewise-linear: 0 at/above sufficient_m, approaching -1 as gap
    shrinks toward 0 (the gap<=0 sentinel case is masked by the caller
    BEFORE this function is reached -- see ``_r_safety``)."""

    if gap >= sufficient_m:
        return 0.0
    return -(sufficient_m - gap) / sufficient_m


def compute_progress_potential(d_m: float, d_m_initial: float, epsilon_m: float) -> float:
    """Phi(s) = clip(1 - d_m/d_m_initial, 0, 1). Degenerate case
    (``d_m_initial <= epsilon_m``): Phi is defined as 1.0 for the whole
    episode (ego was already essentially at the merge point at reset --
    there is no meaningful remaining distance to normalize against, and
    this avoids a division-by-near-zero blowing up into an artificial
    large reward)."""

    if d_m_initial <= epsilon_m:
        return 1.0
    phi = 1.0 - (d_m / d_m_initial)
    return min(1.0, max(0.0, phi))


def _r_progress(reward_config: RewardConfig, observation, next_observation, d_m_initial: Optional[float]) -> float:
    """Potential-difference progress shaping: Phi(s_next) - Phi(s). See
    module docstring for why no gamma factor is applied. Returns 0.0 if
    ``reward_config.progress`` is None (V0), or either observation /
    ``d_m_initial`` is unavailable (e.g. the very first call of an
    episode before a next_observation exists)."""

    progress = reward_config.progress
    if progress is None or observation is None or next_observation is None or d_m_initial is None:
        return 0.0
    if progress.use_gamma:
        raise NotImplementedError(
            "merge_reward._r_progress: use_gamma=True is a reserved config "
            "field, not yet implemented -- the approved Reward V1 formula "
            "(REWARD_V1_SPEC_FINAL.md Section 6) is the plain "
            "Phi(s')-Phi(s) potential-difference form."
        )

    d_m_current = float(observation[_OBS_D_M])
    d_m_next = float(next_observation[_OBS_D_M])
    phi_current = compute_progress_potential(d_m_current, d_m_initial, progress.d_m_initial_epsilon_m)
    phi_next = compute_progress_potential(d_m_next, d_m_initial, progress.d_m_initial_epsilon_m)
    return phi_next - phi_current


def _r_decision(
    reward_config: RewardConfig,
    is_policy_step: bool,
    action,
    previous_policy_action,
) -> float:
    """Action-switching/chattering regularizer. Returns 0.0 if
    ``reward_config.decision`` is None (V0), if this is not a real
    policy decision step, or if there is no previous policy action to
    compare against (the first real decision of an episode)."""

    decision = reward_config.decision
    if decision is None:
        return 0.0
    if not decision.switching_only:
        raise NotImplementedError(
            "merge_reward._r_decision: switching_only=False is a reserved "
            "config field, not yet implemented -- the approved Reward V1 "
            "decision component only penalizes action switches."
        )
    if not is_policy_step:
        return 0.0
    if previous_policy_action is None:
        return 0.0
    if action == previous_policy_action:
        return 0.0
    return -1.0


# String keys are exactly the ``TerminationReason.value`` strings
# defined in src/environment/termination.py:48-53. No other values are
# ever produced by the frozen MergeEnvironment.
_TERMINATION_REASON_VALUES = (
    "success",
    "failure_collision",
    "failure_offroad",
    "truncation_horizon",
    "none",
)


def _terminal_outcome(reward_config: RewardConfig, termination_reason: Optional[str]) -> float:
    """Looks up the fixed terminal-outcome value for one termination
    reason string. Pure table lookup -- no detection logic.

    ``termination_reason`` may be ``None`` (mirrors
    ``MergeEnvironment``'s own ``info["termination_reason"]`` before
    any ``TerminationResult`` has ever been computed for this episode,
    e.g. at ``reset()`` time) -- treated the same as
    ``TerminationReason.NONE`` ("no terminal outcome yet").
    """

    if termination_reason is None:
        termination_reason = "none"

    if termination_reason not in _TERMINATION_REASON_VALUES:
        raise ValueError(
            "merge_reward: unrecognized termination_reason "
            f"{termination_reason!r}. This module only maps the fixed "
            "TerminationReason enum values from "
            "src/environment/termination.py:48-53 "
            f"({_TERMINATION_REASON_VALUES}); it must never receive or "
            "invent any other value (PPO_PLAN.md SS5.1)."
        )

    terminal = reward_config.terminal
    return {
        "success": terminal.success,
        "failure_collision": terminal.failure_collision,
        "failure_offroad": terminal.failure_offroad,
        "truncation_horizon": terminal.truncation_horizon,
        "none": terminal.none,
    }[termination_reason]


def _decision_cost(reward_config: RewardConfig, is_policy_step: bool) -> float:
    """Looks up the fixed decision-cost value for one step, selected
    purely by the caller-supplied pre-step ``is_policy_step`` flag
    (PPO_PLAN.md SS7.1/SS5.1) -- never derived from post-step state.
    """

    cost = reward_config.decision_cost
    return cost.real_decision_step if is_policy_step else cost.auto_execution_step


def compute_reward(
    reward_config: RewardConfig,
    termination_reason: str,
    is_policy_step: bool,
    info: Dict[str, Any] = None,
    observation: Any = None,
    next_observation: Any = None,
    action: Any = None,
    previous_policy_action: Any = None,
    d_m_initial: Optional[float] = None,
) -> float:
    """Computes one step's Reward value (V0 or V1, selected purely by
    which fields ``reward_config`` carries -- see module docstring).

    Args:
        reward_config: loaded via
            ``src.training.config.load_reward_config`` -- the sole
            source of the fixed V0/V1 numeric values (never hardcoded a
            second time in this function body).
        termination_reason: the frozen ``MergeEnvironment``'s own
            ``info["termination_reason"]`` value for this step (a
            ``TerminationReason.value`` string, or ``None``). This
            function performs a fixed lookup against this value; it
            never recomputes it.
        is_policy_step: pre-step decision flag (PPO_PLAN.md SS7.1) --
            ``True`` on a real PPO decision step, ``False`` on an
            auto-executed MERGE-commitment step. Must be computed by
            the caller BEFORE ``env.step()`` from
            ``info_before["merge_committed"]`` (``is_policy_step = not
            info_before["merge_committed"]``); this function does not
            infer it itself.
        info: accepted for call-signature symmetry with
            ``MergeRewardWrapper.compute`` and future callers that may
            want to pass the full step ``info`` dict through for
            diagnostics; NOT read for reward computation itself (this
            module needs only the explicit arguments above) -- kept
            unused here rather than silently reading extra fields out
            of it, since doing so would risk exactly the kind of
            independent re-derivation SS5.1 forbids.
        observation, next_observation: the 14D observation vectors
            (pre-step / post-step) needed by Reward V1's Safety/Progress
            components. Ignored entirely for a V0 ``reward_config``
            (``reward_config.safety``/``progress`` are both None).
        action, previous_policy_action: the current step's executed
            action and the previous REAL policy decision's action
            (``None`` if no previous policy decision exists this
            episode), needed by Reward V1's Decision component. Ignored
            entirely for a V0 ``reward_config``.
        d_m_initial: this episode's ``d_m`` value at reset (a per-episode
            constant the caller must track -- see
            ``src.rewards.reward_wrapper.MergeRewardWrapper`` for the
            stateful lifecycle). Needed by Reward V1's Progress
            component; ignored for V0.

    Returns:
        ``terminal_outcome + decision_cost`` (V0), or additionally
        ``+ w_safety*r_safety + w_progress*r_progress + w_decision*
        r_decision`` (V1), as a finite Python float.
    """

    del info  # unused -- see docstring; kept for signature symmetry.

    breakdown = compute_reward_breakdown(
        reward_config=reward_config,
        termination_reason=termination_reason,
        is_policy_step=is_policy_step,
        observation=observation,
        next_observation=next_observation,
        action=action,
        previous_policy_action=previous_policy_action,
        d_m_initial=d_m_initial,
    )
    return breakdown["total"]


def compute_reward_breakdown(
    reward_config: RewardConfig,
    termination_reason: str,
    is_policy_step: bool,
    observation: Any = None,
    next_observation: Any = None,
    action: Any = None,
    previous_policy_action: Any = None,
    d_m_initial: Optional[float] = None,
) -> Dict[str, float]:
    """Same computation as ``compute_reward``, but returns every
    component (raw and weighted) rather than only the total -- used by
    ``MergeRewardWrapper`` for W&B reward-breakdown logging (never
    recomputed/estimated a second, divergent way downstream)."""

    terminal_outcome = _terminal_outcome(reward_config, termination_reason)
    decision_cost = _decision_cost(reward_config, is_policy_step)

    safety_breakdown = _r_safety_breakdown(reward_config, observation)
    r_progress = _r_progress(reward_config, observation, next_observation, d_m_initial)
    r_decision = _r_decision(reward_config, is_policy_step, action, previous_policy_action)

    w_safety = reward_config.safety.weight if reward_config.safety is not None else 0.0
    w_progress = reward_config.progress.weight if reward_config.progress is not None else 0.0
    w_decision = reward_config.decision.weight if reward_config.decision is not None else 0.0

    weighted_safety = w_safety * safety_breakdown["r_safety"]
    weighted_progress = w_progress * r_progress
    weighted_decision = w_decision * r_decision

    total = (
        float(terminal_outcome)
        + float(decision_cost)
        + weighted_safety
        + weighted_progress
        + weighted_decision
    )

    if not math.isfinite(total):
        raise ValueError(
            f"merge_reward.compute_reward_breakdown produced a non-finite "
            f"reward ({total!r}) for termination_reason="
            f"{termination_reason!r}, is_policy_step={is_policy_step!r}. "
            "This should be unreachable given the fixed reward config "
            "values."
        )

    return {
        "terminal": float(terminal_outcome),
        "decision_cost": float(decision_cost),
        "safety_weighted": weighted_safety,
        "progress_weighted": weighted_progress,
        "decision_weighted": weighted_decision,
        "ttc_penalty_raw": safety_breakdown["r_ttc"],
        "gap_penalty_raw": safety_breakdown["r_gap"],
        "total": total,
    }
