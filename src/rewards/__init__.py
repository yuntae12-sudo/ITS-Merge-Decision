"""PPO reward package (docs/ppo/PPO_PLAN.md SS5 / SS0.1 P2).

P1 scope: package marker only. ``merge_reward.py`` (Reward V0
computation) and ``reward_wrapper.py`` (environment-facing wrapper) are
structural skeletons in P1 -- their real logic lands in P2. Reward code
must consume the frozen ``MergeEnvironment``'s own ``terminated`` /
``truncated`` / ``info["termination_reason"]`` as sole source of truth
(SS5.1) and must never re-derive success/collision/offroad/timeout
itself.
"""
