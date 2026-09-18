"""Discrete PPO core (docs/ppo/PPO_PLAN.md SS0.1 P3, SS6).

14D observation -> policy/value networks -> 4-way categorical
``BehaviorAction`` distribution (``KEEP=0, FOLLOW=1, MERGE=2, STOP=3``,
per SS11's fixed action-mapping regression). P1 scope: package marker
and structural skeletons only (``networks.py``, ``distribution.py``,
``policy.py``, ``loss.py``, ``state.py``) -- real network/loss/optimizer
logic lands in P3.
"""
