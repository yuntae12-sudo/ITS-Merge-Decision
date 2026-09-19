"""Discrete PPO core (docs/ppo/PPO_PLAN.md SS0.1 P3, SS6).

14D observation -> policy/value networks -> 4-way categorical
``BehaviorAction`` distribution (``KEEP=0, FOLLOW=1, MERGE=2, STOP=3``,
per SS11's fixed action-mapping regression).

As of P3, ``networks.py``, ``distribution.py``, ``policy.py``,
``loss.py``, and ``state.py`` all carry real Flax/Optax/JAX
implementations: policy/value networks, categorical sampling/log_prob/
entropy, the PPO clipped surrogate objective + value loss + entropy
regularization, and the Adam-with-gradient-clipping train state. This
package validates the PPO algorithm in isolation (synthetic inputs
only) -- real ``MergeEnvironment`` rollout integration is P4 scope
(``src/training/rollout.py``, ``src/training/gae.py``).
"""
