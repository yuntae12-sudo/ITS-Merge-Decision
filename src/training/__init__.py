"""PPO training package (docs/ppo/PPO_PLAN.md SS0.1 P1/P2/P4).

Config loading (``config.py``), seed handling (``seeding.py``), and the
checkpoint/resume contract skeleton (``checkpoint.py``) land in P1.
Rollout collection (``rollout.py``), GAE (``gae.py``), and the trainer
loop (``trainer.py``) are structural skeletons in P1 -- their real
logic lands in P4 (rollout/GAE) and P5 (trainer/update loop).
"""
