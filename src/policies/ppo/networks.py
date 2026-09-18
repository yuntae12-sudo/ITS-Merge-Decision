"""PPO policy/value networks (docs/ppo/PPO_PLAN.md SS6, SS0.1 P3).

P1 scope: structural skeleton only. Fixed architecture (not tuned in
P0-P5):

    Policy: 14 -> 256 -> 64 -> 32 -> 4 logits, tanh activations
    Value:  14 -> 256 -> 64 -> 32 -> 1,        tanh activations

Policy and value networks are separate (no shared parameters) per
SS7.2's Actor-vs-Critic separation of concerns. Real Flax module
definitions land in P3; this module currently only fixes the layer
sizes so ``configs/ppo/*.yaml`` and this code agree from the start.
"""

from typing import Sequence

# Fixed baseline architecture (PPO_PLAN.md SS6). Not tuned in P0-P5.
POLICY_HIDDEN_SIZES: Sequence[int] = (256, 64, 32)
VALUE_HIDDEN_SIZES: Sequence[int] = (256, 64, 32)
OBSERVATION_DIM = 14
NUM_ACTIONS = 4
ACTIVATION = "tanh"


def build_policy_network(*args, **kwargs):
    """Builds the PPO policy (Actor) network.

    P1 skeleton: not yet implemented. Real Flax ``nn.Module`` definition
    lands in P3 per PPO_PLAN.md SS0.1/P3.
    """

    raise NotImplementedError(
        "PPO policy network construction lands in P3 (docs/ppo/PPO_PLAN.md "
        "SS0.1/P3). P1 only fixes the architecture constants."
    )


def build_value_network(*args, **kwargs):
    """Builds the PPO value (Critic) network.

    P1 skeleton: not yet implemented. Real Flax ``nn.Module`` definition
    lands in P3 per PPO_PLAN.md SS0.1/P3.
    """

    raise NotImplementedError(
        "PPO value network construction lands in P3 (docs/ppo/PPO_PLAN.md "
        "SS0.1/P3). P1 only fixes the architecture constants."
    )
