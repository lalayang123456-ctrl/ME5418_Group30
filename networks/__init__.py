"""Neural networks for the ME5418 ball-catching project.

Three actor-critic networks, one per PPO variant of the proposal:

    single_frame  MLP, 9 inputs   reactive baseline: latest observation, net state, arrival flag
    recurrent     GRU, 9 inputs   core comparator: memory, but no capture timestamps
    temporal      GRU, 11 inputs  main method: memory plus capture interval and observation age

`networks.ppo` holds what is needed to train them with PPO (rollouts, advantages, loss);
`python -m networks.demo` shows a forward and a backward pass of each network.
"""

from networks.policies import (
    VARIANTS,
    ActorCritic,
    MLPActorCritic,
    PolicyOutput,
    RecurrentActorCritic,
    env_config_for,
    make_policy,
)

__all__ = [
    "VARIANTS",
    "ActorCritic",
    "MLPActorCritic",
    "PolicyOutput",
    "RecurrentActorCritic",
    "env_config_for",
    "make_policy",
]
