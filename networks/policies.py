"""Actor-critic networks for the three PPO variants.

Every network maps observations to
    * a Gaussian distribution over the 2-D action (net acceleration, as a fraction of the limit),
    * a value estimate V(s) used by PPO's critic.

Tensor layout. All methods work on *sequences*: observations have shape [T, B, obs_dim], where T is
the number of control steps and B the number of parallel episodes. A single step is simply T = 1.
`episode_start` [T, B] is 1 where a new episode begins; recurrent networks reset their memory there,
so a sequence may contain several consecutive throws.

The action distribution is an unbounded Gaussian (mean from the network, state-independent
standard deviation). The environment clips actions to [-1, 1]; PPO uses the unclipped sample for
its log-probabilities, which is the usual choice for continuous-action PPO.
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from ball_catch.config import EnvConfig

ACTION_DIM = 2
BASE_OBS_DIM = 9  # observation without the timing inputs (EnvConfig.include_timing = False)
TIMED_OBS_DIM = 11  # with capture interval and observation age

# name -> (recurrent?, observation size). See the proposal, section 4.
VARIANTS = {
    "single_frame": (False, BASE_OBS_DIM),
    "recurrent": (True, BASE_OBS_DIM),
    "temporal": (True, TIMED_OBS_DIM),
}


class PolicyOutput(NamedTuple):
    """What a network returns when it acts in the environment (one control step)."""

    action: torch.Tensor  # [B, 2] sampled (or mean) action, before the environment clips it
    log_prob: torch.Tensor  # [B] log-probability of that action
    value: torch.Tensor  # [B] critic's estimate of the return
    state: Optional[torch.Tensor]  # recurrent memory after this step (None for the MLP)


def _layer(n_in: int, n_out: int, gain: float = np.sqrt(2.0)) -> nn.Linear:
    """Linear layer with orthogonal weights and zero bias, the standard initialisation for PPO."""
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _head(n_in: int, hidden: int, n_out: int, out_gain: float) -> nn.Sequential:
    """Two-layer MLP: n_in -> hidden -> n_out (tanh hidden activation)."""
    return nn.Sequential(_layer(n_in, hidden), nn.Tanh(), _layer(hidden, n_out, out_gain))


class ActorCritic(nn.Module):
    """Interface shared by all variants.

    Subclasses implement `forward(obs, state, episode_start) -> (action_mean, value, new_state)`.
    """

    recurrent = False

    def __init__(self, obs_dim: int, hidden_size: int, init_log_std: float = -0.5):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        # One learned standard deviation per action axis, independent of the observation.
        # exp(-0.5) = 0.61: early exploration covers most of the [-1, 1] action range.
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), float(init_log_std)))

    def initial_state(self, batch_size: int) -> Optional[torch.Tensor]:
        """Memory at the start of an episode (None for networks without memory)."""
        return None

    def distribution(self, mean: torch.Tensor) -> Normal:
        return Normal(mean, self.log_std.exp().expand_as(mean))

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        episode_start: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> PolicyOutput:
        """Choose actions for one control step. `obs` is [B, obs_dim]; returns [B, ...] tensors.

        With `deterministic=True` the action is the mean of the distribution (for evaluation).
        """
        if episode_start is not None:
            episode_start = episode_start.unsqueeze(0)
        mean, value, state = self(obs.unsqueeze(0), state, episode_start)
        dist = self.distribution(mean[0])
        action = dist.mean if deterministic else dist.sample()
        return PolicyOutput(action, dist.log_prob(action).sum(-1), value[0], state)

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        episode_start: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate a stored sequence for the PPO update, with gradients.

        obs [T, B, obs_dim], actions [T, B, 2] -> log_prob [T, B], entropy [T, B], value [T, B].
        `state` must be the memory at the start of the sequence, as it was during the rollout.
        """
        mean, value, _ = self(obs, state, episode_start)
        dist = self.distribution(mean)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), value


class MLPActorCritic(ActorCritic):
    """Single-frame baseline: separate actor and critic MLPs, no memory.

        obs -> 64 -> tanh -> 64 -> tanh -> action mean (2)
        obs -> 64 -> tanh -> 64 -> tanh -> value (1)
    """

    def __init__(self, obs_dim: int = BASE_OBS_DIM, hidden_size: int = 64, **kwargs):
        super().__init__(obs_dim, hidden_size, **kwargs)
        self.actor = nn.Sequential(
            _layer(obs_dim, hidden_size), nn.Tanh(), _head(hidden_size, hidden_size, ACTION_DIM, 0.01)
        )
        self.critic = nn.Sequential(
            _layer(obs_dim, hidden_size), nn.Tanh(), _head(hidden_size, hidden_size, 1, 1.0)
        )

    def forward(self, obs, state=None, episode_start=None):
        # Each step is processed independently; `state` and `episode_start` are ignored.
        return self.actor(obs), self.critic(obs).squeeze(-1), None


class RecurrentActorCritic(ActorCritic):
    """GRU policy for the recurrent and the temporal-aware variants.

        obs -> encoder (64, tanh) -> GRU cell (64) -> actor head -> action mean (2)
                                                   -> critic head -> value (1)

    The GRU is updated once per control step, so between camera frames it still sees the net
    state, the new-frame flag and (temporal variant) the growing observation age. Actor and critic
    share the encoder and the GRU, so one hidden state is carried through an episode.
    """

    recurrent = True

    def __init__(self, obs_dim: int, hidden_size: int = 64, **kwargs):
        super().__init__(obs_dim, hidden_size, **kwargs)
        self.encoder = nn.Sequential(_layer(obs_dim, hidden_size), nn.Tanh())
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.actor = _head(hidden_size, hidden_size, ACTION_DIM, 0.01)
        self.critic = _head(hidden_size, hidden_size, 1, 1.0)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size)

    def forward(self, obs, state=None, episode_start=None):
        steps, batch = obs.shape[:2]
        h = self.initial_state(batch) if state is None else state
        features = self.encoder(obs)  # [T, B, H], all steps at once
        outputs = []
        for t in range(steps):
            if episode_start is not None:
                # Forget everything at the first step of a new throw.
                h = h * (1.0 - episode_start[t].unsqueeze(-1))
            h = self.gru(features[t], h)
            outputs.append(h)
        memory = torch.stack(outputs)  # [T, B, H]
        return self.actor(memory), self.critic(memory).squeeze(-1), h


def make_policy(variant: str, hidden_size: int = 64, **kwargs) -> ActorCritic:
    """Build the network of one PPO variant: "single_frame", "recurrent" or "temporal"."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    recurrent, obs_dim = VARIANTS[variant]
    cls = RecurrentActorCritic if recurrent else MLPActorCritic
    return cls(obs_dim=obs_dim, hidden_size=hidden_size, **kwargs)


def env_config_for(variant: str, **overrides) -> EnvConfig:
    """Environment configuration whose observation matches the variant's input size."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    return EnvConfig(include_timing=VARIANTS[variant][1] == TIMED_OBS_DIM, **overrides)
