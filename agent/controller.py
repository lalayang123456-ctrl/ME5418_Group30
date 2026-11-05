"""A trained network behind the same interface as the hand-written controllers.

`NetworkController` lets `ball_catch.evaluation` and `demo.py` run a network exactly like the
chase or oracle controllers, so all of them can be compared on the same throws.
"""

import numpy as np
import torch

from ball_catch.controllers import Controller
from networks.policies import ActorCritic


class NetworkController(Controller):
    """Acts with a policy network, one control step at a time.

    The network sees the observation only, never the privileged `info`. A recurrent network keeps
    its memory from step to step; `reset` clears it at the start of every throw.
    With `deterministic=True` the action is the mean of the policy's Gaussian (for evaluation).
    """

    def __init__(self, policy: ActorCritic, deterministic: bool = True):
        self.policy = policy
        self.deterministic = deterministic
        self.reset()

    def reset(self) -> None:
        self.state = self.policy.initial_state(1)
        self._episode_start = torch.ones(1)

    def act(self, obs, info) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, obs_dim]
        out = self.policy.act(obs_t, self.state, self._episode_start, deterministic=self.deterministic)
        self.state = out.state
        self._episode_start = torch.zeros(1)
        return np.clip(out.action[0].numpy(), -1.0, 1.0)
