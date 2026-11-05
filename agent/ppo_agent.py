"""The learning agent: PPO training of one of the three networks on the ball-catching environment.

    agent = PPOAgent(TrainConfig(variant="temporal", total_steps=1_000_000))
    agent.train(on_update=lambda agent, stats: print(stats["steps"], stats["catch_rate"]))
    agent.save("runs/temporal/model.pt")
    print(agent.evaluate(episodes=500)["success_rate"])

Each update
  1. collects `rollout_steps` control steps in each of `num_envs` environments (`RolloutCollector`),
  2. computes advantages with GAE,
  3. runs `epochs` passes of minibatch PPO updates over that data. A minibatch is a group of
     environments with their whole sequences, so a recurrent network is re-run from the memory it
     had at the start of the rollout, exactly as during collection,
  4. adds the rollout's observations to the network's input statistics (`normalize_obs`).
"""

import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from agent.controller import NetworkController
from ball_catch.env import BallCatchEnv
from ball_catch.evaluation import evaluate
from networks.policies import VARIANTS, ActorCritic, env_config_for, make_policy
from networks.ppo import RolloutCollector, compute_gae, ppo_loss

# First seed of the held-out evaluation throws. Training environment i of run `seed` is seeded with
# seed * num_envs + i, far below this, so evaluation never starts from a throw used in training.
EVAL_SEED = 100_000


@dataclass
class TrainConfig:
    """Everything that defines a training run. Saved with the model, so a run can be reproduced."""

    variant: str = "temporal"  # "single_frame", "recurrent" or "temporal"
    delay: Union[str, float] = "jitter"  # delay model of the training environments
    total_steps: int = 3_000_000  # control steps over all environments
    seed: int = 0
    num_envs: int = 16  # environments played in parallel
    rollout_steps: int = 128  # control steps per environment per update (about 2.4 throws)
    epochs: int = 10  # passes over each rollout
    minibatches: int = 4  # minibatches per pass; each holds num_envs / minibatches environments
    lr: float = 3e-4  # Adam learning rate, decayed linearly to 0 if anneal_lr
    anneal_lr: bool = True
    gamma: float = 0.995  # the reward arrives at the end of a 36-72 step throw: 0.995^72 = 0.70
    lam: float = 0.95  # GAE lambda
    clip: float = 0.2  # PPO ratio clipping
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5  # gradients are clipped to this global norm
    hidden_size: int = 64
    normalize_obs: bool = True  # standardise the inputs with running statistics (ObservationNorm)

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; choose from {sorted(VARIANTS)}")
        if self.num_envs % self.minibatches != 0:
            raise ValueError("num_envs must be a multiple of minibatches")

    @property
    def batch_steps(self) -> int:
        """Control steps collected per update."""
        return self.num_envs * self.rollout_steps


class PPOAgent:
    """A policy network, the environments it plays in, and the PPO update that trains it."""

    def __init__(self, config: TrainConfig):
        self.config = config
        torch.manual_seed(config.seed)
        self.policy = make_policy(config.variant, hidden_size=config.hidden_size)
        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=config.lr, eps=1e-5)
        envs = [BallCatchEnv(env_config_for(config.variant), delay=config.delay)
                for _ in range(config.num_envs)]
        self.collector = RolloutCollector(envs, self.policy, seed=config.seed * config.num_envs)
        self.num_updates = math.ceil(config.total_steps / config.batch_steps)
        self.updates = 0
        self.steps = 0

    def update(self) -> Dict[str, float]:
        """One PPO iteration: collect a rollout, then learn from it. Returns training statistics."""
        cfg = self.config
        started = time.perf_counter()
        lr = cfg.lr * (1.0 - self.updates / self.num_updates) if cfg.anneal_lr else cfg.lr
        for group in self.optimiser.param_groups:
            group["lr"] = lr

        rollout = self.collector.collect(cfg.rollout_steps)
        advantages, returns = compute_gae(
            rollout.rewards, rollout.values, rollout.dones, rollout.last_value, cfg.gamma, cfg.lam
        )

        envs_per_batch = cfg.num_envs // cfg.minibatches
        losses: List[Dict[str, float]] = []
        for _ in range(cfg.epochs):
            order = torch.randperm(cfg.num_envs)
            for first in range(0, cfg.num_envs, envs_per_batch):
                envs = order[first:first + envs_per_batch]
                loss, stats = ppo_loss(
                    self.policy, rollout.select(envs), advantages[:, envs], returns[:, envs],
                    cfg.clip, cfg.value_coef, cfg.entropy_coef,
                )
                self.optimiser.zero_grad()
                loss.backward()
                stats["grad_norm"] = float(nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm))
                self.optimiser.step()
                losses.append(stats)
        if cfg.normalize_obs:
            # Only now, so the rollout was collected and re-evaluated with the same statistics.
            self.policy.obs_norm.update(rollout.obs)

        self.updates += 1
        self.steps += cfg.batch_steps
        return {
            "update": self.updates,
            "steps": self.steps,
            "episodes": rollout.episodes,
            # Throws that ended during this rollout, played with the stochastic (exploring) policy.
            "catch_rate": rollout.catches / rollout.episodes if rollout.episodes else float("nan"),
            "mean_return": float(np.mean(rollout.episode_returns)) if rollout.episodes else float("nan"),
            # Averages over every minibatch step of this update.
            **{key: float(np.mean([s[key] for s in losses])) for key in losses[0]},
            "action_std": float(self.policy.log_std.exp().mean()),
            "lr": lr,
            "seconds": time.perf_counter() - started,
        }

    def train(self, on_update: Optional[Callable[["PPOAgent", Dict[str, float]], None]] = None) -> List[Dict[str, float]]:
        """Update until `total_steps` control steps have been collected.

        `on_update(agent, stats)` is called after every update (logging, evaluation, checkpoints).
        Returns the statistics of every update.
        """
        history = []
        while self.updates < self.num_updates:
            stats = self.update()
            history.append(stats)
            if on_update is not None:
                on_update(self, stats)
        return history

    def controller(self, deterministic: bool = True) -> NetworkController:
        """The current network as a controller for `ball_catch.evaluation` and `demo.py`."""
        return NetworkController(self.policy, deterministic)

    def evaluate(self, episodes: int = 200, seed: int = EVAL_SEED, delay: Union[str, float, None] = None) -> dict:
        """Success rate etc. of the deterministic policy (mean action) on held-out throws.

        Throws, sensor noise and delays depend only on the seeds, so other controllers evaluated
        with the same `seed` and `delay` face exactly the same episodes.
        """
        env = BallCatchEnv(env_config_for(self.config.variant),
                           delay=self.config.delay if delay is None else delay)
        return evaluate(env, self.controller(), episodes, seed=seed)

    def save(self, path: str) -> None:
        """Write the network weights and the training configuration (read back with `load_policy`)."""
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        torch.save({"config": asdict(self.config), "steps": self.steps,
                    "policy": self.policy.state_dict()}, path)


def load_policy(path: str) -> Tuple[ActorCritic, TrainConfig]:
    """Rebuild a trained network from a file written by `PPOAgent.save`."""
    checkpoint = torch.load(path, weights_only=True)
    config = TrainConfig(**checkpoint["config"])
    policy = make_policy(config.variant, hidden_size=config.hidden_size)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    return policy, config
