"""The pieces of PPO that touch the networks: collecting rollouts, advantages, and the loss.

`RolloutCollector` plays a network in several environments at once; `compute_gae` and `ppo_loss`
turn what it collected into a loss. The training loop that repeats these steps (several epochs of
minibatch updates) is `agent.PPOAgent`; the networks demo uses the same pieces to show one forward
and one backward pass.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ball_catch.env import BallCatchEnv
from networks.policies import ActorCritic


@dataclass
class Rollout:
    """`steps` consecutive control steps of B environments, each possibly spanning several throws.

    Every tensor is [T, B, ...] so that it can be fed straight to `ActorCritic.evaluate`.
    """

    obs: torch.Tensor  # [T, B, obs_dim] observation the action was chosen from
    actions: torch.Tensor  # [T, B, 2] sampled actions (before the environment clips them)
    log_probs: torch.Tensor  # [T, B] their log-probabilities under the rollout policy
    values: torch.Tensor  # [T, B] critic estimates at collection time
    rewards: torch.Tensor  # [T, B]
    dones: torch.Tensor  # [T, B] 1 if the episode ended after this step
    episode_start: torch.Tensor  # [T, B] 1 if this step is the first of a throw
    initial_state: Optional[torch.Tensor]  # [B, H] network memory at the first step
    last_value: torch.Tensor  # [B] critic estimate after the last step (for bootstrapping)
    episodes: int  # number of throws that ended inside the rollout
    catches: int  # how many of those were caught
    episode_returns: List[float] = field(default_factory=list)  # return of each of those throws

    def select(self, envs: torch.Tensor) -> "Rollout":
        """The same rollout restricted to some environments (columns), e.g. for a minibatch.

        Each sequence stays whole, so a recurrent network can be re-run on it from its
        `initial_state`, exactly as during collection. The episode counts are not split.
        """
        return replace(
            self,
            obs=self.obs[:, envs],
            actions=self.actions[:, envs],
            log_probs=self.log_probs[:, envs],
            values=self.values[:, envs],
            rewards=self.rewards[:, envs],
            dones=self.dones[:, envs],
            episode_start=self.episode_start[:, envs],
            initial_state=None if self.initial_state is None else self.initial_state[envs],
            last_value=self.last_value[envs],
        )


class RolloutCollector:
    """Plays a policy in B environments in lockstep, with one batched network call per step.

    Environment i is seeded once with `seed + i` and then continues its own random stream, so
    successive rollouts see new throws. The throws and the network's memory carry over from one
    `collect` call to the next: a throw cut off by the end of a rollout continues in the next one.
    """

    def __init__(self, envs: Sequence[BallCatchEnv], policy: ActorCritic, seed: int = 0):
        self.envs = list(envs)
        self.policy = policy
        self.obs = np.stack([env.reset(seed=seed + i)[0] for i, env in enumerate(self.envs)])
        self.state = policy.initial_state(len(self.envs))
        self.start = torch.ones(len(self.envs))  # every environment begins a throw
        self._returns = np.zeros(len(self.envs))  # return so far of each running throw

    def collect(self, steps: int) -> Rollout:
        """Run `steps` control steps in every environment, resetting each one after its throw ends."""
        data = {key: [] for key in ("obs", "actions", "log_probs", "values", "rewards", "dones", "episode_start")}
        episode_returns: List[float] = []
        catches = 0
        initial_state = None if self.state is None else self.state.clone()

        for _ in range(steps):
            obs_t = torch.as_tensor(self.obs, dtype=torch.float32)  # [B, obs_dim]
            out = self.policy.act(obs_t, self.state, self.start)
            rewards = np.zeros(len(self.envs), dtype=np.float32)
            dones = np.zeros(len(self.envs), dtype=np.float32)
            next_obs = []
            for i, (env, action) in enumerate(zip(self.envs, out.action.numpy())):
                obs, reward, terminated, truncated, info = env.step(action)
                rewards[i] = reward
                self._returns[i] += reward
                if terminated or truncated:
                    dones[i] = 1.0
                    catches += int(info["caught"])
                    episode_returns.append(float(self._returns[i]))
                    self._returns[i] = 0.0
                    obs, _ = env.reset()  # continues the environment's random stream
                next_obs.append(obs)

            step = dict(obs=obs_t, actions=out.action, log_probs=out.log_prob, values=out.value,
                        rewards=torch.from_numpy(rewards), dones=torch.from_numpy(dones),
                        episode_start=self.start)
            for key, value in step.items():
                data[key].append(value)

            self.obs = np.stack(next_obs)
            self.state = out.state
            self.start = torch.from_numpy(dones.copy())  # a throw that ended is followed by a new one

        obs_t = torch.as_tensor(self.obs, dtype=torch.float32)
        last_value = self.policy.act(obs_t, self.state, self.start).value
        return Rollout(
            **{key: torch.stack(values) for key, values in data.items()},  # each [T, B, ...]
            initial_state=initial_state,
            last_value=last_value,
            episodes=len(episode_returns),
            catches=catches,
            episode_returns=episode_returns,
        )


def collect_rollout(env: BallCatchEnv, policy: ActorCritic, steps: int, seed: int = 0) -> Rollout:
    """Run `policy` in a single environment for `steps` control steps (B = 1)."""
    return RolloutCollector([env], policy, seed=seed).collect(steps)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation. All inputs are [T, B] (last_value [B]).

    Returns (advantages, returns), with returns = advantages + values as the critic's target.
    A done step does not bootstrap from the next value: the throw is over.
    """
    advantages = torch.zeros_like(rewards)
    next_value, next_advantage = last_value, torch.zeros_like(last_value)
    for t in reversed(range(rewards.shape[0])):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        next_advantage = delta + gamma * lam * not_done * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]
    return advantages, advantages + values


def ppo_loss(
    policy: ActorCritic,
    rollout: Rollout,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Clipped PPO objective on a rollout or a minibatch of it (`Rollout.select`); the sequences
    are kept intact for the GRU.

        loss = -E[min(r A, clip(r, 1 - eps, 1 + eps) A)] + c_v E[(V - R)^2] - c_e E[entropy]

    with r = pi_new(a|s) / pi_old(a|s). Advantages are normalised per batch.
    Returns the loss (with gradients) and its components as plain numbers, including an estimate
    of KL(pi_old || pi_new) that shows how far the update has moved the policy.
    """
    log_prob, entropy, value = policy.evaluate(
        rollout.obs, rollout.actions, rollout.initial_state, rollout.episode_start
    )
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    log_ratio = log_prob - rollout.log_probs
    ratio = torch.exp(log_ratio)
    policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv).mean()
    value_loss = ((value - returns) ** 2).mean()
    entropy_mean = entropy.mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
    stats = {
        "loss": float(loss),
        "policy_loss": float(policy_loss),
        "value_loss": float(value_loss),
        "entropy": float(entropy_mean),
        "clip_fraction": float(((ratio - 1.0).abs() > clip).float().mean()),
        "approx_kl": float(((ratio - 1.0) - log_ratio).mean()),  # >= 0, unbiased (Schulman, 2020)
    }
    return loss, stats


def gradient_norms(policy: ActorCritic) -> Dict[str, float]:
    """L2 norm of the gradient of each top-level part of the network (after `loss.backward()`)."""
    norms: Dict[str, float] = {}
    for name, param in policy.named_parameters():
        part = name.split(".")[0]
        g = 0.0 if param.grad is None else float(param.grad.norm() ** 2)
        norms[part] = norms.get(part, 0.0) + g
    return {part: float(np.sqrt(value)) for part, value in norms.items()}
