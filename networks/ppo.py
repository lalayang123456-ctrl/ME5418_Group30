"""The pieces of PPO that touch the networks: collecting a rollout, advantages, and the loss.

The full training loop (several epochs of minibatch updates, logging, checkpoints) is the next
milestone; these functions are what it will be built from, and what the demo uses to show a
forward and a backward pass on real environment data.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ball_catch.env import BallCatchEnv
from networks.policies import ActorCritic


@dataclass
class Rollout:
    """`steps` consecutive control steps of one environment (B = 1), possibly several throws.

    Every tensor is [T, 1, ...] so that it can be fed straight to `ActorCritic.evaluate`.
    """

    obs: torch.Tensor  # [T, 1, obs_dim] observation the action was chosen from
    actions: torch.Tensor  # [T, 1, 2] sampled actions (before the environment clips them)
    log_probs: torch.Tensor  # [T, 1] their log-probabilities under the rollout policy
    values: torch.Tensor  # [T, 1] critic estimates at collection time
    rewards: torch.Tensor  # [T, 1]
    dones: torch.Tensor  # [T, 1] 1 if the episode ended after this step
    episode_start: torch.Tensor  # [T, 1] 1 if this step is the first of a throw
    initial_state: Optional[torch.Tensor]  # network memory at the first step
    last_value: torch.Tensor  # [1] critic estimate after the last step (for bootstrapping)
    episodes: int  # number of throws that ended inside the rollout
    catches: int  # how many of those were caught


def collect_rollout(env: BallCatchEnv, policy: ActorCritic, steps: int, seed: int = 0) -> Rollout:
    """Run `policy` in `env` for `steps` control steps, resetting after every throw."""
    data = {key: [] for key in ("obs", "actions", "log_probs", "values", "rewards", "dones", "episode_start")}
    episodes = catches = 0

    obs, _ = env.reset(seed=seed)
    state = policy.initial_state(1)
    initial_state = None if state is None else state.clone()
    start = torch.ones(1)
    for _ in range(steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, obs_dim]
        out = policy.act(obs_t, state, start)
        obs, reward, terminated, truncated, info = env.step(out.action[0].numpy())
        done = terminated or truncated

        step = dict(obs=obs_t, actions=out.action, log_probs=out.log_prob, values=out.value,
                    rewards=torch.tensor([reward], dtype=torch.float32),
                    dones=torch.tensor([float(done)]), episode_start=start)
        for key, value in step.items():
            data[key].append(value)

        state = out.state
        start = torch.zeros(1)
        if done:
            episodes += 1
            catches += int(info["caught"])
            obs, _ = env.reset()  # continues the environment's random stream
            start = torch.ones(1)

    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    last_value = policy.act(obs_t, state, start).value
    return Rollout(
        **{key: torch.stack(values) for key, values in data.items()},  # each [T, 1, ...]
        initial_state=initial_state,
        last_value=last_value,
        episodes=episodes,
        catches=catches,
    )


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
    """Clipped PPO objective on a whole rollout (the sequence is kept intact for the GRU).

        loss = -E[min(r A, clip(r, 1 - eps, 1 + eps) A)] + c_v E[(V - R)^2] - c_e E[entropy]

    with r = pi_new(a|s) / pi_old(a|s). Advantages are normalised per batch.
    Returns the loss (with gradients) and its components as plain numbers.
    """
    log_prob, entropy, value = policy.evaluate(
        rollout.obs, rollout.actions, rollout.initial_state, rollout.episode_start
    )
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ratio = torch.exp(log_prob - rollout.log_probs)
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
