"""Running episodes and summarising them: shared by the demo, the calibration and later the
evaluation of trained policies."""

from typing import Callable, List, Optional

import numpy as np

from ball_catch.controllers import Controller
from ball_catch.env import BallCatchEnv


def run_episode(
    env: BallCatchEnv,
    controller: Controller,
    seed: Optional[int] = None,
    options: Optional[dict] = None,
    on_step: Optional[Callable[[BallCatchEnv], None]] = None,
) -> dict:
    """Play one episode and return its outcome and the latency the controller experienced.

    `on_step(env)` is called once after reset and once after every step (e.g. to grab frames).
    """
    obs, info = env.reset(seed=seed, options=options)
    controller.reset()
    if on_step is not None:
        on_step(env)

    episode_return, steps = 0.0, 0
    ages: List[float] = []  # observation age at every step at which something had been seen
    intervals: List[float] = []  # capture interval of every newly accepted frame
    while True:
        obs, reward, terminated, truncated, info = env.step(controller.act(obs, info))
        episode_return += reward
        steps += 1
        if info["frames_accepted"] > 0:
            ages.append(info["obs_age"])
        if info["new_frame"] and info["capture_interval"] > 0.0:
            intervals.append(info["capture_interval"])
        if on_step is not None:
            on_step(env)
        if terminated or truncated:
            break

    return {
        "caught": bool(info["caught"]),
        "timeout": bool(info["timeout"]),
        "miss_distance": float(info["miss_distance"]),
        "return": episode_return,
        "steps": steps,
        "flight_time": float(info["flight_time"]),
        "mean_obs_age": float(np.mean(ages)) if ages else float("nan"),
        "max_obs_age": float(np.max(ages)) if ages else float("nan"),
        "mean_capture_interval": float(np.mean(intervals)) if intervals else float("nan"),
        "frames_accepted": int(info["frames_accepted"]),
        "frames_discarded": int(info["frames_discarded"]),
    }


def evaluate(env: BallCatchEnv, controller: Controller, episodes: int, seed: int = 0) -> dict:
    """Play `episodes` episodes with seeds seed, seed + 1, ... and aggregate them.

    Because the environment's randomness depends only on the seed, two controllers evaluated
    with the same seed face exactly the same throws, sensor noise and delays.
    """
    results = [run_episode(env, controller, seed=seed + i) for i in range(episodes)]
    accepted = sum(r["frames_accepted"] for r in results)
    discarded = sum(r["frames_discarded"] for r in results)
    return {
        "episodes": episodes,
        "success_rate": float(np.mean([r["caught"] for r in results])),
        "mean_return": float(np.mean([r["return"] for r in results])),
        "mean_miss_distance": float(np.nanmean([r["miss_distance"] for r in results])),
        "mean_obs_age": float(np.nanmean([r["mean_obs_age"] for r in results])),
        "max_obs_age": float(np.nanmax([r["max_obs_age"] for r in results])),
        "mean_capture_interval": float(np.nanmean([r["mean_capture_interval"] for r in results])),
        "discard_fraction": discarded / max(accepted + discarded, 1),
        "results": results,
    }
