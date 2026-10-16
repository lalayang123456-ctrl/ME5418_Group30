"""Hand-written reference controllers.

They serve three purposes: driving the demo, checking that the environment behaves sensibly, and
calibrating the task (how hard is it with perfect information, and how much does delay cost?).
None of them is learned.

    RandomController      uniformly random accelerations.
    StationaryController  never moves: the success rate of doing nothing.
    OracleController      PRIVILEGED. Reads the true interception point from `info` and drives
                          there. An upper bound on what any policy can achieve; with a
                          `reaction_time` it bounds what is achievable when information is late.
    ChaseController       Uses the observation only. Moves the net to where the line of sight to
                          the last seen ball position meets the net plane. A purely reactive
                          strategy that ignores gravity and delay.
"""

from typing import Optional

import numpy as np

from ball_catch.camera import PinholeCamera
from ball_catch.config import EnvConfig
from ball_catch.env import OBS_NET_POS, OBS_NET_VEL, OBS_U, OBS_V, OBS_VALID


def oracle_reaction_time(cfg: EnvConfig, delay: float) -> float:
    """Earliest time (s) at which the interception point could be known from the camera.

    The first frames take `delay` seconds to arrive, and two frames are the minimum needed to
    estimate a velocity, so a fair privileged oracle waits for the delay plus two camera periods.
    """
    return delay + 2.0 * cfg.camera_period


def move_to(
    target: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    cfg: EnvConfig,
    brake_fraction: float = 0.8,
) -> np.ndarray:
    """Near time-optimal action that brings the net to rest at `target`.

    Per axis, the desired speed is the largest one from which the net can still brake to a stop
    at the target (using `brake_fraction` of the acceleration limit, to leave some margin), capped
    by the velocity limit. The action is the acceleration that reaches this speed in one step.
    """
    error = np.asarray(target, dtype=float) - pos
    distance = np.abs(error)
    speed = np.minimum(cfg.net_v_max, np.sqrt(2.0 * brake_fraction * cfg.net_a_max * distance))
    speed = np.minimum(speed, 0.5 * distance / cfg.control_dt)  # do not overshoot within one step
    accel = (np.sign(error) * speed - vel) / cfg.control_dt
    return np.clip(accel / cfg.net_a_max, -1.0, 1.0)


class Controller:
    """Maps (observation, info) to an action in [-1, 1]^2."""

    def reset(self) -> None:
        """Called at the start of every episode."""

    def act(self, obs, info) -> np.ndarray:
        raise NotImplementedError


class RandomController(Controller):
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, info):
        return self.rng.uniform(-1.0, 1.0, size=2)


class StationaryController(Controller):
    def act(self, obs, info):
        return np.zeros(2)


class OracleController(Controller):
    """Privileged reference: knows exactly where the ball will cross the net plane."""

    def __init__(self, cfg: EnvConfig, reaction_time: float = 0.0):
        self.cfg = cfg
        self.reaction_time = reaction_time  # the oracle stays idle until this time (s)

    def act(self, obs, info):
        if info["time"] < self.reaction_time:
            return np.zeros(2)
        return move_to(info["intercept"], info["net_position"], info["net_velocity"], self.cfg)


class ChaseController(Controller):
    """Reactive, non-privileged reference: follow the line of sight to the ball."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.camera = PinholeCamera(cfg)
        self._target: Optional[np.ndarray] = None

    def reset(self):
        self._target = None

    def act(self, obs, info):
        cfg = self.cfg
        net_pos = cfg.workspace_center + np.asarray(obs[OBS_NET_POS], dtype=float) * cfg.workspace_half_extent
        net_vel = np.asarray(obs[OBS_NET_VEL], dtype=float) * cfg.net_v_max
        if obs[OBS_VALID] > 0.5:
            line_of_sight = self.camera.ray_to_plane(float(obs[OBS_U]), float(obs[OBS_V]))
            self._target = np.clip(line_of_sight, cfg.workspace_low, cfg.workspace_high)
        # Until the ball has been seen, hold position.
        target = net_pos if self._target is None else self._target
        return move_to(target, net_pos, net_vel, cfg)


def make_controller(
    name: str, cfg: EnvConfig, seed: Optional[int] = None, delay: float = 0.0
) -> Controller:
    """Build a controller by name. `delay` (s) is the typical frame delay of the episode; only the
    oracle uses it, to wait until the information it is given could have reached a camera-based
    controller (see `oracle_reaction_time`)."""
    if name == "random":
        return RandomController(seed)
    if name == "stationary":
        return StationaryController()
    if name == "oracle":
        return OracleController(cfg, oracle_reaction_time(cfg, delay))
    if name == "chase":
        return ChaseController(cfg)
    raise ValueError(f"unknown controller: {name!r}")
