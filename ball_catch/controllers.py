"""Reference controllers.

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
    PredictiveController  Uses the observation only, including the capture timestamps. The
                          physics-based baseline of the proposal: a ballistic estimator predicts
                          where the ball will cross the net plane, a PD controller drives there.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from ball_catch.camera import PinholeCamera
from ball_catch.config import EnvConfig
from ball_catch.env import (
    OBS_AGE,
    OBS_AREA,
    OBS_NET_POS,
    OBS_NET_VEL,
    OBS_NEW_FRAME,
    OBS_U,
    OBS_V,
    OBS_VALID,
)


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


class BallisticEstimator:
    """Predicts when and where the ball crosses the net plane from timestamped detections.

    Each accepted frame gives the ball's image position (u, v) and apparent area. With the camera
    calibration and the known ball radius, the area gives the depth (area = (d_net / depth)^2,
    where d_net is the camera's distance to the net plane) and (u, v) the direction, hence a 3-D
    position at the frame's *capture* time. A drag-free trajectory with known gravity,

        x(t) = x0 + vx t,   y(t) = y0 + vy t,   z(t) = z0 + vz t - g t^2 / 2,

    is fitted to all positions by weighted least squares and extrapolated to the plane x = 0.
    Depth from apparent size is the noisy part: its error grows with depth^2, so each frame is
    weighted by 1 / depth^4 (inverse variance).
    """

    def __init__(self, cfg: EnvConfig, min_frames: int = 3):
        if min_frames < 2:
            raise ValueError("a trajectory needs at least two frames")
        self.cfg = cfg
        self.camera = PinholeCamera(cfg)
        self.min_frames = min_frames
        self.reset()

    def reset(self) -> None:
        self.times: List[float] = []
        self.positions: List[np.ndarray] = []

    def backproject(self, u: float, v: float, area: float) -> np.ndarray:
        """3-D ball position (x, y, z) of a detection: normalised image position and area."""
        depth = self.camera.reference_depth / math.sqrt(max(float(area), 1e-9))
        y, z = self.camera.ray_to_plane(float(u), float(v), depth)
        return np.array([self.camera.position[0] - depth, y, z])

    def add(self, capture_time: float, u: float, v: float, area: float) -> None:
        self.times.append(float(capture_time))
        self.positions.append(self.backproject(u, v, area))

    def predict(self) -> Optional[Tuple[float, np.ndarray]]:
        """(crossing time, (y, z) crossing point), or None without enough frames or if the
        fitted ball does not fly towards the net."""
        if len(self.times) < self.min_frames:
            return None
        t = np.asarray(self.times)
        p = np.asarray(self.positions)
        depth = self.camera.position[0] - p[:, 0]
        sqrt_w = 1.0 / np.maximum(depth, 1.0) ** 2  # square root of the 1 / depth^4 weights
        design = np.stack([np.ones_like(t), t], axis=1) * sqrt_w[:, None]
        g = self.cfg.gravity
        targets = np.stack([p[:, 0], p[:, 1], p[:, 2] + 0.5 * g * t ** 2], axis=1) * sqrt_w[:, None]
        (x0, y0, z0), (vx, vy, vz) = np.linalg.lstsq(design, targets, rcond=None)[0]
        if vx <= 1e-6:
            return None
        crossing_time = -x0 / vx
        point = np.array([y0 + vy * crossing_time, z0 + vz * crossing_time - 0.5 * g * crossing_time ** 2])
        return float(crossing_time), point


class PredictiveController(Controller):
    """Non-privileged, physics-based baseline: ballistic prediction plus a PD controller.

    It reads the observation only, with the timing inputs: the capture time of a newly accepted
    frame is the current time minus its observation age, so delayed and out-of-order frames are
    placed correctly on the trajectory. The control time is counted by the controller itself
    (one call per control step), so `info` is never used.

    The net is driven to the latest predicted crossing point (clipped to the workspace) by

        a = kp (target - position) - kd velocity,

    expressed as a fraction of the acceleration limit and clipped to [-1, 1]. Until the first
    prediction the target is the net's own position: it holds still.

    The gains were chosen by a grid search on training throws only (seeds 0-499, jitter delay;
    `python tune_predictive.py`): kp 80 1/s^2 and kd 20 1/s caught 63.6% of those throws, the best
    of kp 10-160, kd 4-40 and 3 or 5 frames before the first prediction. The held-out throws used
    to compare controllers start at seed 100000.
    """

    def __init__(self, cfg: EnvConfig, kp: float = 80.0, kd: float = 20.0, min_frames: int = 3):
        if not cfg.include_timing:
            raise ValueError("the predictive controller needs the capture timestamps "
                             "(EnvConfig(include_timing=True))")
        self.cfg = cfg
        self.kp, self.kd = kp, kd
        self.estimator = BallisticEstimator(cfg, min_frames)
        self.reset()

    def reset(self) -> None:
        self.estimator.reset()
        self.target: Optional[np.ndarray] = None
        self._steps = 0

    def act(self, obs, info):
        cfg = self.cfg
        now = self._steps * cfg.control_dt
        self._steps += 1
        pos = cfg.workspace_center + np.asarray(obs[OBS_NET_POS], dtype=float) * cfg.workspace_half_extent
        vel = np.asarray(obs[OBS_NET_VEL], dtype=float) * cfg.net_v_max
        if obs[OBS_NEW_FRAME] > 0.5 and obs[OBS_VALID] > 0.5:
            capture_time = now - float(obs[OBS_AGE]) * cfg.delay_max
            self.estimator.add(capture_time, obs[OBS_U], obs[OBS_V], obs[OBS_AREA])
            prediction = self.estimator.predict()
            if prediction is not None:
                self.target = np.clip(prediction[1], cfg.workspace_low, cfg.workspace_high)
        target = pos if self.target is None else self.target
        accel = self.kp * (target - pos) - self.kd * vel
        return np.clip(accel / cfg.net_a_max, -1.0, 1.0)


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
    if name == "predictive":
        return PredictiveController(cfg)
    raise ValueError(f"unknown controller: {name!r}")
