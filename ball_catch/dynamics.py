"""Ball and net dynamics.

The ball is a drag-free, spin-free projectile, so its state at any time is known in closed form
and never accumulates integration error. The net is a per-axis double integrator with
acceleration, velocity and workspace limits, integrated exactly over each step.
"""

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from ball_catch.config import EnvConfig


@dataclass(frozen=True)
class BallLaunch:
    """A ballistic trajectory, defined by the ball state at launch (t = 0)."""

    position: np.ndarray  # launch position (x, y, z); x < 0, in front of the net plane
    velocity: np.ndarray  # launch velocity; the x component is positive (towards the net)
    gravity: float

    def position_at(self, t: float) -> np.ndarray:
        p = self.position + self.velocity * t
        p[2] -= 0.5 * self.gravity * t * t
        return p

    def velocity_at(self, t: float) -> np.ndarray:
        v = np.array(self.velocity, dtype=float)
        v[2] -= self.gravity * t
        return v

    @property
    def flight_time(self) -> float:
        """Time at which the ball centre crosses the net plane x = 0."""
        return float(-self.position[0] / self.velocity[0])

    @property
    def intercept(self) -> np.ndarray:
        """(y, z) of the ball centre when it crosses the net plane."""
        return self.position_at(self.flight_time)[1:]

    @classmethod
    def from_target(
        cls,
        position: Sequence[float],
        target: Sequence[float],
        flight_time: float,
        gravity: float,
    ) -> "BallLaunch":
        """Launch from `position` so that the ball crosses the net plane at `target` = (y, z)
        exactly `flight_time` seconds later."""
        position = np.asarray(position, dtype=float)
        velocity = np.array(
            [
                -position[0] / flight_time,
                (target[0] - position[1]) / flight_time,
                (target[1] - position[2]) / flight_time + 0.5 * gravity * flight_time,
            ]
        )
        return cls(position=position, velocity=velocity, gravity=gravity)


def sample_launch(rng: np.random.Generator, cfg: EnvConfig) -> BallLaunch:
    """Draw a random launch.

    The flight time and the crossing point are sampled directly and the launch velocity is solved
    for. This keeps the two quantities that set the difficulty of an episode (how long the net has
    and how far it must travel) under explicit control. Since the trajectory is a downward-opening
    parabola, the ball never drops below the lower of its launch and crossing heights.
    """
    flight_time = rng.uniform(*cfg.flight_time_range)
    position = np.array(
        [
            -rng.uniform(*cfg.launch_distance_range),
            rng.uniform(*cfg.launch_y_range),
            rng.uniform(*cfg.launch_z_range),
        ]
    )
    target = rng.uniform(cfg.workspace_low, cfg.workspace_high)
    return BallLaunch.from_target(position, target, flight_time, cfg.gravity)


def _step_axis(p: float, v: float, a: float, dt: float, v_max: float, lo: float, hi: float):
    """Advance one axis of the net by `dt` under constant acceleration `a`."""
    v_free = v + a * dt
    if abs(v_free) <= v_max:
        p_new = p + v * dt + 0.5 * a * dt * dt
        v_new = v_free
    else:
        # The velocity limit is reached part-way through the step: accelerate, then coast.
        v_new = math.copysign(v_max, v_free)
        t_acc = (v_new - v) / a
        p_new = p + v * t_acc + 0.5 * a * t_acc * t_acc + v_new * (dt - t_acc)
    # Workspace walls are inelastic: the net stops where it hits them.
    if p_new <= lo:
        p_new, v_new = lo, max(v_new, 0.0)
    elif p_new >= hi:
        p_new, v_new = hi, min(v_new, 0.0)
    return p_new, v_new


def step_net(
    pos: np.ndarray, vel: np.ndarray, accel: np.ndarray, dt: float, cfg: EnvConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance the net centre (y, z) by `dt` under a commanded acceleration (m/s^2).

    The command is clipped to the acceleration limit; returns new (position, velocity) arrays.
    """
    accel = np.clip(accel, -cfg.net_a_max, cfg.net_a_max)
    low, high = cfg.workspace_low, cfg.workspace_high
    new_pos, new_vel = np.empty(2), np.empty(2)
    for i in range(2):
        new_pos[i], new_vel[i] = _step_axis(
            float(pos[i]), float(vel[i]), float(accel[i]), dt, cfg.net_v_max, low[i], high[i]
        )
    return new_pos, new_vel
