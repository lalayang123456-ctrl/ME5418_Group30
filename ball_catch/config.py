"""Single source of truth for every physical, sensing and reward parameter of the task.

All quantities are in SI units (metres, seconds). The world frame is right-handed:

    x : depth axis. The ball flies in +x, the net plane is x = 0, the camera sits at x > 0
        (behind the net) and looks towards -x.
    y : lateral axis, pointing to the right as seen from the camera.
    z : vertical axis, pointing up. Gravity acts along -z.

The default values are the starting point of the pilot calibration (see calibrate.py); they are
meant to be tuned there and then frozen before any policy is evaluated.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

Range = Tuple[float, float]


@dataclass
class EnvConfig:
    # ---------------------------------------------------------------- timing
    control_dt: float = 1.0 / 60.0  # control / simulation step (s)
    camera_rate: float = 30.0  # camera sampling frequency (Hz)
    randomize_camera_phase: bool = True  # random offset of the first capture within one period
    episode_time_limit: float = 2.0  # safety timeout (s); normal episodes end at plane crossing

    # ---------------------------------------------------------------- ball
    gravity: float = 9.81
    ball_radius: float = 0.07
    flight_time_range: Range = (0.6, 1.2)  # launch -> net-plane crossing (s)
    launch_distance_range: Range = (6.0, 9.0)  # distance of the launch point from the net plane
    launch_y_range: Range = (-1.5, 1.5)
    launch_z_range: Range = (0.8, 1.8)

    # ---------------------------------------------------------------- net
    net_radius: float = 0.30  # radius of the circular net opening
    workspace_y: Range = (-1.0, 1.0)  # reachable positions of the net centre
    workspace_z: Range = (0.3, 2.0)
    # Limits chosen in the pilot calibration (calibrate.py --sweep): an oracle still catches every
    # ball without delay, but loses about 12% of them when its information is 0.2 s late.
    net_a_max: float = 6.0  # per-axis acceleration limit (m/s^2)
    net_v_max: float = 3.0  # per-axis velocity limit (m/s); rarely reached within one flight

    # ---------------------------------------------------------------- camera
    camera_position: Tuple[float, float, float] = (2.0, 0.0, 1.15)
    image_width: int = 1280
    image_height: int = 960
    horizontal_fov_deg: float = 80.0
    pixel_noise_std: float = 1.0  # std of the detected ball centre (pixels)
    radius_noise_std: float = 0.5  # std of the detected apparent radius (pixels)

    # ---------------------------------------------------------------- delay
    delay_max: float = 0.20  # d_max: largest simulated delivery delay (s)
    jitter_mean: float = 0.08  # mean of the truncated-Gaussian jitter (s)
    jitter_std: float = 0.04

    # ---------------------------------------------------------------- reward
    catch_reward: float = 1.0  # base reward of any successful catch
    center_bonus: float = 0.5  # extra reward for catching exactly at the net centre
    miss_penalty: float = 1.0  # penalty of a miss at least `miss_distance_ref` outside the net
    miss_distance_ref: float = 1.0  # distance outside the net at which the miss penalty saturates
    timeout_penalty: float = 1.0
    control_cost: float = 0.05  # lambda: weight of the per-step control-effort penalty

    # ---------------------------------------------------------------- observation
    include_timing: bool = True  # append capture interval and observation age (main method)
    time_obs_clip: float = 5.0  # timing inputs are divided by delay_max and clipped to this value

    # ---------------------------------------------------------------- derived quantities
    @property
    def camera_period(self) -> float:
        return 1.0 / self.camera_rate

    @property
    def catch_radius(self) -> float:
        """Largest centre-to-centre distance at which the whole ball still fits in the opening."""
        return self.net_radius - self.ball_radius

    @property
    def workspace_low(self) -> np.ndarray:
        return np.array([self.workspace_y[0], self.workspace_z[0]], dtype=float)

    @property
    def workspace_high(self) -> np.ndarray:
        return np.array([self.workspace_y[1], self.workspace_z[1]], dtype=float)

    @property
    def workspace_center(self) -> np.ndarray:
        return 0.5 * (self.workspace_low + self.workspace_high)

    @property
    def workspace_half_extent(self) -> np.ndarray:
        return 0.5 * (self.workspace_high - self.workspace_low)

    @property
    def max_episode_steps(self) -> int:
        return int(np.ceil(self.episode_time_limit / self.control_dt))
