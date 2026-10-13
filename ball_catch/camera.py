"""A fixed, calibrated pinhole camera that reports where the ball appears and how large it looks.

The camera sits behind the net plane and looks along -x towards the incoming ball. Instead of
rendering images, the ball's projected centre and apparent area are computed analytically and
perturbed with pixel-level noise, which stands in for an image detector. Because the ball radius
is known, the apparent area carries depth information.
"""

import math
from typing import Optional, Tuple

import numpy as np

from ball_catch.config import EnvConfig

AREA_CLIP = 2.0  # upper bound of the normalised area feature


class PinholeCamera:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.position = np.asarray(cfg.camera_position, dtype=float)
        self.half_width = 0.5 * cfg.image_width
        self.half_height = 0.5 * cfg.image_height
        self.focal_px = self.half_width / math.tan(math.radians(cfg.horizontal_fov_deg) / 2.0)
        # Depth of the net plane. The apparent ball area there is the unit of the area feature.
        self.reference_depth = float(self.position[0])
        self.reference_radius_px = self.focal_px * cfg.ball_radius / self.reference_depth

    def project(self, ball_position: np.ndarray) -> Tuple[float, float, float, float]:
        """Noise-free projection: (u, v, apparent radius) in pixels and the depth in metres.

        u points right and v points up, both measured from the image centre.
        """
        depth = float(self.position[0] - ball_position[0])
        if depth <= 0.0:
            return 0.0, 0.0, 0.0, depth
        scale = self.focal_px / depth
        u = scale * float(ball_position[1] - self.position[1])
        v = scale * float(ball_position[2] - self.position[2])
        return u, v, scale * self.cfg.ball_radius, depth

    def observe(
        self, ball_position: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, bool]:
        """Detect the ball in one frame.

        Returns (features, valid) where features = [u, v, area]: the ball centre in normalised
        image coordinates in [-1, 1] and its apparent area relative to the area it has at the net
        plane. If the ball centre is outside the image the detection is invalid and the features
        are zero. Passing `rng` adds detector noise.
        """
        u, v, radius, depth = self.project(ball_position)
        visible = depth > 0.0 and abs(u) <= self.half_width and abs(v) <= self.half_height
        if not visible:
            return np.zeros(3), False
        if rng is not None:
            if self.cfg.pixel_noise_std > 0.0:
                u += rng.normal(0.0, self.cfg.pixel_noise_std)
                v += rng.normal(0.0, self.cfg.pixel_noise_std)
            if self.cfg.radius_noise_std > 0.0:
                radius = max(radius + rng.normal(0.0, self.cfg.radius_noise_std), 0.0)
        features = np.array(
            [
                np.clip(u / self.half_width, -1.0, 1.0),
                np.clip(v / self.half_height, -1.0, 1.0),
                min((radius / self.reference_radius_px) ** 2, AREA_CLIP),
            ]
        )
        return features, True

    def ray_to_plane(self, u: float, v: float, depth: Optional[float] = None) -> np.ndarray:
        """(y, z) where the line of sight through normalised pixel (u, v) meets the plane at the
        given depth in front of the camera (default: the net plane)."""
        depth = self.reference_depth if depth is None else depth
        y = self.position[1] + u * self.half_width * depth / self.focal_px
        z = self.position[2] + v * self.half_height * depth / self.focal_px
        return np.array([y, z])
