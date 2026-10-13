import math
import unittest

import numpy as np

from ball_catch.camera import PinholeCamera
from ball_catch.config import EnvConfig


class PinholeCameraTest(unittest.TestCase):
    def setUp(self):
        self.cfg = EnvConfig()
        self.camera = PinholeCamera(self.cfg)
        self.cx, self.cy, self.cz = self.cfg.camera_position

    def test_ball_on_optical_axis_projects_to_image_centre(self):
        features, valid = self.camera.observe(np.array([-5.0, self.cy, self.cz]))
        self.assertTrue(valid)
        np.testing.assert_allclose(features[:2], [0.0, 0.0], atol=1e-12)

    def test_area_is_one_when_ball_is_at_the_net_plane(self):
        features, _ = self.camera.observe(np.array([0.0, self.cy, self.cz]))
        self.assertAlmostEqual(features[2], 1.0)

    def test_area_follows_inverse_square_of_depth(self):
        twice_as_far = np.array([self.cx - 2.0 * self.camera.reference_depth, self.cy, self.cz])
        features, _ = self.camera.observe(twice_as_far)
        self.assertAlmostEqual(features[2], 0.25)

    def test_image_axes_point_right_and_up(self):
        features, _ = self.camera.observe(np.array([-3.0, self.cy + 0.5, self.cz + 0.5]))
        self.assertGreater(features[0], 0.0)
        self.assertGreater(features[1], 0.0)

    def test_normalised_coordinates_match_pinhole_model(self):
        ball = np.array([-3.0, 0.6, 1.9])
        depth = self.cx - ball[0]
        focal = 0.5 * self.cfg.image_width / math.tan(math.radians(self.cfg.horizontal_fov_deg) / 2)
        expected_u = focal * (ball[1] - self.cy) / depth / (0.5 * self.cfg.image_width)
        expected_v = focal * (ball[2] - self.cz) / depth / (0.5 * self.cfg.image_height)
        features, valid = self.camera.observe(ball)
        self.assertTrue(valid)
        np.testing.assert_allclose(features[:2], [expected_u, expected_v])

    def test_ball_outside_field_of_view_is_not_detected(self):
        features, valid = self.camera.observe(np.array([-1.0, 10.0, self.cz]))
        self.assertFalse(valid)
        np.testing.assert_array_equal(features, [0.0, 0.0, 0.0])

    def test_ball_behind_camera_is_not_detected(self):
        _, valid = self.camera.observe(np.array([self.cx + 1.0, self.cy, self.cz]))
        self.assertFalse(valid)

    def test_noise_perturbs_the_detection_reproducibly(self):
        ball = np.array([-6.0, 0.3, 1.4])
        clean, _ = self.camera.observe(ball)
        noisy_a, _ = self.camera.observe(ball, np.random.default_rng(3))
        noisy_b, _ = self.camera.observe(ball, np.random.default_rng(3))
        np.testing.assert_array_equal(noisy_a, noisy_b)
        self.assertFalse(np.allclose(noisy_a, clean))
        # One-pixel noise is a small perturbation in normalised image units.
        np.testing.assert_allclose(noisy_a[:2], clean[:2], atol=0.02)

    def test_no_noise_is_added_when_noise_std_is_zero(self):
        cfg = EnvConfig(pixel_noise_std=0.0, radius_noise_std=0.0)
        camera = PinholeCamera(cfg)
        ball = np.array([-6.0, 0.3, 1.4])
        clean, _ = camera.observe(ball)
        seeded, _ = camera.observe(ball, np.random.default_rng(3))
        np.testing.assert_array_equal(seeded, clean)

    def test_ray_to_plane_inverts_projection_at_the_net_plane(self):
        features, _ = self.camera.observe(np.array([0.0, 0.4, 0.8]))
        np.testing.assert_allclose(self.camera.ray_to_plane(features[0], features[1]), [0.4, 0.8])

    def test_whole_net_workspace_is_visible_at_the_net_plane(self):
        for y in self.cfg.workspace_y:
            for z in self.cfg.workspace_z:
                _, valid = self.camera.observe(np.array([0.0, y, z]))
                self.assertTrue(valid, msg=f"corner ({y}, {z}) is outside the image")


if __name__ == "__main__":
    unittest.main()
