"""The predictive baseline: a timestamp-aware ballistic estimator and a PD controller (proposal,
section 2, "Predictive control")."""

import unittest

import numpy as np

from ball_catch.camera import PinholeCamera
from ball_catch.config import EnvConfig
from ball_catch.controllers import BallisticEstimator, PredictiveController, make_controller
from ball_catch.dynamics import BallLaunch
from ball_catch.env import BallCatchEnv
from ball_catch.evaluation import evaluate


def quiet_config(**overrides):
    settings = dict(pixel_noise_std=0.0, radius_noise_std=0.0, randomize_camera_phase=False)
    settings.update(overrides)
    return EnvConfig(**settings)


def play_without_info(env, controller, **reset_kwargs):
    """Play one throw, giving the controller the observation only (never the privileged info)."""
    obs, info = env.reset(**reset_kwargs)
    controller.reset()
    while True:
        obs, _, terminated, truncated, info = env.step(controller.act(obs, None))
        if terminated or truncated:
            return info


class BallisticEstimatorTest(unittest.TestCase):
    def setUp(self):
        self.cfg = quiet_config()
        self.camera = PinholeCamera(self.cfg)
        self.estimator = BallisticEstimator(self.cfg)

    def test_backprojects_a_noise_free_detection_to_the_ball_position(self):
        for position in ([-5.0, 0.4, 1.6], [-8.5, -1.2, 0.9], [-0.5, 0.3, 1.0]):
            features, valid = self.camera.observe(np.array(position))
            self.assertTrue(valid)
            np.testing.assert_allclose(self.estimator.backproject(*features), position, atol=1e-6)

    def test_predicts_the_crossing_point_from_a_few_noise_free_frames(self):
        launch = BallLaunch.from_target((-8.0, 0.5, 1.2), (-0.6, 1.5), 1.0, self.cfg.gravity)
        for k in range(5):
            t = k * self.cfg.camera_period
            self.estimator.add(t, *self.camera.observe(launch.position_at(t))[0])
        crossing_time, point = self.estimator.predict()
        self.assertAlmostEqual(crossing_time, 1.0, places=4)
        np.testing.assert_allclose(point, [-0.6, 1.5], atol=1e-3)

    def test_needs_enough_frames_before_predicting(self):
        launch = BallLaunch.from_target((-8.0, 0.0, 1.2), (0.0, 1.2), 1.0, self.cfg.gravity)
        self.assertIsNone(self.estimator.predict())
        self.estimator.add(0.0, *self.camera.observe(launch.position_at(0.0))[0])
        self.assertIsNone(self.estimator.predict())

    def test_reset_forgets_the_previous_throw(self):
        launch = BallLaunch.from_target((-8.0, 0.0, 1.2), (0.0, 1.2), 1.0, self.cfg.gravity)
        for k in range(5):
            t = k * self.cfg.camera_period
            self.estimator.add(t, *self.camera.observe(launch.position_at(t))[0])
        self.estimator.reset()
        self.assertIsNone(self.estimator.predict())


class PredictiveControllerTest(unittest.TestCase):
    def test_needs_the_timing_inputs(self):
        with self.assertRaises(ValueError):
            PredictiveController(EnvConfig(include_timing=False))

    def test_make_controller_builds_it(self):
        self.assertIsInstance(make_controller("predictive", EnvConfig()), PredictiveController)

    def test_holds_still_until_a_ball_has_been_seen(self):
        cfg = EnvConfig()
        env = BallCatchEnv(cfg, delay="none")
        controller = PredictiveController(cfg)
        obs, _ = env.reset(seed=0)
        controller.reset()
        np.testing.assert_array_equal(controller.act(obs, None), [0.0, 0.0])

    def test_catches_a_high_arc_that_fools_the_reactive_controller(self):
        cfg = quiet_config()
        target = cfg.workspace_center + np.array([0.3, 0.2])
        launch = BallLaunch.from_target((-8.0, 0.0, 1.2), target, 1.2, cfg.gravity)
        info = play_without_info(BallCatchEnv(cfg, delay="none"), PredictiveController(cfg),
                                 seed=0, options={"launch": launch})
        self.assertTrue(info["caught"])

    def test_compensates_for_a_fixed_delay(self):
        cfg = quiet_config()
        target = cfg.workspace_center + np.array([-0.5, -0.3])
        launch = BallLaunch.from_target((-7.0, 1.0, 1.5), target, 1.0, cfg.gravity)
        info = play_without_info(BallCatchEnv(cfg, delay=0.15), PredictiveController(cfg),
                                 seed=0, options={"launch": launch})
        self.assertTrue(info["caught"])

    def test_catches_most_balls_with_noise_and_jitter(self):
        cfg = EnvConfig()
        env = BallCatchEnv(cfg, delay="jitter")
        predictive = evaluate(env, PredictiveController(cfg), 150, seed=0)["success_rate"]
        chase = evaluate(env, make_controller("chase", cfg), 150, seed=0)["success_rate"]
        self.assertGreater(predictive, 0.5)
        self.assertGreater(predictive, chase + 0.3)


if __name__ == "__main__":
    unittest.main()
