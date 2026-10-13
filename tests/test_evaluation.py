import math
import unittest

import numpy as np

from ball_catch.config import EnvConfig
from ball_catch.controllers import OracleController, StationaryController
from ball_catch.dynamics import BallLaunch
from ball_catch.env import BallCatchEnv
from ball_catch.evaluation import evaluate, run_episode


def quiet_config(**overrides):
    settings = dict(pixel_noise_std=0.0, radius_noise_std=0.0, randomize_camera_phase=False)
    settings.update(overrides)
    return EnvConfig(**settings)


class RunEpisodeTest(unittest.TestCase):
    def setUp(self):
        self.cfg = quiet_config()
        self.centre_launch = BallLaunch.from_target(
            (-7.5, 0.0, 1.2), self.cfg.workspace_center, 0.905, self.cfg.gravity
        )

    def test_reports_the_outcome_of_the_episode(self):
        env = BallCatchEnv(self.cfg, delay="none")
        result = run_episode(
            env, StationaryController(), seed=0, options={"launch": self.centre_launch}
        )
        self.assertTrue(result["caught"])
        self.assertFalse(result["timeout"])
        self.assertAlmostEqual(result["miss_distance"], 0.0)
        self.assertAlmostEqual(result["return"], self.cfg.catch_reward + self.cfg.center_bonus)
        self.assertEqual(result["steps"], math.ceil(0.905 * 60))
        self.assertAlmostEqual(result["flight_time"], 0.905)

    def test_reports_the_latency_the_controller_experienced(self):
        env = BallCatchEnv(self.cfg, delay=0.1)
        result = run_episode(
            env, StationaryController(), seed=0, options={"launch": self.centre_launch}
        )
        # With a 0.1 s delay and frames every second control step, the age alternates between
        # 0.1 s and 0.1 s plus one control step.
        self.assertAlmostEqual(result["max_obs_age"], 0.1 + self.cfg.control_dt, places=6)
        self.assertGreater(result["mean_obs_age"], 0.1)
        self.assertLess(result["mean_obs_age"], 0.1 + self.cfg.control_dt)
        self.assertEqual(result["frames_discarded"], 0)
        self.assertGreater(result["frames_accepted"], 10)

    def test_calls_the_step_hook_after_reset_and_after_every_step(self):
        env = BallCatchEnv(self.cfg, delay="none")
        times = []
        result = run_episode(
            env,
            StationaryController(),
            seed=0,
            options={"launch": self.centre_launch},
            on_step=lambda env: times.append(env.time),
        )
        self.assertEqual(len(times), result["steps"] + 1)
        self.assertEqual(times[0], 0.0)
        self.assertAlmostEqual(times[-1], 0.905)


class EvaluateTest(unittest.TestCase):
    def test_summary_aggregates_the_individual_episodes(self):
        env = BallCatchEnv(delay="jitter")
        summary = evaluate(env, OracleController(env.cfg), episodes=20, seed=0)
        results = summary["results"]
        self.assertEqual(summary["episodes"], 20)
        self.assertEqual(len(results), 20)
        self.assertAlmostEqual(summary["success_rate"], np.mean([r["caught"] for r in results]))
        self.assertAlmostEqual(summary["mean_return"], np.mean([r["return"] for r in results]))
        self.assertAlmostEqual(summary["mean_obs_age"], np.mean([r["mean_obs_age"] for r in results]))

    def test_same_seed_gives_the_same_summary(self):
        env = BallCatchEnv(delay="jitter")
        first = evaluate(env, StationaryController(), episodes=15, seed=3)
        second = evaluate(env, StationaryController(), episodes=15, seed=3)
        self.assertEqual(first["success_rate"], second["success_rate"])
        self.assertEqual(first["mean_return"], second["mean_return"])

    def test_task_is_feasible_with_perfect_information_but_not_solved_by_standing_still(self):
        env = BallCatchEnv(delay="none")
        oracle = evaluate(env, OracleController(env.cfg), episodes=60, seed=0)
        idle = evaluate(env, StationaryController(), episodes=60, seed=0)
        self.assertGreater(oracle["success_rate"], 0.95)
        self.assertLess(idle["success_rate"], 0.2)


if __name__ == "__main__":
    unittest.main()
