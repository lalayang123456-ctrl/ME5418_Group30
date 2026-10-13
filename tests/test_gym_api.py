import unittest

import gym
import numpy as np

import ball_catch  # noqa: F401  (importing the package registers the environment)
from ball_catch.env import BallCatchEnv

GYM_VERSION = tuple(int(part) for part in gym.__version__.split(".")[:2])
MODERN_GYM = GYM_VERSION >= (0, 26)


class GymApiTest(unittest.TestCase):
    def test_environment_is_registered(self):
        spec = gym.spec("BallCatch-v0")
        self.assertIn("BallCatchEnv", str(spec.entry_point))

    def test_reset_and_step_follow_the_gym_026_signatures(self):
        env = BallCatchEnv()
        self.assertEqual(len(env.reset(seed=0)), 2)
        self.assertEqual(len(env.step(env.action_space.sample())), 5)

    def test_sampled_actions_are_accepted(self):
        env = BallCatchEnv()
        env.reset(seed=0)
        env.action_space.seed(0)
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            self.assertTrue(env.observation_space.contains(obs))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)

    @unittest.skipUnless(MODERN_GYM, "gym.make wrappers of gym < 0.26 expect the old step API")
    def test_gym_make_builds_a_working_environment(self):
        env = gym.make("BallCatch-v0", delay="fixed")
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (11,))
        self.assertEqual(len(env.step(np.zeros(2, dtype=np.float32))), 5)

    @unittest.skipUnless(MODERN_GYM, "the env checker of gym < 0.26 checks the old API")
    def test_environment_passes_the_gym_env_checker(self):
        from gym.utils.env_checker import check_env

        check_env(BallCatchEnv(), skip_render_check=True)


if __name__ == "__main__":
    unittest.main()
