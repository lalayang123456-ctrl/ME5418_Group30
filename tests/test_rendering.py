import unittest

import numpy as np

from ball_catch.env import BallCatchEnv


class RenderTest(unittest.TestCase):
    def test_render_returns_nothing_without_a_render_mode(self):
        env = BallCatchEnv()
        env.reset(seed=0)
        self.assertIsNone(env.render())

    def test_rgb_array_render_returns_an_image(self):
        env = BallCatchEnv(render_mode="rgb_array")
        env.reset(seed=0)
        frame = env.render()
        self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)
        self.assertGreater(frame.std(), 0.0)  # not a blank canvas
        env.close()

    def test_picture_changes_as_the_episode_advances(self):
        env = BallCatchEnv(render_mode="rgb_array")
        env.reset(seed=0)
        first = env.render()
        for _ in range(10):
            env.step(np.array([1.0, 0.5]))
        later = env.render()
        self.assertEqual(first.shape, later.shape)
        self.assertFalse(np.array_equal(first, later))
        env.close()

    def test_whole_episode_can_be_rendered_including_the_outcome(self):
        env = BallCatchEnv(delay="burst", render_mode="rgb_array")
        env.reset(seed=1)
        frames = [env.render()]
        while True:
            _, _, terminated, truncated, _ = env.step(np.zeros(2))
            frames.append(env.render())
            if terminated or truncated:
                break
        self.assertEqual(len({frame.shape for frame in frames}), 1)
        self.assertFalse(np.array_equal(frames[-2], frames[-1]))
        env.close()

    def test_renderer_survives_a_new_episode(self):
        env = BallCatchEnv(render_mode="rgb_array")
        env.reset(seed=0)
        first_episode = env.render()
        env.reset(seed=5)
        second_episode = env.render()
        self.assertFalse(np.array_equal(first_episode, second_episode))
        env.close()

    def test_close_releases_the_renderer_and_render_recreates_it(self):
        env = BallCatchEnv(render_mode="rgb_array")
        env.reset(seed=0)
        env.render()
        env.close()
        self.assertIsNone(env._renderer)
        self.assertIsNotNone(env.render())
        env.close()


if __name__ == "__main__":
    unittest.main()
