import math
import unittest

import numpy as np

from ball_catch.config import EnvConfig
from ball_catch.delay import SequenceDelay
from ball_catch.dynamics import BallLaunch
from ball_catch.env import (
    OBS_AGE,
    OBS_AREA,
    OBS_INTERVAL,
    OBS_NET_POS,
    OBS_NET_VEL,
    OBS_NEW_FRAME,
    OBS_U,
    OBS_V,
    OBS_VALID,
    BallCatchEnv,
    terminal_reward,
)

DT = 1.0 / 60.0


def quiet_config(**overrides):
    """No sensor noise and no random camera phase, so that exact values can be asserted."""
    settings = dict(pixel_noise_std=0.0, radius_noise_std=0.0, randomize_camera_phase=False)
    settings.update(overrides)
    return EnvConfig(**settings)


def launch_to(cfg, target, flight_time=0.905, position=(-7.5, 0.0, 1.2)):
    return BallLaunch.from_target(position, target, flight_time, cfg.gravity)


def run_episode(env, action=(0.0, 0.0), **reset_kwargs):
    """Play one episode with a constant action; returns per-step records."""
    obs, info = env.reset(**reset_kwargs)
    observations, rewards, infos = [obs], [], [info]
    while True:
        obs, reward, terminated, truncated, info = env.step(np.array(action))
        observations.append(obs)
        rewards.append(reward)
        infos.append(info)
        if terminated or truncated:
            return observations, rewards, infos, terminated, truncated


class TerminalRewardTest(unittest.TestCase):
    def setUp(self):
        self.cfg = EnvConfig()
        self.rim = self.cfg.catch_radius

    def test_catch_at_centre_earns_base_reward_plus_full_bonus(self):
        reward, caught = terminal_reward(0.0, self.cfg)
        self.assertTrue(caught)
        self.assertAlmostEqual(reward, self.cfg.catch_reward + self.cfg.center_bonus)

    def test_catch_at_rim_earns_base_reward_only(self):
        reward, caught = terminal_reward(self.rim, self.cfg)
        self.assertTrue(caught)
        self.assertAlmostEqual(reward, self.cfg.catch_reward)

    def test_centre_bonus_decreases_linearly_with_distance(self):
        reward, _ = terminal_reward(0.5 * self.rim, self.cfg)
        self.assertAlmostEqual(reward, self.cfg.catch_reward + 0.5 * self.cfg.center_bonus)

    def test_near_miss_is_penalised_less_than_large_miss(self):
        near, near_caught = terminal_reward(self.rim + 0.05, self.cfg)
        medium, _ = terminal_reward(self.rim + 0.5, self.cfg)
        far, _ = terminal_reward(self.rim + 3.0, self.cfg)
        self.assertFalse(near_caught)
        self.assertAlmostEqual(near, -0.05)
        self.assertAlmostEqual(medium, -0.5)
        self.assertAlmostEqual(far, -self.cfg.miss_penalty)

    def test_every_catch_beats_every_miss_by_at_least_the_catch_reward(self):
        worst_catch, _ = terminal_reward(self.rim, self.cfg)
        best_miss, _ = terminal_reward(self.rim + 1e-6, self.cfg)
        self.assertGreaterEqual(worst_catch - best_miss, self.cfg.catch_reward - 1e-5)

    def test_reward_never_increases_with_distance(self):
        rewards = [terminal_reward(d, self.cfg)[0] for d in np.linspace(0.0, 3.0, 400)]
        self.assertTrue(all(a >= b for a, b in zip(rewards, rewards[1:])))


class SpacesTest(unittest.TestCase):
    def test_timing_inputs_extend_the_observation(self):
        timed = BallCatchEnv(EnvConfig(include_timing=True))
        base = BallCatchEnv(EnvConfig(include_timing=False))
        self.assertEqual(timed.observation_space.shape, (11,))
        self.assertEqual(base.observation_space.shape, (9,))

    def test_action_is_a_normalised_planar_acceleration(self):
        env = BallCatchEnv()
        self.assertEqual(env.action_space.shape, (2,))
        np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0])
        np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0])


class ResetTest(unittest.TestCase):
    def test_reset_returns_observation_inside_the_observation_space(self):
        env = BallCatchEnv()
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertIsInstance(info, dict)

    def test_net_starts_at_rest_in_the_workspace_centre(self):
        env = BallCatchEnv()
        obs, info = env.reset(seed=0)
        np.testing.assert_array_equal(obs[OBS_NET_POS], [0.0, 0.0])
        np.testing.assert_array_equal(obs[OBS_NET_VEL], [0.0, 0.0])
        np.testing.assert_allclose(info["net_position"], env.cfg.workspace_center)

    def test_same_seed_reproduces_the_episode(self):
        first = run_episode(BallCatchEnv(), action=(0.3, -0.2), seed=11)
        second = run_episode(BallCatchEnv(), action=(0.3, -0.2), seed=11)
        np.testing.assert_array_equal(np.array(first[0]), np.array(second[0]))
        self.assertEqual(first[1], second[1])

    def test_different_seeds_give_different_launches(self):
        env = BallCatchEnv()
        _, info_a = env.reset(seed=1)
        _, info_b = env.reset(seed=2)
        self.assertNotAlmostEqual(info_a["flight_time"], info_b["flight_time"])

    def test_launch_can_be_prescribed_through_options(self):
        cfg = quiet_config()
        env = BallCatchEnv(cfg, delay="none")
        launch = launch_to(cfg, target=(0.4, 1.0), flight_time=0.75)
        _, info = env.reset(seed=0, options={"launch": launch})
        self.assertAlmostEqual(info["flight_time"], 0.75)
        np.testing.assert_allclose(info["intercept"], [0.4, 1.0])
        np.testing.assert_allclose(info["ball_position"], launch.position)


class EpisodeTest(unittest.TestCase):
    def setUp(self):
        self.cfg = quiet_config()
        self.env = BallCatchEnv(self.cfg, delay="none")
        self.centre = self.cfg.workspace_center

    def test_episode_terminates_in_the_step_where_the_ball_crosses_the_plane(self):
        launch = launch_to(self.cfg, self.centre, flight_time=0.905)
        _, rewards, infos, terminated, truncated = run_episode(
            self.env, seed=0, options={"launch": launch}
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(len(rewards), math.ceil(0.905 / DT))
        self.assertAlmostEqual(infos[-1]["time"], 0.905)
        self.assertAlmostEqual(infos[-1]["ball_position"][0], 0.0)

    def test_ball_arriving_at_the_net_centre_is_caught_with_full_reward(self):
        launch = launch_to(self.cfg, self.centre)
        _, rewards, infos, _, _ = run_episode(self.env, seed=0, options={"launch": launch})
        self.assertTrue(infos[-1]["caught"])
        self.assertTrue(infos[-1]["is_success"])
        self.assertAlmostEqual(infos[-1]["miss_distance"], 0.0)
        self.assertAlmostEqual(sum(rewards), self.cfg.catch_reward + self.cfg.center_bonus)

    def test_ball_arriving_away_from_a_stationary_net_is_missed(self):
        target = self.centre + np.array([0.6, 0.0])
        launch = launch_to(self.cfg, target)
        _, rewards, infos, _, _ = run_episode(self.env, seed=0, options={"launch": launch})
        outside = 0.6 - self.cfg.catch_radius
        self.assertFalse(infos[-1]["caught"])
        self.assertAlmostEqual(infos[-1]["miss_distance"], 0.6)
        self.assertAlmostEqual(sum(rewards), -outside / self.cfg.miss_distance_ref)

    def test_outcome_is_only_reported_at_the_end(self):
        launch = launch_to(self.cfg, self.centre)
        _, _, infos, _, _ = run_episode(self.env, seed=0, options={"launch": launch})
        self.assertTrue(all("caught" not in info for info in infos[:-1]))

    def test_each_step_charges_the_control_effort(self):
        launch = launch_to(self.cfg, self.centre)
        self.env.reset(seed=0, options={"launch": launch})
        _, full, _, _, _ = self.env.step(np.array([1.0, -1.0]))
        _, half, _, _, _ = self.env.step(np.array([0.5, 0.0]))
        self.assertAlmostEqual(full, -self.cfg.control_cost * 2.0 * DT)
        self.assertAlmostEqual(half, -self.cfg.control_cost * 0.25 * DT)

    def test_action_scales_to_the_acceleration_limit(self):
        self.env.reset(seed=0, options={"launch": launch_to(self.cfg, self.centre)})
        _, _, _, _, info = self.env.step(np.array([1.0, -0.5]))
        np.testing.assert_allclose(
            info["net_velocity"], [self.cfg.net_a_max * DT, -0.5 * self.cfg.net_a_max * DT]
        )

    def test_out_of_range_actions_are_clipped(self):
        launch = launch_to(self.cfg, self.centre)
        self.env.reset(seed=0, options={"launch": launch})
        obs_big, reward_big, _, _, _ = self.env.step(np.array([5.0, -5.0]))
        self.env.reset(seed=0, options={"launch": launch})
        obs_unit, reward_unit, _, _, _ = self.env.step(np.array([1.0, -1.0]))
        np.testing.assert_array_equal(obs_big, obs_unit)
        self.assertEqual(reward_big, reward_unit)

    def test_net_is_integrated_up_to_the_exact_crossing_time(self):
        # The flight time falls between two control ticks; the net must stop moving at 0.905 s.
        launch = launch_to(self.cfg, self.centre, flight_time=0.905)
        _, _, infos, _, _ = run_episode(
            self.env, action=(0.1, 0.0), seed=0, options={"launch": launch}
        )
        accel = 0.1 * self.cfg.net_a_max
        self.assertAlmostEqual(infos[-1]["net_position"][0], self.centre[0] + 0.5 * accel * 0.905**2)
        self.assertAlmostEqual(infos[-1]["miss_distance"], 0.5 * accel * 0.905**2)

    def test_observations_stay_inside_the_observation_space(self):
        env = BallCatchEnv(delay="burst")
        for seed in range(5):
            observations, _, _, _, _ = run_episode(env, action=(1.0, 1.0), seed=seed)
            for obs in observations:
                self.assertTrue(env.observation_space.contains(obs))

    def test_ball_that_never_arrives_times_out_with_a_penalty(self):
        slow = launch_to(self.cfg, self.centre, flight_time=self.cfg.episode_time_limit + 1.0)
        _, rewards, infos, terminated, truncated = run_episode(
            self.env, seed=0, options={"launch": slow}
        )
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(len(rewards), self.cfg.max_episode_steps)
        self.assertAlmostEqual(rewards[-1], -self.cfg.timeout_penalty)
        self.assertFalse(infos[-1]["caught"])


class DelayedObservationTest(unittest.TestCase):
    def setUp(self):
        self.cfg = quiet_config()
        self.launch = launch_to(self.cfg, (0.5, 1.4), flight_time=0.905)

    def expected_features(self, capture_time):
        env = BallCatchEnv(self.cfg)
        return env.camera.observe(self.launch.position_at(capture_time))[0]

    def test_without_delay_the_first_frame_is_visible_immediately(self):
        env = BallCatchEnv(self.cfg, delay="none")
        obs, info = env.reset(seed=0, options={"launch": self.launch})
        self.assertEqual(obs[OBS_NEW_FRAME], 1.0)
        self.assertEqual(obs[OBS_VALID], 1.0)
        np.testing.assert_allclose(
            obs[[OBS_U, OBS_V, OBS_AREA]], self.expected_features(0.0), rtol=1e-6
        )
        self.assertEqual(info["obs_age"], 0.0)

    def test_nothing_is_seen_until_the_first_frame_arrives(self):
        env = BallCatchEnv(self.cfg, delay=0.1)
        observations, _, _, _, _ = run_episode(env, seed=0, options={"launch": self.launch})
        for obs in observations[:6]:  # t = 0 ... 5/60 s, before the 0.1 s delay has elapsed
            self.assertEqual(obs[OBS_VALID], 0.0)
            self.assertEqual(obs[OBS_NEW_FRAME], 0.0)
            np.testing.assert_array_equal(obs[[OBS_U, OBS_V, OBS_AREA]], [0.0, 0.0, 0.0])
        self.assertEqual(observations[6][OBS_NEW_FRAME], 1.0)
        self.assertEqual(observations[6][OBS_VALID], 1.0)

    def test_delayed_frame_shows_the_ball_where_it_was_at_capture_time(self):
        env = BallCatchEnv(self.cfg, delay=0.1)
        observations, _, _, _, _ = run_episode(env, seed=0, options={"launch": self.launch})
        np.testing.assert_allclose(
            observations[6][[OBS_U, OBS_V, OBS_AREA]], self.expected_features(0.0), rtol=1e-6
        )

    def test_observation_age_grows_until_the_next_frame_arrives(self):
        env = BallCatchEnv(self.cfg, delay=0.1)
        observations, _, infos, _, _ = run_episode(env, seed=0, options={"launch": self.launch})
        # 30 Hz camera, 60 Hz control: frames arrive on every second control step.
        self.assertAlmostEqual(infos[6]["obs_age"], 0.1)
        self.assertEqual(observations[7][OBS_NEW_FRAME], 0.0)
        self.assertAlmostEqual(infos[7]["obs_age"], 0.1 + DT)
        self.assertEqual(observations[8][OBS_NEW_FRAME], 1.0)
        self.assertAlmostEqual(infos[8]["obs_age"], 0.1)
        self.assertAlmostEqual(observations[7][OBS_AGE], (0.1 + DT) / self.cfg.delay_max, places=6)

    def test_visual_features_are_held_between_frames(self):
        env = BallCatchEnv(self.cfg, delay=0.1)
        observations, _, _, _, _ = run_episode(env, seed=0, options={"launch": self.launch})
        visual = [OBS_U, OBS_V, OBS_AREA, OBS_VALID]
        np.testing.assert_array_equal(observations[7][visual], observations[6][visual])

    def test_capture_interval_reports_the_time_between_accepted_captures(self):
        env = BallCatchEnv(self.cfg, delay=0.1)
        observations, _, infos, _, _ = run_episode(env, seed=0, options={"launch": self.launch})
        self.assertEqual(infos[6]["capture_interval"], 0.0)  # only one frame so far
        self.assertAlmostEqual(infos[8]["capture_interval"], 1.0 / 30.0)
        self.assertAlmostEqual(
            observations[8][OBS_INTERVAL], (1.0 / 30.0) / self.cfg.delay_max, places=6
        )

    def test_frames_overtaken_by_newer_ones_are_discarded(self):
        # Even frames take 0.2 s, odd frames arrive instantly and overtake them.
        env = BallCatchEnv(self.cfg, delay=SequenceDelay([0.2, 0.0]))
        _, _, infos, _, _ = run_episode(env, seed=0, options={"launch": self.launch})
        self.assertGreater(infos[-1]["frames_discarded"], 0)
        self.assertGreater(infos[-1]["frames_accepted"], 0)
        # Odd frames are 1/15 s apart, which is what the controller must be told.
        self.assertAlmostEqual(infos[-1]["capture_interval"], 2.0 / 30.0)
        self.assertTrue(all(info["obs_age"] >= 0.0 for info in infos))

    def test_ball_outside_the_image_gives_a_new_but_invalid_frame(self):
        env = BallCatchEnv(self.cfg, delay="none")
        off_screen = launch_to(self.cfg, (0.0, 1.0), position=(-6.0, 12.0, 1.2))
        obs, _ = env.reset(seed=0, options={"launch": off_screen})
        self.assertEqual(obs[OBS_NEW_FRAME], 1.0)
        self.assertEqual(obs[OBS_VALID], 0.0)
        np.testing.assert_array_equal(obs[[OBS_U, OBS_V, OBS_AREA]], [0.0, 0.0, 0.0])

    def test_base_observation_is_the_timed_observation_without_timing_inputs(self):
        timed_env = BallCatchEnv(EnvConfig(include_timing=True), delay="jitter")
        base_env = BallCatchEnv(EnvConfig(include_timing=False), delay="jitter")
        timed, _, _, _, _ = run_episode(timed_env, action=(0.2, 0.1), seed=4)
        base, _, _, _, _ = run_episode(base_env, action=(0.2, 0.1), seed=4)
        np.testing.assert_array_equal(np.array(timed)[:, :9], np.array(base))

    def test_privileged_information_is_only_in_info(self):
        env = BallCatchEnv(self.cfg, delay="none")
        _, info = env.reset(seed=0, options={"launch": self.launch})
        for key in ("ball_position", "ball_velocity", "flight_time", "intercept"):
            self.assertIn(key, info)


if __name__ == "__main__":
    unittest.main()
