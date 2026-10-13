import unittest

import numpy as np

from ball_catch.config import EnvConfig
from ball_catch.dynamics import BallLaunch, sample_launch, step_net


class BallLaunchTest(unittest.TestCase):
    def test_position_follows_projectile_motion(self):
        launch = BallLaunch(
            position=np.array([-8.0, 0.5, 1.0]),
            velocity=np.array([10.0, -1.0, 4.0]),
            gravity=9.81,
        )
        expected = [-3.0, 0.0, 1.0 + 4.0 * 0.5 - 0.5 * 9.81 * 0.25]
        np.testing.assert_allclose(launch.position_at(0.5), expected)

    def test_gravity_only_changes_vertical_velocity(self):
        launch = BallLaunch(
            position=np.array([-8.0, 0.5, 1.0]),
            velocity=np.array([10.0, -1.0, 4.0]),
            gravity=9.81,
        )
        np.testing.assert_allclose(launch.velocity_at(0.5), [10.0, -1.0, 4.0 - 9.81 * 0.5])

    def test_flight_time_is_when_ball_reaches_net_plane(self):
        launch = BallLaunch(
            position=np.array([-8.0, 0.0, 1.0]),
            velocity=np.array([10.0, 0.0, 0.0]),
            gravity=9.81,
        )
        self.assertAlmostEqual(launch.flight_time, 0.8)
        self.assertAlmostEqual(launch.position_at(launch.flight_time)[0], 0.0)

    def test_from_target_hits_requested_point_at_requested_time(self):
        launch = BallLaunch.from_target(
            position=[-7.0, 1.0, 1.5], target=[-0.4, 0.9], flight_time=0.9, gravity=9.81
        )
        np.testing.assert_allclose(launch.position_at(0.9), [0.0, -0.4, 0.9], atol=1e-12)
        self.assertAlmostEqual(launch.flight_time, 0.9)
        np.testing.assert_allclose(launch.intercept, [-0.4, 0.9], atol=1e-12)


class SampleLaunchTest(unittest.TestCase):
    def test_sampled_launches_respect_configured_ranges(self):
        cfg = EnvConfig()
        rng = np.random.default_rng(0)
        for _ in range(200):
            launch = sample_launch(rng, cfg)
            x0, y0, z0 = launch.position
            self.assertTrue(cfg.flight_time_range[0] <= launch.flight_time <= cfg.flight_time_range[1])
            self.assertTrue(cfg.launch_distance_range[0] <= -x0 <= cfg.launch_distance_range[1])
            self.assertTrue(cfg.launch_y_range[0] <= y0 <= cfg.launch_y_range[1])
            self.assertTrue(cfg.launch_z_range[0] <= z0 <= cfg.launch_z_range[1])
            y_hit, z_hit = launch.intercept
            self.assertTrue(cfg.workspace_y[0] - 1e-9 <= y_hit <= cfg.workspace_y[1] + 1e-9)
            self.assertTrue(cfg.workspace_z[0] - 1e-9 <= z_hit <= cfg.workspace_z[1] + 1e-9)

    def test_ball_never_drops_below_launch_and_intercept_heights(self):
        cfg = EnvConfig()
        rng = np.random.default_rng(1)
        for _ in range(50):
            launch = sample_launch(rng, cfg)
            floor = min(launch.position[2], launch.intercept[1])
            for t in np.linspace(0.0, launch.flight_time, 25):
                self.assertGreaterEqual(launch.position_at(t)[2], floor - 1e-9)

    def test_same_seed_gives_same_launch(self):
        cfg = EnvConfig()
        a = sample_launch(np.random.default_rng(7), cfg)
        b = sample_launch(np.random.default_rng(7), cfg)
        np.testing.assert_array_equal(a.position, b.position)
        np.testing.assert_array_equal(a.velocity, b.velocity)


class StepNetTest(unittest.TestCase):
    def setUp(self):
        # A huge workspace so that only the limit under test is active.
        self.open_cfg = EnvConfig(workspace_y=(-50.0, 50.0), workspace_z=(-50.0, 50.0))

    def test_accelerates_from_rest(self):
        cfg = self.open_cfg
        pos, vel = step_net(np.zeros(2), np.zeros(2), np.array([4.0, -2.0]), 0.1, cfg)
        np.testing.assert_allclose(pos, [0.5 * 4.0 * 0.01, -0.5 * 2.0 * 0.01])
        np.testing.assert_allclose(vel, [0.4, -0.2])

    def test_velocity_saturates_inside_a_single_step(self):
        cfg = self.open_cfg
        a, v = cfg.net_a_max, cfg.net_v_max
        pos, vel = step_net(np.zeros(2), np.zeros(2), np.array([a, 0.0]), 1.0, cfg)
        t_sat = v / a
        self.assertAlmostEqual(vel[0], v)
        self.assertAlmostEqual(pos[0], 0.5 * a * t_sat**2 + v * (1.0 - t_sat))

    def test_many_small_steps_match_one_big_step(self):
        cfg = self.open_cfg
        command = np.array([cfg.net_a_max, -0.5 * cfg.net_a_max])
        pos, vel = np.zeros(2), np.zeros(2)
        for _ in range(100):
            pos, vel = step_net(pos, vel, command, 0.01, cfg)
        big_pos, big_vel = step_net(np.zeros(2), np.zeros(2), command, 1.0, cfg)
        np.testing.assert_allclose(pos, big_pos, atol=1e-9)
        np.testing.assert_allclose(vel, big_vel, atol=1e-9)

    def test_command_is_clipped_to_acceleration_limit(self):
        cfg = self.open_cfg
        limited = step_net(np.zeros(2), np.zeros(2), np.array([cfg.net_a_max, 0.0]), 0.05, cfg)
        excessive = step_net(np.zeros(2), np.zeros(2), np.array([1000.0, 0.0]), 0.05, cfg)
        np.testing.assert_allclose(excessive[0], limited[0])
        np.testing.assert_allclose(excessive[1], limited[1])

    def test_stops_at_workspace_boundary(self):
        cfg = EnvConfig()
        start = np.array([cfg.workspace_y[1] - 0.01, cfg.workspace_z[0] + 0.01])
        moving_out = np.array([cfg.net_v_max, -cfg.net_v_max])
        pos, vel = step_net(start, moving_out, np.zeros(2), 0.1, cfg)
        np.testing.assert_allclose(pos, [cfg.workspace_y[1], cfg.workspace_z[0]])
        np.testing.assert_allclose(vel, [0.0, 0.0])

    def test_inputs_are_not_modified(self):
        cfg = self.open_cfg
        pos, vel = np.zeros(2), np.zeros(2)
        step_net(pos, vel, np.array([1.0, 1.0]), 0.1, cfg)
        np.testing.assert_array_equal(pos, [0.0, 0.0])
        np.testing.assert_array_equal(vel, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
