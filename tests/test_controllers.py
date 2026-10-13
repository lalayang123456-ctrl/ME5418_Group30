import unittest

import numpy as np

from ball_catch.config import EnvConfig
from ball_catch.controllers import (
    ChaseController,
    OracleController,
    RandomController,
    StationaryController,
    make_controller,
    move_to,
)
from ball_catch.dynamics import BallLaunch, step_net
from ball_catch.env import OBS_NET_POS, OBS_NET_VEL, OBS_U, OBS_V, OBS_VALID, BallCatchEnv


# The scenarios below are worked out for these net limits; pinning them keeps the tests valid
# when the defaults of EnvConfig are recalibrated.
LIMITS = dict(net_a_max=10.0, net_v_max=3.0)


def quiet_config(**overrides):
    settings = dict(pixel_noise_std=0.0, radius_noise_std=0.0, randomize_camera_phase=False, **LIMITS)
    settings.update(overrides)
    return EnvConfig(**settings)


def play(env, controller, **reset_kwargs):
    obs, info = env.reset(**reset_kwargs)
    controller.reset()
    while True:
        obs, _, terminated, truncated, info = env.step(controller.act(obs, info))
        if terminated or truncated:
            return info


class MoveToTest(unittest.TestCase):
    def setUp(self):
        self.cfg = EnvConfig(**LIMITS)

    def test_no_command_when_resting_on_the_target(self):
        target = np.array([0.2, 1.0])
        np.testing.assert_array_equal(move_to(target, target, np.zeros(2), self.cfg), [0.0, 0.0])

    def test_full_acceleration_towards_a_distant_target(self):
        action = move_to(np.array([0.9, 0.4]), np.array([0.0, 1.15]), np.zeros(2), self.cfg)
        np.testing.assert_array_equal(action, [1.0, -1.0])

    def test_brakes_when_approaching_too_fast(self):
        action = move_to(np.array([0.05, 1.15]), np.array([0.0, 1.15]), np.array([3.0, 0.0]), self.cfg)
        self.assertEqual(action[0], -1.0)

    def test_reaches_the_target_and_stops_there(self):
        cfg = self.cfg
        target = np.array([0.9, 0.5])
        pos, vel = cfg.workspace_center, np.zeros(2)
        for _ in range(90):  # 1.5 s
            accel = move_to(target, pos, vel, cfg) * cfg.net_a_max
            pos, vel = step_net(pos, vel, accel, cfg.control_dt, cfg)
        np.testing.assert_allclose(pos, target, atol=5e-3)
        np.testing.assert_allclose(vel, [0.0, 0.0], atol=5e-2)

    def test_travel_time_is_close_to_time_optimal(self):
        # 0.9 m from rest: bang-bang with a = 10 m/s^2 and v = 3 m/s takes 0.6 s.
        cfg = self.cfg
        target = cfg.workspace_center + np.array([0.9, 0.0])
        pos, vel = cfg.workspace_center, np.zeros(2)
        steps = 0
        while np.linalg.norm(target - pos) > 0.02:
            accel = move_to(target, pos, vel, cfg) * cfg.net_a_max
            pos, vel = step_net(pos, vel, accel, cfg.control_dt, cfg)
            steps += 1
            self.assertLess(steps, 200)
        self.assertLess(steps * cfg.control_dt, 0.75)


class SimpleControllerTest(unittest.TestCase):
    def test_stationary_controller_never_moves(self):
        np.testing.assert_array_equal(StationaryController().act(np.zeros(11), {}), [0.0, 0.0])

    def test_random_controller_is_bounded_and_reproducible(self):
        first, second = RandomController(seed=3), RandomController(seed=3)
        actions = np.array([first.act(None, None) for _ in range(50)])
        again = np.array([second.act(None, None) for _ in range(50)])
        self.assertEqual(actions.shape, (50, 2))
        np.testing.assert_array_equal(actions, again)
        self.assertTrue(np.all(np.abs(actions) <= 1.0))
        self.assertGreater(actions.std(), 0.3)

    def test_make_controller_builds_every_named_controller(self):
        cfg = EnvConfig()
        self.assertIsInstance(make_controller("random", cfg), RandomController)
        self.assertIsInstance(make_controller("stationary", cfg), StationaryController)
        self.assertIsInstance(make_controller("oracle", cfg), OracleController)
        self.assertIsInstance(make_controller("chase", cfg), ChaseController)
        with self.assertRaises(ValueError):
            make_controller("psychic", cfg)


class OracleControllerTest(unittest.TestCase):
    def setUp(self):
        self.cfg = quiet_config()
        self.corner = np.array([self.cfg.workspace_y[1], self.cfg.workspace_z[1]])

    def test_catches_a_ball_aimed_at_the_workspace_corner_given_enough_time(self):
        env = BallCatchEnv(self.cfg, delay="none")
        launch = BallLaunch.from_target((-8.0, 0.0, 1.2), self.corner, 1.2, self.cfg.gravity)
        info = play(env, OracleController(self.cfg), seed=0, options={"launch": launch})
        self.assertTrue(info["caught"])
        self.assertLess(info["miss_distance"], 0.05)

    def test_does_not_act_before_its_reaction_time(self):
        controller = OracleController(self.cfg, reaction_time=0.3)
        info = {
            "time": 0.2,
            "intercept": self.corner,
            "net_position": self.cfg.workspace_center,
            "net_velocity": np.zeros(2),
        }
        np.testing.assert_array_equal(controller.act(None, info), [0.0, 0.0])
        info["time"] = 0.3
        np.testing.assert_array_equal(controller.act(None, info), [1.0, 1.0])

    def test_late_start_turns_a_feasible_catch_into_a_miss(self):
        env = BallCatchEnv(self.cfg, delay="none")
        launch = BallLaunch.from_target((-8.0, 0.0, 1.2), self.corner, 0.7, self.cfg.gravity)
        prompt = play(env, OracleController(self.cfg), seed=0, options={"launch": launch})
        late = play(
            env, OracleController(self.cfg, reaction_time=0.4), seed=0, options={"launch": launch}
        )
        self.assertTrue(prompt["caught"])
        self.assertFalse(late["caught"])


class ChaseControllerTest(unittest.TestCase):
    def setUp(self):
        self.cfg = quiet_config()
        self.controller = ChaseController(self.cfg)

    def observation(self, u=0.0, v=0.0, valid=1.0, net_vel=(0.0, 0.0)):
        obs = np.zeros(11, dtype=np.float32)
        obs[OBS_U], obs[OBS_V], obs[OBS_VALID] = u, v, valid
        obs[OBS_NET_VEL] = net_vel
        return obs

    def test_uses_the_observation_only(self):
        action = self.controller.act(self.observation(u=0.3, v=0.2), None)
        self.assertEqual(np.shape(action), (2,))

    def test_moves_towards_the_side_where_the_ball_is_seen(self):
        right_up = self.controller.act(self.observation(u=0.3, v=0.2), None)
        left_down = self.controller.act(self.observation(u=-0.3, v=-0.2), None)
        self.assertTrue(np.all(right_up > 0.0))
        self.assertTrue(np.all(left_down < 0.0))

    def test_brakes_when_no_ball_has_been_seen(self):
        action = self.controller.act(self.observation(valid=0.0, net_vel=(0.5, -0.5)), None)
        self.assertLess(action[0], 0.0)
        self.assertGreater(action[1], 0.0)

    def launch_slightly_off_centre(self, flight_time):
        target = self.cfg.workspace_center + np.array([0.3, 0.2])
        return BallLaunch.from_target((-8.0, 0.0, 1.2), target, flight_time, self.cfg.gravity)

    def test_catches_a_fast_flat_ball_without_delay(self):
        env = BallCatchEnv(self.cfg, delay="none")
        launch = self.launch_slightly_off_centre(flight_time=0.6)
        info = play(env, self.controller, seed=0, options={"launch": launch})
        self.assertTrue(info["caught"])

    def test_is_fooled_by_a_high_arc_because_it_ignores_gravity(self):
        # A slow throw rises far above its landing point; following the line of sight drags the
        # net upwards and it cannot come back down in time.
        env = BallCatchEnv(self.cfg, delay="none")
        launch = self.launch_slightly_off_centre(flight_time=1.2)
        info = play(env, self.controller, seed=0, options={"launch": launch})
        self.assertFalse(info["caught"])

    def test_delay_makes_it_catch_fewer_balls(self):
        def success_rate(delay):
            env = BallCatchEnv(EnvConfig(**LIMITS), delay=delay)
            controller = ChaseController(env.cfg)
            return np.mean([play(env, controller, seed=seed)["caught"] for seed in range(150)])

        self.assertGreater(success_rate("none"), success_rate(0.2) + 0.05)


if __name__ == "__main__":
    unittest.main()
