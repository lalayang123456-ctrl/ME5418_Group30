import math
import os
import tempfile
import unittest

import numpy as np
import torch

from agent import NetworkController, PPOAgent, TrainConfig, load_policy
from ball_catch.env import BallCatchEnv
from ball_catch.evaluation import evaluate
from networks.policies import VARIANTS, env_config_for, make_policy


def tiny_config(variant="temporal", **overrides):
    """A run small enough for a unit test: 2 environments, 2 updates of 2 x 24 steps."""
    settings = dict(variant=variant, total_steps=96, num_envs=2, rollout_steps=24, minibatches=2, epochs=2)
    settings.update(overrides)
    return TrainConfig(**settings)


class TrainConfigTest(unittest.TestCase):
    def test_rejects_an_unknown_variant(self):
        with self.assertRaises(ValueError):
            TrainConfig(variant="lstm")

    def test_minibatches_must_split_the_environments_evenly(self):
        with self.assertRaises(ValueError):
            TrainConfig(num_envs=6, minibatches=4)


class UpdateTest(unittest.TestCase):
    def test_every_variant_updates_its_network(self):
        for variant in VARIANTS:
            agent = PPOAgent(tiny_config(variant))
            before = [p.detach().clone() for p in agent.policy.parameters()]
            stats = agent.update()
            self.assertEqual(stats["steps"], 48)
            self.assertEqual(stats["update"], 1)
            for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm"):
                self.assertTrue(math.isfinite(stats[key]), (variant, key, stats[key]))
            changed = [not torch.equal(a, b) for a, b in zip(before, agent.policy.parameters())]
            self.assertTrue(all(changed), variant)

    def test_gradients_are_clipped(self):
        agent = PPOAgent(tiny_config(max_grad_norm=1e-3))
        agent.update()
        total = math.sqrt(sum(float(p.grad.norm() ** 2) for p in agent.policy.parameters()))
        self.assertLessEqual(total, 1e-3 * 1.001)

    def test_same_seed_gives_the_same_training(self):
        a = PPOAgent(tiny_config()).train()
        b = PPOAgent(tiny_config()).train()
        for key in ("policy_loss", "value_loss", "approx_kl", "grad_norm"):
            self.assertEqual([s[key] for s in a], [s[key] for s in b])

    def test_train_runs_until_the_step_budget_and_anneals_the_learning_rate(self):
        agent = PPOAgent(tiny_config(total_steps=100))  # not a multiple of 48: rounded up
        calls = []
        history = agent.train(lambda ag, stats: calls.append(stats["update"]))
        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(agent.steps, 144)
        self.assertEqual([round(s["lr"] / 3e-4, 6) for s in history], [1.0, round(2 / 3, 6), round(1 / 3, 6)])

    def test_constant_learning_rate_without_annealing(self):
        history = PPOAgent(tiny_config(anneal_lr=False)).train()
        self.assertTrue(all(s["lr"] == 3e-4 for s in history))

    def test_input_statistics_change_only_when_normalisation_is_on(self):
        plain = PPOAgent(tiny_config(normalize_obs=False))
        plain.update()
        torch.testing.assert_close(plain.policy.obs_norm.mean, torch.zeros(11))
        normalised = PPOAgent(tiny_config())  # on by default
        normalised.update()
        self.assertAlmostEqual(float(normalised.policy.obs_norm.count), 48.0, places=3)
        self.assertGreater(float(normalised.policy.obs_norm.mean.abs().sum()), 0.0)


class SaveLoadTest(unittest.TestCase):
    def test_a_saved_network_acts_exactly_like_the_trained_one(self):
        agent = PPOAgent(tiny_config("recurrent"))
        agent.train()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sub", "model.pt")
            agent.save(path)
            policy, config = load_policy(path)
        self.assertEqual(config, agent.config)
        env = BallCatchEnv(env_config_for("recurrent"))
        original = evaluate(env, agent.controller(), episodes=3, seed=5)
        loaded = evaluate(env, NetworkController(policy), episodes=3, seed=5)
        self.assertEqual(original["mean_return"], loaded["mean_return"])


class NetworkControllerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.policy = make_policy("temporal")
        with torch.no_grad():
            self.policy.actor[-1].bias.fill_(5.0)  # mean action far outside [-1, 1]
        self.env = BallCatchEnv(env_config_for("temporal"))

    def test_actions_are_clipped_to_the_action_space(self):
        obs, info = self.env.reset(seed=0)
        action = NetworkController(self.policy).act(obs, info)
        self.assertEqual(action.shape, (2,))
        np.testing.assert_array_equal(action, [1.0, 1.0])

    def test_reset_clears_the_memory(self):
        controller = NetworkController(self.policy)
        obs, info = self.env.reset(seed=0)
        for _ in range(5):
            controller.act(obs, info)
        controller.reset()
        torch.testing.assert_close(controller.state, self.policy.initial_state(1))

    def test_deterministic_evaluation_is_repeatable(self):
        policy = make_policy("temporal")
        a = evaluate(self.env, NetworkController(policy), episodes=3, seed=0)
        b = evaluate(self.env, NetworkController(policy), episodes=3, seed=0)
        self.assertEqual(a["mean_return"], b["mean_return"])


class EvaluateTest(unittest.TestCase):
    def test_agent_evaluation_reports_a_success_rate(self):
        agent = PPOAgent(tiny_config("single_frame"))
        result = agent.evaluate(episodes=4)
        self.assertEqual(result["episodes"], 4)
        self.assertTrue(0.0 <= result["success_rate"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
