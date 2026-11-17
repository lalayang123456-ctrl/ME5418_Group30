"""Training with the auxiliary crossing-point prediction (`TrainConfig.aux_coef`)."""

import os
import tempfile
import unittest

import torch

from agent import PPOAgent, TrainConfig, load_policy, run_name


def tiny_config(variant="temporal", **overrides):
    settings = dict(variant=variant, total_steps=96, num_envs=2, rollout_steps=24, minibatches=2, epochs=2)
    settings.update(overrides)
    return TrainConfig(**settings)


class PredictionConfigTest(unittest.TestCase):
    def test_off_by_default(self):
        self.assertEqual(TrainConfig().aux_coef, 0.0)

    def test_needs_a_recurrent_variant(self):
        with self.assertRaises(ValueError):
            TrainConfig(variant="single_frame", aux_coef=0.5)

    def test_rejects_a_negative_weight(self):
        with self.assertRaises(ValueError):
            TrainConfig(aux_coef=-0.1)

    def test_run_names_keep_the_original_runs_and_mark_prediction(self):
        self.assertEqual(run_name("temporal", "jitter", 0), "temporal_jitter_seed0")
        self.assertEqual(run_name("recurrent", "jitter", 2, aux_coef=0.5), "recurrent_pred_jitter_seed2")


class PredictionTrainingTest(unittest.TestCase):
    def test_update_trains_the_prediction_head_and_logs_its_loss(self):
        agent = PPOAgent(tiny_config(aux_coef=0.5))
        before = [p.detach().clone() for p in agent.policy.predictor.parameters()]
        stats = agent.update()
        self.assertGreater(stats["aux_loss"], 0.0)
        after = list(agent.policy.predictor.parameters())
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))

    def test_without_prediction_nothing_changes(self):
        agent = PPOAgent(tiny_config())
        self.assertIsNone(agent.policy.predictor)
        self.assertNotIn("aux_loss", agent.update())

    def test_saved_network_with_prediction_loads_and_acts_identically(self):
        agent = PPOAgent(tiny_config(aux_coef=0.5))
        agent.train()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.pt")
            agent.save(path)
            policy, config = load_policy(path)
        self.assertEqual(config.aux_coef, 0.5)
        self.assertIsNotNone(policy.predictor)
        obs = torch.randn(1, policy.obs_dim)
        a = agent.policy.act(obs, agent.policy.initial_state(1), deterministic=True).action
        b = policy.act(obs, policy.initial_state(1), deterministic=True).action
        torch.testing.assert_close(a, b)

    def test_checkpoints_saved_before_this_option_still_load(self):
        agent = PPOAgent(tiny_config())
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.pt")
            agent.save(path)
            checkpoint = torch.load(path, weights_only=True)
            del checkpoint["config"]["aux_coef"]  # as in the models of the Learning Agent milestone
            torch.save(checkpoint, path)
            policy, config = load_policy(path)
        self.assertEqual(config.aux_coef, 0.0)
        self.assertIsNone(policy.predictor)


if __name__ == "__main__":
    unittest.main()
