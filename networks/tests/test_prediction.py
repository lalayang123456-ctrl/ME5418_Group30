"""Auxiliary prediction of the crossing point (proposal, section 4: "Temporal-aware PPO + prediction").

The simulator's crossing point is used only as a training target: it must never reach the policy's
inputs, and every step must be labelled with the crossing point of its own throw.
"""

import unittest

import numpy as np
import torch

from ball_catch.env import BallCatchEnv
from networks.policies import env_config_for, make_policy
from networks.ppo import RolloutCollector, compute_gae, ppo_loss


class RecordingEnv(BallCatchEnv):
    """Remembers the crossing point of every throw it starts."""

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.crossings = getattr(self, "crossings", []) + [np.array(info["intercept"], dtype=float)]
        return obs, info


def normalised(cfg, point):
    return (np.asarray(point) - cfg.workspace_center) / cfg.workspace_half_extent


class PredictionHeadTest(unittest.TestCase):
    def test_head_exists_only_when_requested(self):
        for variant in ("recurrent", "temporal"):
            plain = make_policy(variant)
            with_head = make_policy(variant, predict_crossing=True)
            self.assertFalse(any(k.startswith("predictor") for k in plain.state_dict()))
            self.assertTrue(any(k.startswith("predictor") for k in with_head.state_dict()))
            # Same inputs as the plain network: the target never becomes an input.
            self.assertEqual(with_head.obs_dim, plain.obs_dim)

    def test_single_frame_network_has_no_prediction_head(self):
        with self.assertRaises(ValueError):
            make_policy("single_frame", predict_crossing=True)

    def test_prediction_shape_and_it_trains_the_shared_memory(self):
        torch.manual_seed(0)
        policy = make_policy("temporal", predict_crossing=True)
        obs = torch.randn(7, 3, policy.obs_dim)
        log_prob, entropy, value, prediction = policy.evaluate(
            obs, torch.zeros(7, 3, 2), with_prediction=True)
        self.assertEqual(prediction.shape, (7, 3, 2))
        self.assertEqual(log_prob.shape, (7, 3))
        prediction.pow(2).sum().backward()
        for part in ("encoder", "gru", "predictor"):
            grads = [p.grad for name, p in policy.named_parameters() if name.startswith(part)]
            self.assertTrue(all(g is not None and g.abs().sum() > 0 for g in grads), part)

    def test_plain_evaluate_is_unchanged(self):
        policy = make_policy("temporal", predict_crossing=True)
        out = policy.evaluate(torch.randn(5, 2, policy.obs_dim), torch.zeros(5, 2, 2))
        self.assertEqual(len(out), 3)


class TargetTest(unittest.TestCase):
    def collect(self, steps=160, num_envs=2):
        torch.manual_seed(0)
        cfg = env_config_for("temporal")
        envs = [RecordingEnv(cfg, delay="jitter") for _ in range(num_envs)]
        policy = make_policy("temporal", predict_crossing=True)
        rollout = RolloutCollector(envs, policy, seed=3).collect(steps)
        return cfg, envs, rollout

    def test_every_step_is_labelled_with_the_crossing_point_of_its_own_throw(self):
        cfg, envs, rollout = self.collect()
        self.assertEqual(rollout.targets.shape, (160, 2, 2))
        for b, env in enumerate(envs):
            # Throw k of environment b covers the steps from its k-th episode start up to its end.
            starts = rollout.episode_start[:, b].nonzero().flatten().tolist()
            self.assertGreaterEqual(len(starts), 2, "the rollout should span several throws")
            bounds = starts + [rollout.targets.shape[0]]
            for k in range(len(starts)):
                expected = torch.tensor(normalised(cfg, env.crossings[k]), dtype=torch.float32)
                segment = rollout.targets[bounds[k]:bounds[k + 1], b]
                torch.testing.assert_close(segment, expected.expand_as(segment))

    def test_the_last_step_of_a_throw_keeps_that_throws_label(self):
        cfg, envs, rollout = self.collect()
        ends = rollout.dones[:, 0].nonzero().flatten()
        self.assertGreater(len(ends), 0)
        t = int(ends[0])
        torch.testing.assert_close(
            rollout.targets[t, 0], torch.tensor(normalised(cfg, envs[0].crossings[0]), dtype=torch.float32))

    def test_select_keeps_the_targets_of_the_chosen_environments(self):
        _, _, rollout = self.collect(steps=40, num_envs=3)
        part = rollout.select(torch.tensor([2, 0]))
        torch.testing.assert_close(part.targets, rollout.targets[:, [2, 0]])


class PredictionLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        cfg = env_config_for("temporal")
        self.policy = make_policy("temporal", predict_crossing=True)
        self.rollout = RolloutCollector([BallCatchEnv(cfg, delay="jitter")], self.policy, seed=0).collect(80)
        r = self.rollout
        self.adv, self.ret = compute_gae(r.rewards, r.values, r.dones, r.last_value)

    def test_the_prediction_loss_is_added_with_its_weight(self):
        base, base_stats = ppo_loss(self.policy, self.rollout, self.adv, self.ret)
        loss, stats = ppo_loss(self.policy, self.rollout, self.adv, self.ret, aux_coef=0.5)
        self.assertNotIn("aux_loss", base_stats)
        self.assertGreater(stats["aux_loss"], 0.0)
        self.assertAlmostEqual(float(loss), float(base) + 0.5 * stats["aux_loss"], places=5)

    def test_the_prediction_loss_is_the_squared_error_to_the_crossing_point(self):
        r = self.rollout
        _, _, _, prediction = self.policy.evaluate(
            r.obs, r.actions, r.initial_state, r.episode_start, with_prediction=True)
        expected = ((prediction - r.targets) ** 2).sum(-1).mean()
        _, stats = ppo_loss(self.policy, r, self.adv, self.ret, aux_coef=1.0)
        self.assertAlmostEqual(stats["aux_loss"], float(expected), places=5)

    def test_only_the_weighted_loss_trains_the_prediction_head(self):
        loss, _ = ppo_loss(self.policy, self.rollout, self.adv, self.ret)
        loss.backward()
        self.assertTrue(all(p.grad is None for n, p in self.policy.named_parameters() if n.startswith("predictor")))
        self.policy.zero_grad()
        loss, _ = ppo_loss(self.policy, self.rollout, self.adv, self.ret, aux_coef=0.5)
        loss.backward()
        self.assertTrue(all(p.grad is not None for n, p in self.policy.named_parameters() if n.startswith("predictor")))


if __name__ == "__main__":
    unittest.main()
