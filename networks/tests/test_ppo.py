import os
import subprocess
import sys
import unittest

import torch

from ball_catch.env import BallCatchEnv
from networks.policies import VARIANTS, env_config_for, make_policy
from networks.ppo import collect_rollout, compute_gae, gradient_norms, ppo_loss

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GaeTest(unittest.TestCase):
    def test_matches_a_hand_computed_example(self):
        rewards = torch.tensor([[0.0], [1.0]])
        values = torch.tensor([[0.5], [0.2]])
        dones = torch.zeros(2, 1)
        last_value = torch.tensor([0.4])
        gamma, lam = 0.9, 0.8
        adv, ret = compute_gae(rewards, values, dones, last_value, gamma, lam)
        delta1 = 1.0 + gamma * 0.4 - 0.2
        delta0 = 0.0 + gamma * 0.2 - 0.5
        torch.testing.assert_close(adv, torch.tensor([[delta0 + gamma * lam * delta1], [delta1]]))
        torch.testing.assert_close(ret, adv + values)

    def test_does_not_bootstrap_across_the_end_of_a_throw(self):
        rewards = torch.tensor([[1.0], [0.0]])
        values = torch.tensor([[0.0], [5.0]])
        dones = torch.tensor([[1.0], [0.0]])
        adv, _ = compute_gae(rewards, values, dones, torch.tensor([0.0]))
        self.assertAlmostEqual(float(adv[0, 0]), 1.0)  # neither V(next throw) nor its advantage


class RolloutTest(unittest.TestCase):
    def collect(self, variant, steps=90):
        torch.manual_seed(0)
        policy = make_policy(variant)
        env = BallCatchEnv(env_config_for(variant), delay="jitter")
        return policy, collect_rollout(env, policy, steps, seed=0)

    def test_shapes_and_episode_bookkeeping(self):
        for variant in VARIANTS:
            policy, rollout = self.collect(variant)
            self.assertEqual(rollout.obs.shape, (90, 1, policy.obs_dim))
            self.assertEqual(rollout.actions.shape, (90, 1, 2))
            for tensor in (rollout.log_probs, rollout.values, rollout.rewards, rollout.dones):
                self.assertEqual(tensor.shape, (90, 1))
            self.assertEqual(float(rollout.episode_start[0, 0]), 1.0)
            # Every throw that ends is followed by a fresh start, except at the very end.
            ends = rollout.dones[:-1, 0].nonzero().flatten()
            self.assertTrue(torch.all(rollout.episode_start[ends + 1, 0] == 1.0))
            self.assertEqual(int(rollout.dones.sum()), rollout.episodes)
            self.assertGreaterEqual(rollout.episodes, 1)

    def test_same_seed_gives_the_same_rollout(self):
        _, a = self.collect("temporal")
        _, b = self.collect("temporal")
        torch.testing.assert_close(a.actions, b.actions)
        torch.testing.assert_close(a.rewards, b.rewards)

    def test_stored_log_probs_match_re_evaluation(self):
        for variant in VARIANTS:
            policy, rollout = self.collect(variant)
            log_prob, _, value = policy.evaluate(
                rollout.obs, rollout.actions, rollout.initial_state, rollout.episode_start
            )
            torch.testing.assert_close(log_prob.detach(), rollout.log_probs)
            torch.testing.assert_close(value.detach(), rollout.values)


class LossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.policy = make_policy("temporal")
        env = BallCatchEnv(env_config_for("temporal"), delay="jitter")
        self.rollout = collect_rollout(env, self.policy, 80, seed=0)
        r = self.rollout
        self.adv, self.ret = compute_gae(r.rewards, r.values, r.dones, r.last_value)

    def test_policy_term_is_zero_before_any_update(self):
        _, stats = ppo_loss(self.policy, self.rollout, self.adv, self.ret)
        self.assertAlmostEqual(stats["policy_loss"], 0.0, places=5)
        self.assertEqual(stats["clip_fraction"], 0.0)

    def test_backward_reaches_every_part_and_a_step_lowers_the_loss(self):
        optimiser = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        loss, before = ppo_loss(self.policy, self.rollout, self.adv, self.ret)
        loss.backward()
        norms = gradient_norms(self.policy)
        self.assertEqual(set(norms), {"log_std", "encoder", "gru", "actor", "critic"})
        self.assertTrue(all(n > 0.0 for n in norms.values()), norms)
        optimiser.step()
        _, after = ppo_loss(self.policy, self.rollout, self.adv, self.ret)
        self.assertLess(after["loss"], before["loss"])

    def test_ratio_is_clipped(self):
        # Pretend the rollout policy was much less likely to take these actions.
        self.rollout.log_probs = self.rollout.log_probs - 1.0
        _, stats = ppo_loss(self.policy, self.rollout, self.adv, self.ret, clip=0.2)
        self.assertEqual(stats["clip_fraction"], 1.0)


class DemoScriptTest(unittest.TestCase):
    def test_demo_runs_every_variant(self):
        done = subprocess.run(
            [sys.executable, "-m", "networks.demo", "--steps", "40"],
            cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace", timeout=300,
        )
        self.assertEqual(done.returncode, 0, msg=done.stderr)
        for variant in VARIANTS:
            self.assertIn(f"{variant}:", done.stdout)
        self.assertEqual(done.stdout.count("Backward pass"), 3)


if __name__ == "__main__":
    unittest.main()
