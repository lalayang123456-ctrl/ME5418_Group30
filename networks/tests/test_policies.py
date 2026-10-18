import unittest

import torch

from ball_catch.env import BallCatchEnv
from networks.policies import (
    BASE_OBS_DIM,
    TIMED_OBS_DIM,
    VARIANTS,
    MLPActorCritic,
    RecurrentActorCritic,
    env_config_for,
    make_policy,
)


def random_sequence(policy, steps=7, batch=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(steps, batch, policy.obs_dim, generator=g)


class VariantTest(unittest.TestCase):
    def test_variants_match_the_proposal(self):
        self.assertIsInstance(make_policy("single_frame"), MLPActorCritic)
        self.assertIsInstance(make_policy("recurrent"), RecurrentActorCritic)
        self.assertIsInstance(make_policy("temporal"), RecurrentActorCritic)
        self.assertEqual(make_policy("single_frame").obs_dim, BASE_OBS_DIM)
        self.assertEqual(make_policy("recurrent").obs_dim, BASE_OBS_DIM)
        self.assertEqual(make_policy("temporal").obs_dim, TIMED_OBS_DIM)
        with self.assertRaises(ValueError):
            make_policy("transformer")

    def test_input_size_matches_the_environment_observation(self):
        for variant in VARIANTS:
            env = BallCatchEnv(env_config_for(variant))
            self.assertEqual(env.observation_space.shape[0], make_policy(variant).obs_dim, variant)

    def test_recurrent_variants_differ_only_in_the_timing_inputs(self):
        recurrent, temporal = make_policy("recurrent"), make_policy("temporal")
        n_rec = sum(p.numel() for p in recurrent.parameters())
        n_tmp = sum(p.numel() for p in temporal.parameters())
        # Two extra inputs x 64 encoder units, nothing else.
        self.assertEqual(n_tmp - n_rec, 2 * 64)
        self.assertEqual(recurrent.hidden_size, temporal.hidden_size)


class ShapeTest(unittest.TestCase):
    def test_forward_shapes(self):
        for variant in VARIANTS:
            policy = make_policy(variant)
            mean, value, state = policy(random_sequence(policy))
            self.assertEqual(mean.shape, (7, 3, 2), variant)
            self.assertEqual(value.shape, (7, 3), variant)
            if policy.recurrent:
                self.assertEqual(state.shape, (3, 64))
            else:
                self.assertIsNone(state)

    def test_act_and_evaluate_shapes(self):
        for variant in VARIANTS:
            policy = make_policy(variant)
            out = policy.act(torch.zeros(4, policy.obs_dim), policy.initial_state(4))
            self.assertEqual(out.action.shape, (4, 2))
            self.assertEqual(out.log_prob.shape, (4,))
            self.assertEqual(out.value.shape, (4,))
            obs = random_sequence(policy)
            log_prob, entropy, value = policy.evaluate(obs, torch.zeros(7, 3, 2))
            for tensor in (log_prob, entropy, value):
                self.assertEqual(tensor.shape, (7, 3))

    def test_deterministic_action_is_the_mean(self):
        torch.manual_seed(0)
        policy = make_policy("temporal")
        obs = torch.randn(2, policy.obs_dim)
        out = policy.act(obs, deterministic=True)
        mean, _, _ = policy(obs.unsqueeze(0))
        torch.testing.assert_close(out.action, mean[0])

    def test_initial_actions_are_centred_with_wide_exploration(self):
        # Small output layer: the untrained mean is close to 0, and the spread covers the range.
        policy = make_policy("recurrent")
        mean, _, _ = policy(random_sequence(policy))
        self.assertLess(float(mean.abs().max()), 0.1)
        self.assertAlmostEqual(float(policy.log_std.exp()[0]), 0.607, places=3)


class MemoryTest(unittest.TestCase):
    def test_gru_output_depends_on_history_but_mlp_does_not(self):
        for variant in VARIANTS:
            policy = make_policy(variant)
            last = torch.zeros(1, 1, policy.obs_dim)
            history_a = torch.cat([torch.ones(3, 1, policy.obs_dim), last])
            history_b = torch.cat([-torch.ones(3, 1, policy.obs_dim), last])
            mean_a, _, _ = policy(history_a)
            mean_b, _, _ = policy(history_b)
            differs = not torch.allclose(mean_a[-1], mean_b[-1])
            self.assertEqual(differs, policy.recurrent, variant)

    def test_episode_start_resets_the_memory(self):
        policy = make_policy("temporal")
        obs = random_sequence(policy, steps=6, batch=1)
        start = torch.zeros(6, 1)
        start[3] = 1.0  # a new throw begins at step 3
        mean_joined, _, _ = policy(obs, episode_start=start)
        mean_fresh, _, _ = policy(obs[3:])
        torch.testing.assert_close(mean_joined[3:], mean_fresh)

    def test_stepwise_and_sequence_evaluation_agree(self):
        # The rollout runs one step at a time; the PPO update re-runs the whole sequence.
        torch.manual_seed(1)
        policy = make_policy("temporal")
        obs = random_sequence(policy, steps=8, batch=2)
        start = torch.zeros(8, 2)
        start[0] = 1.0
        start[5, 1] = 1.0
        state, actions, log_probs = policy.initial_state(2), [], []
        for t in range(8):
            out = policy.act(obs[t], state, start[t])
            state = out.state
            actions.append(out.action)
            log_probs.append(out.log_prob)
        log_prob, _, _ = policy.evaluate(obs, torch.stack(actions), policy.initial_state(2), start)
        torch.testing.assert_close(log_prob, torch.stack(log_probs))


class GradientTest(unittest.TestCase):
    def test_every_parameter_receives_a_gradient(self):
        for variant in VARIANTS:
            policy = make_policy(variant)
            obs = random_sequence(policy)
            log_prob, entropy, value = policy.evaluate(obs, torch.randn(7, 3, 2))
            (log_prob.mean() + entropy.mean() + value.mean()).backward()
            for name, param in policy.named_parameters():
                self.assertIsNotNone(param.grad, f"{variant}: {name}")
                self.assertGreater(float(param.grad.abs().sum()), 0.0, f"{variant}: {name}")


if __name__ == "__main__":
    unittest.main()
