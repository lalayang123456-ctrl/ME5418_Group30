"""evaluate_delays.py: every controller and trained network under fixed, jitter and burst delays."""

import csv
import json
import math
import os
import tempfile
import unittest

import evaluate_delays
from agent import PPOAgent, TrainConfig
from tests.test_scripts import run_script


def save_tiny_run(runs, folder, finished=True, **config):
    settings = dict(total_steps=48, num_envs=2, rollout_steps=24, minibatches=2, epochs=1)
    settings.update(config)
    agent = PPOAgent(TrainConfig(**settings))
    agent.train()
    agent.save(os.path.join(runs, folder, "model.pt"))
    if finished:
        with open(os.path.join(runs, folder, "evaluation.json"), "w", encoding="utf-8") as stream:
            stream.write("{}")


class SummaryTest(unittest.TestCase):
    def row(self, seed, rate):
        return {"controller": "temporal", "seed": seed, "delay": "jitter", "success_rate": rate,
                "mean_return": 0.0, "mean_miss_distance": 0.3}

    def test_mean_and_sample_standard_deviation_over_seeds(self):
        cell = evaluate_delays.summarize([self.row(0, 0.2), self.row(1, 0.4)])["temporal"]["jitter"]
        self.assertAlmostEqual(cell["success_rate"]["mean"], 0.3)
        self.assertAlmostEqual(cell["success_rate"]["sd"], math.sqrt(0.02))
        self.assertEqual(cell["n"], 2)

    def test_a_single_run_has_no_standard_deviation(self):
        cell = evaluate_delays.summarize([self.row(0, 0.2)])["temporal"]["jitter"]
        self.assertIsNone(cell["success_rate"]["sd"])


class DiscoveryTest(unittest.TestCase):
    def test_finds_finished_runs_of_every_group_and_skips_unfinished_ones(self):
        with tempfile.TemporaryDirectory() as runs:
            save_tiny_run(runs, "temporal_jitter_seed0", variant="temporal")
            save_tiny_run(runs, "temporal_pred_jitter_seed3", variant="temporal", aux_coef=0.5)
            save_tiny_run(runs, "recurrent_jitter_seed0", finished=False, variant="recurrent")
            found = evaluate_delays.discover_runs(runs)
        self.assertEqual([(label, seed) for label, seed, _ in found],
                         [("temporal", 0), ("temporal + prediction", 3)])


class ScriptTest(unittest.TestCase):
    def test_evaluates_baselines_and_networks_under_every_delay(self):
        with tempfile.TemporaryDirectory() as folder:
            runs = os.path.join(folder, "runs")
            save_tiny_run(runs, "temporal_jitter_seed0", variant="temporal")
            out = os.path.join(folder, "out", "sweep")
            done = run_script("evaluate_delays.py", "--runs", runs, "--episodes", "2", "--delays", "0.1",
                              "burst", "--workers", "1", "--out", out)
            self.assertEqual(done.returncode, 0, msg=done.stderr)
            with open(out + ".csv", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r["controller"] for r in rows},
                             {"stationary", "chase", "predictive", "oracle", "temporal"})
            self.assertEqual({r["delay"] for r in rows}, {"0.1", "burst"})
            self.assertEqual(len(rows), 5 * 2)
            with open(out + ".json", encoding="utf-8") as stream:
                summary = json.load(stream)
            self.assertEqual(summary["protocol"]["episodes"], 2)
            self.assertIn("burst", summary["delay_statistics"])
            self.assertIn("predictive", summary["summary"])
            for suffix in (".png", ".pdf"):
                self.assertTrue(os.path.exists(out + suffix), suffix)

    def test_refuses_held_out_overlap_with_training_seeds(self):
        done = run_script("evaluate_delays.py", "--episodes", "2", "--seed", "5", "--delays", "none")
        self.assertNotEqual(done.returncode, 0)


if __name__ == "__main__":
    unittest.main()
