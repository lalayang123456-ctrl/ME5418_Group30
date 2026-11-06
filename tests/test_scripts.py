"""Smoke tests: the command-line scripts run from start to finish."""

import os
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_script(*args):
    environment = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        encoding="utf-8",  # matches PYTHONIOENCODING; the system default may be a legacy codepage
        errors="replace",
        timeout=300,
    )


class DemoScriptTest(unittest.TestCase):
    def test_runs_without_rendering_and_prints_a_summary(self):
        done = run_script("demo.py", "--no-render", "--episodes", "3", "--policy", "random")
        self.assertEqual(done.returncode, 0, msg=done.stderr)
        self.assertEqual(done.stdout.count("episode "), 3)
        self.assertIn("success rate", done.stdout)

    def test_accepts_a_numeric_delay(self):
        done = run_script("demo.py", "--no-render", "--episodes", "1", "--delay", "0.15")
        self.assertEqual(done.returncode, 0, msg=done.stderr)

    def test_saves_an_animation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "demo.gif")
            done = run_script("demo.py", "--episodes", "1", "--save", path)
            self.assertEqual(done.returncode, 0, msg=done.stderr)
            self.assertGreater(os.path.getsize(path), 10_000)

    def test_live_window_mode_runs_to_completion(self):
        # With the Agg backend no window opens, but the whole "human" code path is exercised.
        done = run_script("demo.py", "--episodes", "1", "--policy", "oracle")
        self.assertEqual(done.returncode, 0, msg=done.stderr)


class TrainScriptTest(unittest.TestCase):
    def test_trains_saves_and_the_demo_replays_the_network(self):
        with tempfile.TemporaryDirectory() as folder:
            done = run_script("train.py", "--variant", "recurrent", "--steps", "256", "--num-envs", "4",
                              "--rollout-steps", "32", "--eval-every", "1", "--eval-episodes", "2",
                              "--final-episodes", "3", "--out", folder)
            self.assertEqual(done.returncode, 0, msg=done.stderr)
            self.assertIn("Held-out evaluation", done.stdout)
            for name in ("model.pt", "train_log.csv", "training_curve.png", "evaluation.json"):
                self.assertTrue(os.path.exists(os.path.join(folder, name)), name)
            with open(os.path.join(folder, "evaluation.json"), encoding="utf-8") as stream:
                evaluation = json.load(stream)
            self.assertEqual(evaluation["config"]["variant"], "recurrent")
            self.assertEqual(evaluation["eval_episodes"], 3)
            self.assertEqual(evaluation["steps"], 256)
            self.assertIn("stationary", evaluation["results"])
            with open(os.path.join(folder, "train_log.csv"), encoding="utf-8") as log:
                self.assertEqual(len(log.readlines()), 1 + 2)  # header and two updates

            model = os.path.join(folder, "model.pt")
            done = run_script("demo.py", "--model", model, "--no-render", "--episodes", "2")
            self.assertEqual(done.returncode, 0, msg=done.stderr)
            self.assertIn("recurrent network", done.stdout)
            self.assertEqual(done.stdout.count("episode "), 2)

    def test_refuses_to_overwrite_previous_results(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.pt")
            with open(path, "wb") as stream:
                stream.write(b"keep this checkpoint")
            done = run_script("train.py", "--steps", "256", "--out", folder)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("choose a new --out folder", done.stderr)
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), b"keep this checkpoint")
            self.assertFalse(os.path.exists(os.path.join(folder, "train_log.csv")))


class CalibrateScriptTest(unittest.TestCase):
    def test_prints_the_feasibility_table(self):
        done = run_script("calibrate.py", "--episodes", "8")
        self.assertEqual(done.returncode, 0, msg=done.stderr)
        for column in ("stationary", "chase", "oracle"):
            self.assertIn(column, done.stdout)


if __name__ == "__main__":
    unittest.main()
