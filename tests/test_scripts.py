"""Smoke tests: the command-line scripts run from start to finish."""

import os
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


class CalibrateScriptTest(unittest.TestCase):
    def test_prints_the_feasibility_table(self):
        done = run_script("calibrate.py", "--episodes", "8")
        self.assertEqual(done.returncode, 0, msg=done.stderr)
        for column in ("stationary", "chase", "oracle"):
            self.assertIn(column, done.stdout)


if __name__ == "__main__":
    unittest.main()
