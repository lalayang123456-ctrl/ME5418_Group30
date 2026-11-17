"""run_seeds.py: which finished runs may be reused, and where new ones go."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import run_seeds
from agent import TrainConfig


def finished_run(folder: Path, config: dict, steps=3_000_000, episodes=1000):
    folder.mkdir(parents=True)
    for name in ("model.pt", "train_log.csv", "training_curve.png"):
        (folder / name).write_text("x")
    (folder / "evaluation.json").write_text(json.dumps(
        {"config": config, "steps": steps, "eval_seed": 100_000, "eval_episodes": episodes}))


class ReuseTest(unittest.TestCase):
    def test_folder_names(self):
        self.assertEqual(run_seeds.run_folder("runs", "temporal", 1, 0.0), Path("runs/temporal_jitter_seed1"))
        self.assertEqual(run_seeds.run_folder("runs", "temporal", 1, 0.5), Path("runs/temporal_pred_jitter_seed1"))

    def test_runs_saved_before_the_prediction_option_are_still_reused(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "recurrent_jitter_seed1"
            old = asdict(TrainConfig(variant="recurrent", seed=1))
            del old["aux_coef"]
            finished_run(folder, old)
            self.assertTrue(run_seeds.reusable(folder, "recurrent", 1, 3_000_000, 1000, aux_coef=0.0))

    def test_a_run_with_another_prediction_weight_is_not_reused(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "temporal_pred_jitter_seed0"
            finished_run(folder, asdict(TrainConfig(variant="temporal", seed=0, aux_coef=0.5)))
            self.assertTrue(run_seeds.reusable(folder, "temporal", 0, 3_000_000, 1000, aux_coef=0.5))
            self.assertFalse(run_seeds.reusable(folder, "temporal", 0, 3_000_000, 1000, aux_coef=1.0))


if __name__ == "__main__":
    unittest.main()
