"""Repeat the unchanged training protocol with independent seeds.

    python run_seeds.py --seeds 1 2 --workers 2

Existing complete runs are reused only when their saved protocol matches. Incomplete
or different runs are refused by train.py; choose another --out-root to preserve them.
Each subprocess uses one CPU thread and writes its own console log and artifacts.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys


def run_one(variant, seed, args):
    folder = Path(args.out_root) / f"{variant}_jitter_seed{seed}"
    result_path = folder / "evaluation.json"
    if result_path.exists():
        from agent import TrainConfig

        with result_path.open(encoding="utf-8") as stream:
            previous = json.load(stream)
        cfg = previous["config"]
        expected = asdict(TrainConfig(variant=variant, seed=seed, total_steps=args.steps))
        if (cfg == expected and previous["steps"] >= args.steps
                and previous["eval_seed"] == 100_000
                and previous["eval_episodes"] == args.final_episodes
                and all((folder / name).exists() for name in ("model.pt", "train_log.csv", "training_curve.png"))):
            return f"REUSED {folder}"
    # Check before opening the log: failed reruns should preserve their diagnostics too.
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError(f"nonempty output folder: {folder}; choose a new --out-root")
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "train.py", "--variant", variant, "--seed", str(seed),
               "--steps", str(args.steps), "--final-episodes", str(args.final_episodes),
               "--out", str(folder)]
    environment = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    print(f"START {folder}", flush=True)
    with (folder / "train_output.txt").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    if process.returncode:
        raise RuntimeError(f"training failed ({process.returncode}): {folder / 'train_output.txt'}")
    return f"DONE {folder}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--variants", nargs="+", choices=("single_frame", "recurrent", "temporal"),
                        default=["single_frame", "recurrent", "temporal"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--final-episodes", type=int, default=1000)
    parser.add_argument("--out-root", default="runs")
    args = parser.parse_args()
    if args.workers < 1 or args.steps < 1 or args.final_episodes < 1 or min(args.seeds) < 0:
        parser.error("workers, steps and final-episodes must be positive; seeds must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error("seeds and variants must not contain duplicates")
    # Paths and train.py always resolve relative to this repository.
    os.chdir(Path(__file__).resolve().parent)
    failed = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_one, variant, seed, args) for seed in args.seeds for variant in args.variants]
        for job in as_completed(jobs):
            try:
                print(job.result(), flush=True)
            except Exception as error:
                failed = True
                print(f"FAILED {error}", file=sys.stderr, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
