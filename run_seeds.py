"""Repeat the unchanged training protocol with independent seeds.

    python run_seeds.py --seeds 1 2 --workers 2
    python run_seeds.py --seeds 0 1 2 --variants recurrent temporal --aux-coef 0.5 --workers 6

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


def run_folder(out_root, variant, seed, aux_coef=0.0):
    """runs/<variant>[_pred]_jitter_seed<seed>, as train.py names it."""
    from agent import run_name

    return Path(out_root) / run_name(variant, "jitter", seed, aux_coef)


def reusable(folder, variant, seed, steps, final_episodes, aux_coef=0.0):
    """True if `folder` holds a finished run of exactly this protocol."""
    from agent import TrainConfig

    result_path = Path(folder) / "evaluation.json"
    if not result_path.exists():
        return False
    with result_path.open(encoding="utf-8") as stream:
        previous = json.load(stream)
    # Runs saved before the prediction option existed have no aux_coef: they trained without it.
    cfg = {"aux_coef": 0.0, **previous["config"]}
    expected = asdict(TrainConfig(variant=variant, seed=seed, total_steps=steps, aux_coef=aux_coef))
    return (cfg == expected and previous["steps"] >= steps
            and previous["eval_seed"] == 100_000
            and previous["eval_episodes"] == final_episodes
            and all((Path(folder) / name).exists() for name in ("model.pt", "train_log.csv", "training_curve.png")))


def run_one(variant, seed, args):
    folder = run_folder(args.out_root, variant, seed, args.aux_coef)
    if reusable(folder, variant, seed, args.steps, args.final_episodes, args.aux_coef):
        return f"REUSED {folder}"
    # Check before opening the log: failed reruns should preserve their diagnostics too.
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError(f"nonempty output folder: {folder}; choose a new --out-root")
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "train.py", "--variant", variant, "--seed", str(seed),
               "--steps", str(args.steps), "--final-episodes", str(args.final_episodes),
               "--aux-coef", str(args.aux_coef), "--out", str(folder)]
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
    parser.add_argument("--aux-coef", type=float, default=0.0,
                        help="weight of the crossing-point prediction loss (recurrent variants only); "
                             "runs go to <variant>_pred_jitter_seed<seed>")
    args = parser.parse_args()
    if args.workers < 1 or args.steps < 1 or args.final_episodes < 1 or min(args.seeds) < 0:
        parser.error("workers, steps and final-episodes must be positive; seeds must be nonnegative")
    if args.aux_coef < 0 or (args.aux_coef > 0 and "single_frame" in args.variants):
        parser.error("--aux-coef must be nonnegative and needs recurrent variants (--variants recurrent temporal)")
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
