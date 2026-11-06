"""Validate and summarize completed runs; regenerate the report's multi-seed figure.

    python report/summarize_seeds.py --seeds 0 1 2

For the original seed-0 checkpoints, evaluation.json is created by re-evaluation.
Later runs already contain this file. Every run must use the same training budget,
hyperparameters and evaluation throws. No incomplete run is included in the mean.
"""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import EVAL_SEED, NetworkController, load_policy
from ball_catch import BallCatchEnv
from ball_catch.controllers import make_controller
from ball_catch.delay import nominal_delay
from ball_catch.evaluation import evaluate
from networks.policies import VARIANTS, env_config_for

METRICS = ("success_rate", "mean_return", "mean_miss_distance")


def read_run(folder, episodes, baseline_cache):
    """Read complete, machine-readable results or evaluate an older checkpoint."""
    path = folder / "evaluation.json"
    with (folder / "train_log.csv").open(encoding="utf-8", newline="") as stream:
        log = list(csv.DictReader(stream))
    checkpoint = torch.load(folder / "model.pt", weights_only=True)
    if not log or int(log[-1]["steps"]) != checkpoint["steps"]:
        raise ValueError(f"checkpoint and final training log disagree: {folder}")
    if checkpoint["steps"] < checkpoint["config"]["total_steps"]:
        raise ValueError(f"training is incomplete: {folder}")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["config"] != checkpoint["config"] or data["steps"] != checkpoint["steps"]:
            raise ValueError(f"evaluation and checkpoint disagree: {folder}")
    else:
        policy, cfg = load_policy(str(folder / "model.pt"))
        env_cfg = env_config_for(cfg.variant)
        result = evaluate(BallCatchEnv(env_cfg, delay=cfg.delay), NetworkController(policy),
                          episodes, seed=EVAL_SEED)
        key = (cfg.delay, episodes)
        if key not in baseline_cache:
            baseline_cache[key] = {}
            for name in ("stationary", "chase", "oracle"):
                controller = make_controller(name, env_cfg, delay=nominal_delay(cfg.delay, env_cfg))
                baseline_cache[key][name] = evaluate(BallCatchEnv(env_cfg, delay=cfg.delay),
                                                       controller, episodes, seed=EVAL_SEED)
        data = {"config": asdict(cfg), "steps": checkpoint["steps"], "eval_seed": EVAL_SEED,
                "eval_episodes": episodes,
                "results": {f"{cfg.variant} network": result, **baseline_cache[key]}}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"evaluated {folder.name}", flush=True)
    if data["eval_seed"] != EVAL_SEED or data["eval_episodes"] != episodes:
        raise ValueError(f"evaluation protocol mismatch: {folder}")
    return data, log


def plot_runs(logs, seeds, out, chase):
    """Mean and sample SD over seeds; smooth each training curve before aggregating."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.1))
    for variant, color in zip(VARIANTS, ("tab:blue", "tab:orange", "tab:green")):
        curves = [logs[(variant, seed)] for seed in seeds]
        eval_curves = [[r for r in rows if r["eval_success_rate"]] for rows in curves]
        eval_steps = [int(r["steps"]) for r in eval_curves[0]]
        steps = [int(r["steps"]) for r in curves[0]]
        if any([int(r["steps"]) for r in rows] != eval_steps for rows in eval_curves):
            raise ValueError(f"evaluation schedules differ for {variant}")
        if any([int(r["steps"]) for r in rows] != steps for rows in curves):
            raise ValueError(f"training schedules differ for {variant}")
        catches = np.array([[100 * float(r["eval_success_rate"]) for r in rows] for rows in eval_curves])
        returns = []
        for rows in curves:
            values = [float(r["mean_return"]) for r in rows]
            returns.append([statistics.mean(values[max(0, i - 9):i + 1]) for i in range(len(values))])
        for axis, x, values in ((axes[0], eval_steps, catches), (axes[1], steps, np.array(returns))):
            x = np.array(x) / 1e6
            mean, std = values.mean(0), values.std(0, ddof=1)
            axis.plot(x, mean, color=color, lw=1.2, label=variant)
            axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    if chase is not None:
        axes[0].axhline(100 * chase, color="grey", ls="--", lw=0.9, label="chase controller")
    axes[0].set_ylabel("held-out catch rate (%)")
    axes[1].set_ylabel("training return per throw")
    axes[0].legend(fontsize=6.5, loc="upper left")
    for axis in axes:
        axis.set_xlabel("control steps (millions)")
        axis.grid(alpha=0.3)
        axis.tick_params(labelsize=7)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "learning_curves_multiseed.pdf")
    fig.savefig(out / "learning_curves_multiseed.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--monitor-episodes", type=int, default=200,
                        help="number of throws used by train.py's intermediate evaluations")
    parser.add_argument("--out", type=Path, default=ROOT / "report" / "figures")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        parser.error("supply at least two distinct seeds")
    if args.episodes < 1 or args.monitor_episodes < 1:
        parser.error("episodes and monitor-episodes must be positive")
    torch.set_num_threads(1)
    logs, per_run, baselines = {}, [], {}
    protocol, baseline_rates, monitor_chase = None, None, None
    for variant in VARIANTS:
        for seed in args.seeds:
            folder = args.runs / f"{variant}_jitter_seed{seed}"
            data, logs[(variant, seed)] = read_run(folder, args.episodes, baselines)
            cfg = data["config"]
            if cfg["variant"] != variant or cfg["seed"] != seed:
                raise ValueError(f"run folder and configuration disagree: {folder}")
            current = {k: v for k, v in cfg.items() if k not in ("seed", "variant")}
            if protocol is not None and current != protocol:
                raise ValueError(f"training protocol differs: {folder}")
            protocol = current
            result = data["results"][f"{variant} network"]
            per_run.append({"variant": variant, "seed": seed, "steps": data["steps"],
                            **{metric: result[metric] for metric in METRICS}})
            baseline_rates = {name: {m: data["results"][name][m] for m in METRICS}
                              for name in ("stationary", "chase", "oracle")}
            # Curves and final tables use different evaluation counts. Compare the curve
            # against the same prefix of held-out throws, not the 1,000-throw table value.
            chase_throws = data["results"]["chase"]["results"]
            if len(chase_throws) >= args.monitor_episodes:
                monitor_chase = statistics.mean(r["caught"] for r in chase_throws[:args.monitor_episodes])
    grouped = {}
    for variant in VARIANTS:
        rows = [r for r in per_run if r["variant"] == variant]
        grouped[variant] = {m: {"mean": statistics.mean(r[m] for r in rows),
                                "sample_sd": statistics.stdev(r[m] for r in rows)} for m in METRICS}
    differences = [next(r["success_rate"] for r in per_run if r["variant"] == "temporal" and r["seed"] == seed)
                   - next(r["success_rate"] for r in per_run if r["variant"] == "recurrent" and r["seed"] == seed)
                   for seed in args.seeds]
    summary = {"seeds": args.seeds, "eval_seed": EVAL_SEED, "eval_episodes": args.episodes,
               "training_protocol": protocol, "per_run": per_run, "summary": grouped,
               "baselines": baseline_rates, "temporal_minus_recurrent": differences,
               "monitor_eval_episodes": args.monitor_episodes, "monitor_chase_success_rate": monitor_chase,
               "note": "Sample SD measures training-seed variability, not a confidence interval. "
                       "The fixed held-out set was also used for monitoring and tuning."}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "multiseed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out / "multiseed_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_run[0]))
        writer.writeheader()
        writer.writerows(per_run)
    plot_runs(logs, args.seeds, args.out, monitor_chase)
    print(json.dumps(grouped, indent=2))


if __name__ == "__main__":
    main()
