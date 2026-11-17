"""Evaluate every controller and trained network across delay conditions (proposal, section 5).

    python evaluate_delays.py                     everything, 1000 held-out throws per condition
    python evaluate_delays.py --episodes 200 --workers 4 --delays none 0.1 jitter burst

Conditions (default): no delay, fixed delays of 0.05-0.2 s, the jittered delay all networks were
trained on, and the burst delay (jitter plus a stall of maximum-delay frames) that no network saw
during training. Every controller faces exactly the same throws, sensor noise and delays: the
seeds start at 100000, the held-out throws of train.py (under jitter, those throws were also used
to monitor training; under the other delays they were never used before).

Controllers: stationary, chase, predictive (ballistic estimator + PD, observation only), oracle
(privileged, waits for the nominal delay plus two camera periods), and every finished run found in
--runs: single_frame, recurrent, temporal, and recurrent/temporal + prediction, one row per seed.

Writes <out>.csv (one row per controller, seed and delay), <out>.json (protocol, delay statistics
actually experienced, mean and sample SD over seeds) and <out>.png/.pdf (catch rate vs delay).
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

from ball_catch import BallCatchEnv, EnvConfig
from ball_catch.controllers import make_controller
from ball_catch.delay import nominal_delay
from ball_catch.evaluation import evaluate

HELD_OUT_SEED = 100_000  # agent.EVAL_SEED; repeated so that the baselines do not need PyTorch
DEFAULT_DELAYS = ["none", "0.05", "0.1", "0.15", "0.2", "jitter", "burst"]
BASELINES = ["stationary", "chase", "predictive", "oracle"]
# (label, run folder prefix, variant, trained with the crossing-point prediction)
GROUPS = [
    ("single_frame", "single_frame_jitter_seed", "single_frame", False),
    ("recurrent", "recurrent_jitter_seed", "recurrent", False),
    ("temporal", "temporal_jitter_seed", "temporal", False),
    ("recurrent + prediction", "recurrent_pred_jitter_seed", "recurrent", True),
    ("temporal + prediction", "temporal_pred_jitter_seed", "temporal", True),
]
METRICS = ["success_rate", "mean_return", "mean_miss_distance"]
DELAY_STATS = ["mean_obs_age", "max_obs_age", "mean_capture_interval", "discard_fraction"]


def delay_spec(text: str):
    """"0.1" -> 0.1 s fixed delay; preset names stay strings."""
    try:
        return float(text)
    except ValueError:
        return text


def discover_runs(runs_dir):
    """[(label, seed, model path)] of every finished run (model.pt and evaluation.json) in runs_dir."""
    found = []
    for label, prefix, _, _ in GROUPS:
        pattern = re.compile(re.escape(prefix) + r"(\d+)$")
        seeds = []
        for name in os.listdir(runs_dir) if os.path.isdir(runs_dir) else []:
            match = pattern.match(name)
            folder = os.path.join(runs_dir, name)
            if match and all(os.path.exists(os.path.join(folder, f)) for f in ("model.pt", "evaluation.json")):
                seeds.append((int(match.group(1)), os.path.join(folder, "model.pt")))
        found += [(label, seed, path) for seed, path in sorted(seeds)]
    return found


def run_job(job):
    """Evaluate one controller (or trained network) under one delay. Runs in a worker process."""
    label, seed, model, delay, episodes, first_seed = job
    spec = delay_spec(delay)
    if model is None:
        cfg = EnvConfig()
        controller = make_controller(label, cfg, delay=nominal_delay(spec, cfg))
    else:
        import torch

        from agent import NetworkController, load_policy
        from networks.policies import env_config_for

        torch.set_num_threads(1)
        policy, train_cfg = load_policy(model)
        _, _, variant, prediction = next(g for g in GROUPS if g[0] == label)
        if train_cfg.variant != variant or train_cfg.predict_crossing != prediction:
            raise ValueError(f"{model} does not hold a {label} network")
        cfg = env_config_for(train_cfg.variant)
        controller = NetworkController(policy)
    started = time.perf_counter()
    result = evaluate(BallCatchEnv(cfg, delay=spec), controller, episodes, seed=first_seed)
    return {"controller": label, "seed": "" if seed is None else seed, "delay": delay,
            **{key: result[key] for key in METRICS + DELAY_STATS},
            "seconds": round(time.perf_counter() - started, 1)}


def summarize(rows):
    """{controller: {delay: {metric: {mean, sd}, n}}}; sd is the sample SD over seeds (None for one run)."""
    cells = {}
    for row in rows:
        cells.setdefault(row["controller"], {}).setdefault(row["delay"], []).append(row)
    summary = {}
    for controller, by_delay in cells.items():
        summary[controller] = {}
        for delay, group in by_delay.items():
            cell = {"n": len(group)}
            for metric in METRICS:
                values = [float(r[metric]) for r in group]
                cell[metric] = {"mean": statistics.mean(values),
                                "sd": statistics.stdev(values) if len(values) > 1 else None}
            summary[controller][delay] = cell
    return summary


# ------------------------------------------------------------------------------------------- figure
# Reference categorical palette (validated for colour-vision deficiency: dataviz validator, all
# pairs). Hue = network variant, line style = with/without the crossing-point prediction.
VARIANT_COLOURS = {"temporal": "#2a78d6", "recurrent": "#eb6834", "single_frame": "#1baf7a"}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
REFERENCE_STYLES = {  # non-learned controllers in neutral ink, told apart by line style and marker
    "predictive": dict(color=INK_2, ls="-", marker="s", label="predictive + PD (observation only)"),
    "oracle": dict(color=MUTED, ls=":", marker="^", label="oracle (privileged)"),
    "chase": dict(color=MUTED, ls="--", marker="v", label="chase (reactive)"),
}


def series_style(label):
    if label in REFERENCE_STYLES:
        return dict(REFERENCE_STYLES[label])
    variant, _, _ = label.partition(" + ")
    prediction = label.endswith("+ prediction")
    return dict(color=VARIANT_COLOURS[variant], ls="-" if prediction else (0, (4, 2)),
                marker="o" if prediction else "D", label=label)


def fixed_delay_seconds(delay, cfg):
    if delay == "none":
        return 0.0
    if delay == "fixed":
        return cfg.jitter_mean
    value = delay_spec(delay)
    return value if isinstance(value, float) else None


def plot(summary, delays, stem, n_runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cfg = EnvConfig()
    fixed = [(fixed_delay_seconds(d, cfg), d) for d in delays if fixed_delay_seconds(d, cfg) is not None]
    fixed.sort()
    random = [d for d in delays if fixed_delay_seconds(d, cfg) is None]
    order = [g[0] for g in GROUPS if g[0] in summary] + [c for c in ("predictive", "oracle", "chase") if c in summary]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                         "text.color": INK})
    panels = [p for p in (fixed, random) if p]
    widths = [3.2 if p is fixed else 1.6 for p in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 3.4), sharey=True,
                             gridspec_kw={"width_ratios": widths}, squeeze=False)
    axes = list(axes[0])
    handles = {}
    for ax, panel in zip(axes, panels):
        is_fixed = panel is fixed
        for k, label in enumerate(order):
            style = series_style(label)
            cells = [summary[label].get(d if not is_fixed else d[1]) for d in panel]
            if any(c is None for c in cells):
                continue
            mean = [100 * c["success_rate"]["mean"] for c in cells]
            sd = [100 * (c["success_rate"]["sd"] or 0.0) for c in cells]
            if is_fixed:
                x = [1000 * s for s, _ in panel]
            else:  # dodge the series around each category so their error bars do not overlap
                x = [i + (k - (len(order) - 1) / 2) * 0.07 for i in range(len(panel))]
            line_style = style["ls"] if is_fixed else "none"
            artist = ax.errorbar(x, mean, yerr=sd if any(sd) else None, color=style["color"], ls=line_style,
                                 lw=1.5, marker=style["marker"], ms=5, mec="white", mew=1.0,
                                 elinewidth=0.9, capsize=2, solid_capstyle="round", zorder=3)
            handles[label] = artist
        if is_fixed:
            ax.set_xticks([1000 * s for s, _ in panel])
            ax.set_xlabel("fixed frame delay (ms)")
            ax.set_title("(a) Fixed delay", loc="left", fontsize=9, color=INK)
        else:
            names = {"jitter": "jitter\n(training)", "burst": "burst\n(unseen)"}
            ax.set_xticks(range(len(panel)))
            ax.set_xticklabels([names.get(d, d) for d in panel])
            ax.set_xlim(-0.6, len(panel) - 0.4)
            ax.set_title("(b) Random delay", loc="left", fontsize=9, color=INK)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("catch rate (%)")
    # Legend entries show line style and marker (error-bar artists would hide the dashes).
    shown = [label for label in order if label in handles]
    proxies = []
    for label in shown:
        style = series_style(label)
        proxies.append(Line2D([], [], color=style["color"], ls=style["ls"], lw=1.5, marker=style["marker"],
                              ms=5, mec="white", mew=1.0))
    fig.legend(proxies, [series_style(label)["label"] for label in shown], loc="lower center",
               ncol=3, frameon=False, fontsize=7.5, handlelength=3.5)
    fig.text(0.99, 0.995, f"{n_runs}; mean ± SD over training seeds", ha="right", va="top",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.17, 1, 0.97))
    for suffix in (".png", ".pdf"):
        fig.savefig(stem + suffix, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- main
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=1000, help="throws per controller and delay")
    parser.add_argument("--seed", type=int, default=HELD_OUT_SEED, help="first evaluation seed")
    parser.add_argument("--delays", nargs="+", default=DEFAULT_DELAYS,
                        help="none, fixed, jitter, burst or seconds")
    parser.add_argument("--runs", default="runs", help="folder with the trained runs")
    parser.add_argument("--out", default=os.path.join("report", "figures", "delay_sweep"),
                        help="output path without extension")
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.workers <= 0:
        parser.error("episodes and workers must be positive")
    if args.seed < HELD_OUT_SEED:
        parser.error(f"evaluation seeds start at {HELD_OUT_SEED} (held-out throws); "
                     "lower seeds belong to training and tuning")
    for delay in args.delays:
        try:
            nominal_delay(delay_spec(delay), EnvConfig())
        except ValueError as error:
            parser.error(str(error))

    runs = discover_runs(args.runs)
    jobs = [(label, seed, path, d, args.episodes, args.seed) for label, seed, path in runs for d in args.delays]
    jobs += [(name, None, None, d, args.episodes, args.seed) for name in BASELINES for d in args.delays]
    print(f"{len(runs)} trained networks + {len(BASELINES)} controllers x {len(args.delays)} delays "
          f"= {len(jobs)} evaluations of {args.episodes} throws (seeds from {args.seed}), "
          f"{args.workers} workers", flush=True)
    started = time.perf_counter()
    if args.workers == 1:
        rows = [run_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(run_job, jobs))
    print(f"done in {time.perf_counter() - started:.0f} s")

    order = {label: i for i, label in enumerate([g[0] for g in GROUPS] + BASELINES)}
    delay_order = {d: i for i, d in enumerate(args.delays)}
    rows.sort(key=lambda r: (order[r["controller"]], str(r["seed"]), delay_order[r["delay"]]))
    folder = os.path.dirname(args.out)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(args.out + ".csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    # The delays a controller experiences do not depend on what it does: report them once.
    delay_statistics = {r["delay"]: {k: r[k] for k in DELAY_STATS} for r in rows if r["controller"] == "predictive"}
    with open(args.out + ".json", "w", encoding="utf-8") as stream:
        json.dump({"protocol": {"episodes": args.episodes, "first_seed": args.seed, "delays": args.delays,
                                "runs": [{"controller": l, "seed": s, "model": p} for l, s, p in runs]},
                   "delay_statistics": delay_statistics, "summary": summary}, stream, indent=2)

    n_seeds = {label: summary[label][args.delays[0]]["n"] for label in summary}
    print(f"\nCatch rate (%), mean ± SD over training seeds, {args.episodes} throws per cell")
    print(f"{'':26s}" + "".join(f"{d:>13s}" for d in args.delays))
    for label in sorted(summary, key=lambda name: order[name]):
        cells = []
        for d in args.delays:
            rate = summary[label][d]["success_rate"]
            sd = "" if rate["sd"] is None else f" ±{100 * rate['sd']:4.1f}"
            cells.append(f"{100 * rate['mean']:6.1f}{sd:>6s}")
        print(f"{label + ' (n=' + str(n_seeds[label]) + ')':26s}" + "".join(f"{c:>13s}" for c in cells))
    print("\nDelay actually experienced")
    for d, stats in delay_statistics.items():
        print(f"  {d:>7s}: mean observation age {1000 * stats['mean_obs_age']:5.0f} ms, largest "
              f"{1000 * stats['max_obs_age']:4.0f} ms, mean capture interval "
              f"{1000 * stats['mean_capture_interval']:3.0f} ms, frames discarded {stats['discard_fraction']:.0%}")

    n_runs = f"{args.episodes} held-out throws per point"
    plot(summary, args.delays, args.out, n_runs)
    print(f"\nsaved {args.out}.csv, .json, .png, .pdf")


if __name__ == "__main__":
    main()
