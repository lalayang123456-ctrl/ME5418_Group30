"""Figure of the Learning Agent report: held-out catch rate of the three variants during training.

    python report/plot_learning_curves.py runs/single_frame_jitter_seed0 runs/recurrent_jitter_seed0 \
        runs/temporal_jitter_seed0 --chase 0.105

Each folder is the output of train.py; the chase and oracle rates are the ones train.py prints at
the end (same held-out throws), drawn as reference lines.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_log(folder: str):
    with open(os.path.join(folder, "train_log.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="output folders of train.py")
    parser.add_argument("--chase", type=float, help="success rate of the chase controller")
    parser.add_argument("--oracle", type=float, help="success rate of the delayed oracle")
    parser.add_argument("--out", default=os.path.join("report", "figures", "learning_curves.pdf"))
    args = parser.parse_args(argv)

    fig, (ax_eval, ax_return) = plt.subplots(1, 2, figsize=(7.2, 2.1))
    for folder in args.runs:
        rows = read_log(folder)
        label = os.path.basename(os.path.normpath(folder)).split("_jitter")[0].split("_none")[0]
        evals = [r for r in rows if r["eval_success_rate"]]
        ax_eval.plot([int(r["steps"]) / 1e6 for r in evals], [100 * float(r["eval_success_rate"]) for r in evals],
                     marker="o", ms=2.5, lw=1.2, label=label)
        # Mean return of the training throws, smoothed over 10 updates.
        steps = [int(r["steps"]) / 1e6 for r in rows]
        returns = [float(r["mean_return"]) for r in rows]
        smooth = [sum(returns[max(0, i - 9):i + 1]) / len(returns[max(0, i - 9):i + 1]) for i in range(len(returns))]
        ax_return.plot(steps, smooth, lw=1.2, label=label)
    for rate, name, style in ((args.chase, "chase controller", "--"), (args.oracle, "oracle (privileged)", ":")):
        if rate is not None:
            ax_eval.axhline(100 * rate, color="grey", ls=style, lw=1, label=name)
    ax_eval.set_xlabel("control steps (millions)")
    ax_eval.set_ylabel("held-out catch rate (%)")
    ax_return.set_xlabel("control steps (millions)")
    ax_return.set_ylabel("training return per throw")
    ax_eval.legend(fontsize=6.5, loc="upper left")
    for ax in (ax_eval, ax_return):
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
