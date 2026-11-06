"""Train the learning agent: PPO on one of the three networks, in the ball-catching environment.

    python train.py --out runs/new_temporal                temporal variant, jitter delay, 3M steps
    python train.py --variant recurrent --out runs/new_recurrent    memory without timestamps
    python train.py --variant single_frame --out runs/new_mlp       reactive baseline
    python train.py --steps 20000 --final-episodes 50 --out runs/smoke_test   quick check

Writes to runs/<variant>_<delay>_seed<seed>/ (or --out):
    model.pt             network weights and training configuration (python demo.py --model ...)
    train_log.csv        one row per update: catch rate, return, losses, KL, held-out evaluations
    training_curve.png   catch rate and return over training
    evaluation.json      configuration, evaluation protocol and per-throw outcomes

Every --eval-every updates the deterministic policy (mean action) is evaluated on held-out throws.
At the end it is compared with the hand-written chase and oracle controllers on the same throws.
1M steps take about 9 minutes on one CPU core for the recurrent networks, 2 minutes for the MLP.
"""

import argparse
import csv
import json
import os
from dataclasses import asdict

import torch

from agent import EVAL_SEED, PPOAgent, TrainConfig
from ball_catch.controllers import make_controller
from ball_catch.delay import nominal_delay
from ball_catch.env import BallCatchEnv
from ball_catch.evaluation import evaluate
from networks.policies import VARIANTS, env_config_for

EVAL_FIELDS = ["eval_success_rate", "eval_mean_return", "eval_miss_distance"]


def parse_delay(text: str):
    """A preset name, or a fixed delay in seconds."""
    try:
        return float(text)
    except ValueError:
        return text


class TrainingLog:
    """Writes one CSV row per update (flushed at once, so an interrupted run keeps its log)."""

    def __init__(self, path: str):
        self.path = path
        self.rows = []
        self._file = None
        self._writer = None

    def write(self, row: dict) -> None:
        if self._writer is None:
            self._file = open(self.path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()
        self.rows.append(row)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def plot_curves(rows, path: str, title: str, references: dict) -> None:
    """Catch rate and return over training; held-out evaluations as markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["steps"] / 1e6 for r in rows]
    evals = [r for r in rows if r["eval_success_rate"] != ""]
    fig, (ax_catch, ax_return) = plt.subplots(1, 2, figsize=(10, 3.6))
    ax_catch.plot(steps, [100 * r["catch_rate"] for r in rows], lw=0.8, alpha=0.6,
                  label="training throws (exploring)")
    if evals:
        ax_catch.plot([r["steps"] / 1e6 for r in evals], [100 * r["eval_success_rate"] for r in evals],
                      "o-", label="held-out throws (mean action)")
    for (name, rate), style in zip(references.items(), ("--", ":")):
        ax_catch.axhline(100 * rate, color="grey", ls=style, lw=1, label=f"{name} controller")
    ax_catch.set_xlabel("control steps (millions)")
    ax_catch.set_ylabel("catch rate (%)")
    ax_catch.legend(fontsize=8)
    ax_return.plot(steps, [r["mean_return"] for r in rows], lw=0.8)
    ax_return.set_xlabel("control steps (millions)")
    ax_return.set_ylabel("mean return per throw")
    for ax in (ax_catch, ax_return):
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    default = TrainConfig()  # the defaults of every training option live in TrainConfig
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=list(VARIANTS), default=default.variant)
    parser.add_argument("--delay", default=default.delay, type=parse_delay,
                        help="delay model for training and evaluation: none, fixed, jitter, burst or seconds")
    parser.add_argument("--steps", type=int, default=default.total_steps, help="total control steps of training")
    parser.add_argument("--seed", type=int, default=default.seed)
    parser.add_argument("--num-envs", type=int, default=default.num_envs, help="environments played in parallel")
    parser.add_argument("--rollout-steps", type=int, default=default.rollout_steps,
                        help="steps per environment per update")
    parser.add_argument("--epochs", type=int, default=default.epochs, help="passes over each rollout")
    parser.add_argument("--lr", type=float, default=default.lr, help="initial Adam learning rate")
    parser.add_argument("--gamma", type=float, default=default.gamma, help="discount factor")
    parser.add_argument("--lam", type=float, default=default.lam, help="GAE lambda")
    parser.add_argument("--normalize-obs", action=argparse.BooleanOptionalAction, default=default.normalize_obs,
                        help="standardise the inputs with running statistics")
    parser.add_argument("--eval-every", type=int, default=50, help="updates between evaluations (0: never)")
    parser.add_argument("--eval-episodes", type=int, default=200, help="throws per evaluation during training")
    parser.add_argument("--final-episodes", type=int, default=1000, help="throws of the final comparison")
    parser.add_argument("--log-every", type=int, default=10, help="updates between printed lines")
    parser.add_argument("--out", help="output folder (default runs/<variant>_<delay>_seed<seed>)")
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.epochs <= 0 or args.log_every <= 0:
        parser.error("steps, epochs and log-every must be positive")
    if args.final_episodes <= 0 or args.eval_episodes <= 0 or args.eval_every < 0:
        parser.error("episode counts must be positive and eval-every must be nonnegative")

    # The networks are small: one thread is faster than PyTorch's default multi-threading here.
    torch.set_num_threads(1)
    config = TrainConfig(
        variant=args.variant, delay=args.delay, total_steps=args.steps, seed=args.seed,
        num_envs=args.num_envs, rollout_steps=args.rollout_steps, epochs=args.epochs, lr=args.lr,
        gamma=args.gamma, lam=args.lam, normalize_obs=args.normalize_obs,
    )
    out = args.out or os.path.join("runs", f"{args.variant}_{args.delay}_seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    # A quick check must not silently destroy the models used in the report.
    artifacts = ("model.pt", "train_log.csv", "training_curve.png", "evaluation.json")
    if any(os.path.exists(os.path.join(out, name)) for name in artifacts):
        parser.error(f"output folder already contains results: {out}; choose a new --out folder")
    model_path = os.path.join(out, "model.pt")

    agent = PPOAgent(config)
    n_params = sum(p.numel() for p in agent.policy.parameters())
    print(f"variant {config.variant} ({n_params:,} parameters) | delay {config.delay} | "
          f"{agent.num_updates} updates of {config.batch_steps} steps | output {out}")
    print(f"{'update':>6} {'steps':>9} {'catch':>6} {'return':>7} {'value loss':>10} "
          f"{'KL':>7} {'clip':>5} {'std':>5} {'held-out':>8}")
    log = TrainingLog(os.path.join(out, "train_log.csv"))

    def on_update(agent: PPOAgent, stats: dict) -> None:
        row = dict(stats, **{key: "" for key in EVAL_FIELDS})
        last = agent.updates == agent.num_updates
        if args.eval_every and (agent.updates % args.eval_every == 0 or last):
            result = agent.evaluate(args.eval_episodes)
            row.update(eval_success_rate=result["success_rate"], eval_mean_return=result["mean_return"],
                       eval_miss_distance=result["mean_miss_distance"])
            agent.save(model_path)  # keep the latest evaluated network if the run is interrupted
        log.write(row)
        if agent.updates % args.log_every == 0 or last or row["eval_success_rate"] != "":
            held_out = "" if row["eval_success_rate"] == "" else f"{row['eval_success_rate']:.1%}"
            print(f"{stats['update']:>6} {stats['steps']:>9} {stats['catch_rate']:>6.1%} "
                  f"{stats['mean_return']:>+7.3f} {stats['value_loss']:>10.4f} {stats['approx_kl']:>7.4f} "
                  f"{stats['clip_fraction']:>5.2f} {stats['action_std']:>5.2f} {held_out:>8}")

    try:
        agent.train(on_update)
    finally:
        log.close()
        agent.save(model_path)
    print(f"saved {model_path}")

    # Final comparison on the same held-out throws, sensor noise and delays.
    print(f"\nHeld-out evaluation: {args.final_episodes} throws (seeds from {EVAL_SEED}), delay {config.delay}")
    env_cfg = env_config_for(config.variant)
    results = {f"{config.variant} network": agent.evaluate(args.final_episodes)}
    for name in ("stationary", "chase", "oracle"):
        controller = make_controller(name, env_cfg, delay=nominal_delay(config.delay, env_cfg))
        env = BallCatchEnv(env_cfg, delay=config.delay)
        results[name] = evaluate(env, controller, args.final_episodes, seed=EVAL_SEED)
        env.close()
    for name, result in results.items():
        print(f"  {name:<24} success {result['success_rate']:6.1%} | mean return {result['mean_return']:+.3f}"
              f" | mean miss distance {result['mean_miss_distance']:.3f} m")

    # Preserve every final metric and the evaluation protocol for multi-seed aggregation.
    with open(os.path.join(out, "evaluation.json"), "w", encoding="utf-8") as stream:
        json.dump({"config": asdict(config), "steps": agent.steps, "eval_seed": EVAL_SEED,
                   "eval_episodes": args.final_episodes, "results": results}, stream, indent=2)

    plot_path = os.path.join(out, "training_curve.png")
    plot_curves(log.rows, plot_path, f"{config.variant} network, {config.delay} delay",
                {name: results[name]["success_rate"] for name in ("chase", "oracle")})
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
