"""Choose the gains of the predictive baseline (ballistic estimator + PD controller).

    python tune_predictive.py                     full grid, 500 training throws, jitter delay
    python tune_predictive.py --episodes 100      quicker

Every combination of PD gains (kp, kd) and minimum number of frames before the first prediction
is played on the same training throws (seeds from --seed, default 0). The held-out throws used to
compare controllers start at seed 100000 and are refused here, so the baseline is tuned exactly
like the networks were trained: on throws that are never used for the final comparison.
The chosen values are the defaults of `PredictiveController` in ball_catch/controllers.py.
"""

import argparse
import itertools

from ball_catch import BallCatchEnv, EnvConfig
from ball_catch.controllers import PredictiveController
from ball_catch.evaluation import evaluate

HELD_OUT_SEED = 100_000  # agent.EVAL_SEED, repeated here so this script does not need PyTorch


def parse_delay(text: str):
    try:
        return float(text)
    except ValueError:
        return text


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0, help="first training seed")
    parser.add_argument("--delay", default="jitter", type=parse_delay)
    parser.add_argument("--kp", type=float, nargs="+", default=[10.0, 20.0, 40.0, 80.0, 160.0])
    parser.add_argument("--kd", type=float, nargs="+", default=[4.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0])
    parser.add_argument("--min-frames", type=int, nargs="+", default=[3, 5])
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    if args.seed + args.episodes > HELD_OUT_SEED:
        parser.error(f"seeds from {HELD_OUT_SEED} are the held-out throws; tune on lower seeds")

    cfg = EnvConfig()
    env = BallCatchEnv(cfg, delay=args.delay)
    print(f"{args.episodes} training throws (seeds {args.seed}-{args.seed + args.episodes - 1}), "
          f"delay {args.delay}")
    rows = []
    for kp, kd, frames in itertools.product(args.kp, args.kd, args.min_frames):
        result = evaluate(env, PredictiveController(cfg, kp=kp, kd=kd, min_frames=frames),
                          args.episodes, seed=args.seed)
        rows.append((result["success_rate"], result["mean_return"], kp, kd, frames))
        print(f"kp {kp:6.1f}  kd {kd:5.1f}  min frames {frames} | caught {result['success_rate']:6.1%} "
              f"| mean return {result['mean_return']:+.3f} | miss {result['mean_miss_distance']:.3f} m",
              flush=True)
    # Catch rate is the principal outcome; mean return breaks ties.
    rate, ret, kp, kd, frames = max(rows)
    print(f"best: kp {kp:g}, kd {kd:g}, min frames {frames} | caught {rate:.1%} | mean return {ret:+.3f}")


if __name__ == "__main__":
    main()
