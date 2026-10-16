"""Pilot calibration: is catching feasible but non-trivial, and does the delay matter?

    python calibrate.py                    tables for the current defaults of EnvConfig
    python calibrate.py --episodes 2000    more episodes, tighter estimates
    python calibrate.py --sweep            also sweep the net's acceleration and velocity limits

Three hand-written controllers bracket the difficulty of the task for every delay:

    stationary  the net never moves: what you get for free.
    chase       reactive and non-privileged: follows the line of sight to the last seen ball.
    oracle      privileged upper bound: knows the true interception point, but only starts moving
                once that information could exist, i.e. after the delay plus two camera periods
                (two frames are the minimum needed to estimate a velocity).

The task is well posed if the oracle catches almost everything without delay (feasible), the
stationary net catches almost nothing (non-trivial), and the success rates drop as the delay
grows (the delay matters).
"""

import argparse

import numpy as np

from ball_catch import BallCatchEnv, EnvConfig
from ball_catch.controllers import (
    ChaseController,
    OracleController,
    StationaryController,
    oracle_reaction_time,
)
from ball_catch.evaluation import evaluate

DELAYS = [0.0, 0.05, 0.10, 0.15, 0.20]
SHORT_FLIGHT, LONG_FLIGHT = 0.8, 1.0  # boundaries of the flight-time bins (s)


def rate(results, keep=lambda r: True) -> float:
    selected = [r["caught"] for r in results if keep(r)]
    return float(np.mean(selected)) if selected else float("nan")


def print_parameters(cfg: EnvConfig) -> None:
    t_lo, t_hi = cfg.flight_time_range
    d_lo, d_hi = cfg.launch_distance_range
    farthest = float(np.linalg.norm(cfg.workspace_half_extent))
    print("Task parameters")
    print(f"  flight time         {t_lo:.2f} - {t_hi:.2f} s")
    print(f"  launch distance     {d_lo:.1f} - {d_hi:.1f} m  (approach speed {d_lo / t_hi:.1f} - {d_hi / t_lo:.1f} m/s)")
    print(f"  camera              {cfg.camera_rate:.0f} Hz  ({t_lo * cfg.camera_rate:.0f} - {t_hi * cfg.camera_rate:.0f} frames per flight)")
    print(f"  control step        {1000 * cfg.control_dt:.1f} ms  ({t_lo / cfg.control_dt:.0f} - {t_hi / cfg.control_dt:.0f} steps per flight)")
    print(f"  maximum delay       {cfg.delay_max:.2f} s  ({100 * cfg.delay_max / t_hi:.0f}% - {100 * cfg.delay_max / t_lo:.0f}% of the flight time)")
    print(f"  net limits          {cfg.net_a_max:.1f} m/s^2 and {cfg.net_v_max:.1f} m/s per axis; starts at rest in the workspace centre")
    print(f"  catch radius        {cfg.catch_radius:.2f} m  (net {cfg.net_radius:.2f} m, ball {cfg.ball_radius:.2f} m)")
    print(f"  farthest crossing   {farthest:.2f} m from the net's starting position")
    print()


def print_success_table(cfg: EnvConfig, episodes: int, seed: int) -> None:
    print(f"Success rate over {episodes} throws (every controller faces the same throws)")
    print("  delay   stationary   chase   | oracle: all   T<%.1fs   T>=%.1fs" % (SHORT_FLIGHT, LONG_FLIGHT))
    for delay in DELAYS:
        env = BallCatchEnv(cfg, delay=delay)
        idle = evaluate(env, StationaryController(), episodes, seed)
        chase = evaluate(env, ChaseController(cfg), episodes, seed)
        oracle = evaluate(env, OracleController(cfg, oracle_reaction_time(cfg, delay)), episodes, seed)
        short = rate(oracle["results"], lambda r: r["flight_time"] < SHORT_FLIGHT)
        long_ = rate(oracle["results"], lambda r: r["flight_time"] >= LONG_FLIGHT)
        print(
            f"  {delay:.2f} s  {idle['success_rate']:9.1%}  {chase['success_rate']:7.1%}   |"
            f" {oracle['success_rate']:11.1%}  {short:7.1%}  {long_:8.1%}"
        )
    print()


def print_delay_statistics(cfg: EnvConfig, episodes: int, seed: int) -> None:
    print("Latency experienced by the controller under each delay preset")
    print("  preset   mean age   max age   mean capture interval   frames discarded")
    for preset in ("none", "fixed", "jitter", "burst"):
        stats = evaluate(BallCatchEnv(cfg, delay=preset), StationaryController(), episodes, seed)
        print(
            f"  {preset:<7} {1000 * stats['mean_obs_age']:6.0f} ms {1000 * stats['max_obs_age']:6.0f} ms"
            f" {1000 * stats['mean_capture_interval']:15.0f} ms {stats['discard_fraction']:17.1%}"
        )
    print()


def print_limit_sweep(episodes: int, seed: int) -> None:
    base = EnvConfig()
    print("Oracle success rate for different net limits: no delay -> maximum delay")
    print("  a_max \\ v_max " + "".join(f"{v:>18.1f} m/s" for v in (2.0, 3.0, 4.0)))
    for a_max in (5.0, 10.0, 15.0, 20.0):
        cells = []
        for v_max in (2.0, 3.0, 4.0):
            cfg = EnvConfig(net_a_max=a_max, net_v_max=v_max)
            pair = []
            for delay in (0.0, base.delay_max):
                env = BallCatchEnv(cfg, delay=delay)
                oracle = OracleController(cfg, oracle_reaction_time(cfg, delay))
                pair.append(evaluate(env, oracle, episodes, seed)["success_rate"])
            cells.append(f"{pair[0]:9.1%} -> {pair[1]:6.1%}")
        print(f"  {a_max:5.1f} m/s^2  " + "  ".join(cells))
    print()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", action="store_true", help="also sweep the net limits (slow)")
    args = parser.parse_args(argv)

    cfg = EnvConfig()
    print_parameters(cfg)
    print_success_table(cfg, args.episodes, args.seed)
    print_delay_statistics(cfg, args.episodes, args.seed)
    if args.sweep:
        print_limit_sweep(args.episodes, args.seed)


if __name__ == "__main__":
    main()
