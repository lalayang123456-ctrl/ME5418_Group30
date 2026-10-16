"""Demo: watch the environment in action under a hand-written policy.

    python demo.py                                   live window, reactive "chase" policy, jitter
    python demo.py --policy random --delay burst     random policy, delay with a stall
    python demo.py --policy oracle --delay 0.15      privileged oracle, fixed 150 ms delay
    python demo.py --save demo.gif --episodes 3      write an animation instead of opening a window
    python demo.py --no-render --episodes 200        statistics only

Policies (see ball_catch/controllers.py): random, stationary, chase (uses the observation only),
oracle (privileged: reads the true interception point, but only starts moving after the typical
frame delay plus two camera periods).
Delays: none, fixed, jitter, burst, or a number of seconds.
"""

import argparse
import time

from PIL import Image

from ball_catch import BallCatchEnv, EnvConfig
from ball_catch.controllers import make_controller
from ball_catch.delay import nominal_delay
from ball_catch.evaluation import run_episode

HOLD_SECONDS = 1.0  # how long the final picture of each episode stays on screen


def parse_delay(text: str):
    """A preset name, or a fixed delay in seconds."""
    try:
        return float(text)
    except ValueError:
        return text


def save_gif(frames, path: str, frame_ms: float) -> None:
    palette = Image.Palette.ADAPTIVE
    images = [Image.fromarray(f).convert("P", palette=palette, colors=128) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=frame_ms, loop=0)


def describe(index: int, result: dict, cfg: EnvConfig) -> str:
    if result["caught"]:
        outcome = f"CAUGHT  {result['miss_distance']:.2f} m from the net centre"
    elif result["timeout"]:
        outcome = "TIMEOUT"
    else:
        outcome = f"MISSED  by {result['miss_distance'] - cfg.catch_radius:.2f} m"
    return (
        f"episode {index}: {outcome} | return {result['return']:+.2f} | "
        f"flight {result['flight_time']:.2f} s | "
        f"mean observation age {1000 * result['mean_obs_age']:.0f} ms | "
        f"frames {result['frames_accepted']} accepted, {result['frames_discarded']} discarded"
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default="chase", choices=["random", "stationary", "chase", "oracle"])
    parser.add_argument("--delay", default="jitter", type=parse_delay)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", metavar="FILE.gif", help="write an animation instead of opening a window")
    parser.add_argument("--slowmo", type=float, default=3.0, help="slow-motion factor of the saved animation")
    parser.add_argument("--no-render", action="store_true", help="print the results only")
    parser.add_argument("--no-timing", action="store_true", help="use the observation without timing inputs")
    args = parser.parse_args(argv)

    cfg = EnvConfig(include_timing=not args.no_timing)
    render_mode = None if args.no_render else ("rgb_array" if args.save else "human")
    env = BallCatchEnv(cfg, delay=args.delay, render_mode=render_mode)
    # The oracle waits for the typical frame delay plus two camera periods before it moves, as in
    # calibrate.py; otherwise it would ignore the delay completely.
    controller = make_controller(args.policy, cfg, seed=args.seed, delay=nominal_delay(args.delay, cfg))
    print(f"policy: {args.policy} | delay: {args.delay} | observation size: {env.observation_space.shape[0]}")

    frames = []
    grab_frame = (lambda e: frames.append(e.render())) if args.save else None
    hold_frames = int(HOLD_SECONDS / (cfg.control_dt * args.slowmo))
    caught = 0
    for episode in range(1, args.episodes + 1):
        # In "human" mode the environment draws itself at every reset and step.
        result = run_episode(env, controller, seed=args.seed + episode - 1, on_step=grab_frame)
        caught += result["caught"]
        print(describe(episode, result, cfg))
        if args.save:
            frames.extend([frames[-1]] * hold_frames)
        elif render_mode == "human":
            _pause(HOLD_SECONDS)
    print(f"success rate: {caught}/{args.episodes}")

    if args.save:
        save_gif(frames, args.save, frame_ms=1000.0 * cfg.control_dt * args.slowmo)
        print(f"saved {len(frames)} frames to {args.save}")
    env.close()


def _pause(seconds: float) -> None:
    """Keep the window responsive while waiting."""
    import matplotlib.pyplot as plt

    if plt.get_fignums():
        plt.pause(seconds)
    else:
        time.sleep(seconds)


if __name__ == "__main__":
    main()
