"""Movement onset and velocity-reversal analysis for ME5418 ball catching.

Run from the repository root:
  python analyze_movement.py --episodes 100 --delays burst_early burst_middle burst_late

Requires trained runs in ./runs (same naming as evaluate_delays.py).
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from ball_catch import BallCatchEnv, EnvConfig
from ball_catch.controllers import make_controller
from ball_catch.delay import nominal_delay
from ball_catch.evaluation import run_episode
from evaluate_delays import GROUPS, discover_runs, delay_spec


def load_controller(label, model, spec):
    if model is None:
        cfg = EnvConfig()
        return cfg, make_controller(label, cfg, delay=nominal_delay(spec, cfg))
    import torch
    from agent import NetworkController, load_policy
    from networks.policies import env_config_for
    torch.set_num_threads(1)
    policy, train_cfg = load_policy(model)
    _, _, variant, prediction = next(g for g in GROUPS if g[0] == label)
    if train_cfg.variant != variant or train_cfg.predict_crossing != prediction:
        raise ValueError(f"Model does not match {label}: {model}")
    return env_config_for(train_cfg.variant), NetworkController(policy)


def metrics(samples, speed_threshold, reversal_threshold):
    t = np.asarray([s[0] for s in samples], dtype=float)
    v = np.asarray([s[3:5] for s in samples], dtype=float)
    speed = np.linalg.norm(v, axis=1)
    active = np.flatnonzero(speed > speed_threshold)
    onset = float(t[active[0]]) if active.size else float('nan')
    # Count sign reversals of velocity in each axis, ignoring velocities near zero.
    reversals = 0
    for axis in range(2):
        significant = v[np.abs(v[:, axis]) >= reversal_threshold, axis]
        if len(significant) > 1:
            reversals += int(np.count_nonzero(np.diff(np.sign(significant))))
    return onset, reversals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=100000)
    p.add_argument('--delays', nargs='+', default=['burst_early', 'burst_middle', 'burst_late'])
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='report/figures/movement_analysis')
    p.add_argument('--speed-threshold', type=float, default=0.05, help='m/s: movement onset')
    p.add_argument('--reversal-threshold', type=float, default=0.05, help='m/s: ignore tiny velocity sign changes')
    args = p.parse_args()
    if args.episodes < 1 or args.speed_threshold <= 0 or args.reversal_threshold <= 0:
        p.error('episodes and thresholds must be positive')
    runs = discover_runs(args.runs)
    if not runs:
        p.error(f'No trained models found in {args.runs!r}; run from project root or specify --runs')
    controllers = [(label, seed, model) for label, seed, model in runs]
    controllers += [(name, '', None) for name in ('stationary', 'chase', 'predictive', 'oracle')]
    rows = []
    for label, train_seed, model in controllers:
        for delay in args.delays:
            spec = delay_spec(delay)
            cfg, controller = load_controller(label, model, spec)
            env = BallCatchEnv(cfg, delay=spec)
            for episode in range(args.episodes):
                samples = []
                def capture(e):
                    samples.append((float(e.time), float(e.net_pos[0]), float(e.net_pos[1]),
                                    float(e.net_vel[0]), float(e.net_vel[1])))
                outcome = run_episode(env, controller, seed=args.seed + episode, on_step=capture)
                onset, corrections = metrics(samples, args.speed_threshold, args.reversal_threshold)
                rows.append(dict(controller=label, training_seed=train_seed, delay=delay,
                                 throw_seed=args.seed + episode, caught=outcome['caught'],
                                 miss_distance=outcome['miss_distance'], flight_time=outcome['flight_time'],
                                 movement_onset_s=onset, velocity_reversals=corrections))
            print(f'{label} seed={train_seed} {delay}: {args.episodes} throws', flush=True)
    folder = os.path.dirname(args.out)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(args.out + '.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['controller'], row['delay'])].append(row)
    summary = []
    for (controller, delay), group in sorted(grouped.items()):
        onset = np.array([r['movement_onset_s'] for r in group])
        valid = onset[np.isfinite(onset)]
        summary.append(dict(controller=controller, delay=delay, n=len(group),
                            catch_rate=float(np.mean([r['caught'] for r in group])),
                            mean_onset_s=float(np.mean(valid)) if len(valid) else float('nan'),
                            no_onset_fraction=float(1-len(valid)/len(group)),
                            mean_velocity_reversals=float(np.mean([r['velocity_reversals'] for r in group])),
                            mean_miss_distance=float(np.nanmean([r['miss_distance'] for r in group]))))
    with open(args.out + '_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric, ylabel, suffix in [('mean_onset_s', 'Movement onset (s)', 'onset'),
                                    ('mean_velocity_reversals', 'Velocity reversals per throw', 'reversals')]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        labels = list(dict.fromkeys(r['controller'] for r in summary))
        delays = args.delays
        x = np.arange(len(labels))
        width = 0.8 / len(delays)
        for i, d in enumerate(delays):
            values = [next((r[metric] for r in summary if r['controller'] == label and r['delay'] == d), np.nan)
                      for label in labels]
            ax.bar(x + (i-(len(delays)-1)/2)*width, values, width, label=d)
        ax.set_xticks(x, labels, rotation=30, ha='right')
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out + '_' + suffix + '.png', dpi=180)
        plt.close(fig)
    print('Saved:', args.out + '.csv, ' + args.out + '_summary.csv and two PNG plots')
    print('Note: reversals are per-axis velocity sign changes, not all steering corrections.')


if __name__ == '__main__':
    main()
