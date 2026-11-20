# Ball catching under visual delay

A Gym environment for the ME5418 project *Time-Aware Reinforcement Learning for Ball Catching
under Visual Delay*. A ball is thrown towards a net that can slide in a vertical plane. The agent
commands the net's acceleration. It knows the net's own state instantly, but it sees the ball only
through camera frames that arrive late, at irregular times and sometimes out of order.

The research question the environment is built for: does telling a recurrent policy *when* each
frame was captured (capture interval and observation age) help it beyond memory alone?

**Project reports:** [OpenAI Gym Code Report](report/OpenAI%20Gym%20Code%20Report.pdf),
[Neural Network Code Report](report/Neural%20Network%20Code%20Report.pdf), and
[Learning Agent Code Report](report/Learning%20Agent%20Code%20Report.pdf).
Their LaTeX sources are also included in `report/`. Experimental data and plots are in
`report/figures/`, including the `additional_experiments/` subdirectory.

## Setup

Run these commands from the project root (the directory containing `environment.yml`).

```
conda env create -f environment.yml
conda activate ball-catch
```

Python 3.9.25 with `gym 0.26.2`, `numpy`, `matplotlib`, `pillow` and `torch 2.5.1` (CPU is enough).
`environment.yml` pins the validated Python/pip versions and Python dependencies, including
transitive dependencies tested on Windows CPU. Linux may additionally install platform-specific CUDA packages.
No external simulator is needed.
Without conda, create a Python 3.9.25 virtual environment and `pip install` the packages pinned in
`environment.yml`. On Linux, pip installs the CUDA build of PyTorch (about 2 GB); it also runs on a
CPU-only machine, it is just a larger download.

### Running on Google Colab

The validated environment is the Conda setup above. Colab uses a different, pre-installed package
stack, so it is a convenience option rather than the pinned reproduction environment. With NumPy 2,
Gym's checker needs the `np.bool8` reference changed to `np.bool_`.
Upload and extract the project archive into Colab before running the following cells:

```python
%cd ME5418_Group30
!pip install -q gym==0.26.2
import gym, os
!sed -i 's/np\.bool8/np.bool_/g' {os.path.dirname(gym.__file__)}/utils/passive_env_checker.py
!python -m unittest
!python -m networks.demo
!python train.py --steps 200000 --out runs/colab_check
```

Colab cannot open the live window. Save an animation and show it instead:

```python
!python demo.py --save demo.gif
from IPython.display import Image
Image("demo.gif")
```

or print the statistics only: `!python demo.py --no-render`.

## Run it

```
python demo.py                                 # live window: reactive "chase" policy, jittered delay
python demo.py --policy random --delay burst   # random policy, delay with a stall
python demo.py --policy oracle --delay 0.15    # privileged oracle, fixed 150 ms delay
python demo.py --save demo.gif --episodes 3    # write an animation instead of opening a window
python demo.py --no-render --episodes 200      # statistics only
python calibrate.py                            # pilot calibration tables
python calibrate.py --sweep                    # also sweep the net's acceleration and velocity limits
python -m networks.demo                        # neural networks: forward and backward pass
python train.py --out runs/new_temporal         # learning agent: train the temporal network with PPO
python train.py --variant recurrent --out runs/new_recurrent  # or single_frame
python demo.py --model runs/temporal_jitter_seed0/model.pt   # watch a trained network
python -m unittest                             # all unit tests (environment, networks, agent)
python evaluate_delays.py --episodes 1000       # compare controllers across delay conditions
python analyze_movement.py --episodes 100      # movement onset and velocity reversals
python tune_predictive.py --episodes 500       # tune predictive PD controller
```

`python demo.py --help` lists every option. Policies: `random`, `stationary`, `chase` (uses the
observation only), `predictive` (ballistic prediction and PD control from observations), `oracle` (privileged, reads the true interception point, but only starts moving
after the typical frame delay plus two camera periods, as in the calibration). Delays: `none`,
`fixed`, `jitter`, `burst` (including `burst_early`, `burst_middle`, and `burst_late` in the burst-pattern experiments), or a number of seconds.

### What the window shows

| Panel | Content |
|---|---|
| Side view | The throw in the x-z plane. Filled dot: where the ball is. Hollow ring: where the controller last saw it. The gap between them is the effect of the delay. |
| Net plane | The plane the net moves in, seen from the camera: net opening, reachable net centres, and the point where the ball will cross. |
| Latest frame | The most recent camera frame that has reached the controller, and how old it is. |
| Frame delivery | One line per camera frame, from its capture time to its arrival time. Slanted lines are delayed frames; a frame overtaken by a newer one is discarded (red, dashed). |

## The task

World frame: the ball flies in +x, the net plane is x = 0, the camera sits behind the net at
x = 2 m and looks towards the thrower; y points right as seen from the camera and z points up.

**Episode.** One throw. The flight time (0.6 - 1.2 s) and the point where the ball crosses the net
plane are sampled, and the launch velocity is solved for. The ball follows a drag-free parabola.
The net starts at rest in the centre of its workspace. The episode ends when the ball crosses the
net plane; the catch is judged at the exact crossing time, between control steps.

**Action.** `Box(-1, 1, (2,))`: net acceleration along (y, z) as a fraction of the per-axis limit.
The net is a double integrator with acceleration, velocity and workspace limits.

**Observation.** `Box` of 11 numbers (9 without timing inputs):

| Index | Content | Range |
|---|---|---|
| 0, 1 | Ball centre (u, v) in the latest accepted frame, normalised image coordinates | [-1, 1] |
| 2 | Apparent ball area in that frame (1 = ball at the net plane) | [0, 2] |
| 3 | 1 if the ball was detected in that frame | {0, 1} |
| 4 | 1 if that frame was accepted in this control step | {0, 1} |
| 5, 6 | Net position, normalised over the workspace | [-1, 1] |
| 7, 8 | Net velocity / velocity limit | [-1, 1] |
| 9 | Capture interval between the last two accepted frames / maximum delay | [0, 5] |
| 10 | Age of the latest accepted frame / maximum delay | [0, 5] |

`EnvConfig(include_timing=False)` drops entries 9 and 10. That is the observation of the baselines
without capture timestamps; everything else, including all random streams, is identical, so the
two variants can be compared on exactly the same throws.

The true ball state, the flight time and the interception point are never observed. They are
returned in `info` for evaluation, analysis and the privileged oracle.

**Delay.** Frame *i* is captured at time S_i (30 Hz) and delivered at S_i + d_i. Control runs at
60 Hz and continues between arrivals. A frame is accepted only if it is newer than every frame
accepted before it; otherwise it is discarded. Delay models (`ball_catch/delay.py`): fixed, jitter
(Gaussian truncated to [0, d_max]), burst (a run of consecutive high-delay frames) and an explicit
sequence (to replay the same delays in a different order).

**Reward.** With *e* the distance between ball centre and net centre at the crossing, and
*R* = net radius - ball radius:

| Outcome | Reward |
|---|---|
| Catch (*e* <= *R*) | +1 + 0.5 (1 - *e*/*R*): between +1 at the rim and +1.5 at the centre |
| Miss (*e* > *R*) | -min(1, (*e* - *R*) / 1 m): close to 0 for a near miss, -1 for a large one |
| Timeout | -1 |
| Every step | -0.05 * ‖action‖² * dt (control effort) |

The reward jumps by at least 1 at the rim, so catching always dominates the shaping terms.

Every parameter lives in `ball_catch/config.py`.

## Pilot calibration

`python calibrate.py --episodes 1000` with the default parameters (same 1000 throws per column):

| Quantity | Value |
|---|---|
| Flight time | 0.6 - 1.2 s (18 - 36 camera frames, 36 - 72 control steps) |
| Camera / control rate | 30 Hz / 60 Hz |
| Maximum delivery delay | 0.20 s, i.e. 17% - 33% of the flight time |
| Net limits | 6 m/s² and 3 m/s per axis, starting at rest in the workspace centre |
| Catch radius | 0.23 m; the farthest crossing point is 1.31 m from the net's start |

Success rate of three other controllers. *Stationary* never moves. *Chase* is reactive and
non-privileged. *Oracle* is a privileged upper bound that knows the true interception point but
only starts moving after the delay plus two camera periods.

| Fixed delay | Stationary | Chase | Oracle, all throws | Oracle, flight < 0.8 s | Oracle, flight >= 1.0 s |
|---|---|---|---|---|---|
| 0.00 s | 5.1% | 18.8% | 100.0% | 100.0% | 100.0% |
| 0.05 s | 5.1% | 13.0% | 98.9% | 96.5% | 100.0% |
| 0.10 s | 5.1% | 10.8% | 95.8% | 86.6% | 100.0% |
| 0.15 s | 5.1% | 9.0% | 92.5% | 76.0% | 100.0% |
| 0.20 s | 5.1% | 8.0% | 88.0% | 62.0% | 100.0% |

Catching is feasible (the oracle catches everything without delay), non-trivial (standing still
catches 5%, reacting without prediction 19%), and the delay matters where there is little time to
recover: short flights lose over a third of their catches at the maximum delay, long flights none.
The acceleration limit was chosen with `python calibrate.py --sweep --episodes 1000`. 6 m/s² is the
smallest limit at which the oracle catches every ball without delay (5 m/s²: 98.9%), and at the
maximum delay it still loses 12% of them (88.0%); with 10 m/s² it caught 97%, so late information
cost almost nothing. At 6 m/s², raising the velocity limit from 3 to 4 m/s changes nothing: the net
never reaches 3 m/s within a throw, so acceleration is the limit that matters.

Latency the controller actually experiences under each delay preset:

| Preset | Mean observation age | Largest age | Mean capture interval | Frames discarded |
|---|---|---|---|---|
| none | 16 ms | 33 ms | 33 ms | 0% |
| fixed (80 ms) | 96 ms | 113 ms | 33 ms | 0% |
| jitter (80 ± 40 ms, truncated to 0 - 200 ms) | 88 ms | 210 ms | 53 ms | 36% |
| burst (jitter plus 4 stalled frames) | 101 ms | 233 ms | 60 ms | 41% |

## Neural networks

`networks/` holds one actor-critic network per PPO variant of the proposal. Each maps observations
to a Gaussian over the 2-D action and a value estimate.

| Variant | Network | Inputs | Role |
|---|---|---|---|
| `single_frame` | MLP: 2 x 64 tanh, separate actor and critic | 9 | Reactive baseline, no memory |
| `recurrent` | encoder 64 -> GRU 64 -> actor and critic heads | 9 | Memory without capture timestamps |
| `temporal` | same as `recurrent` | 11 | Memory plus capture interval and observation age |

The two GRU variants differ only in the two timing inputs (128 extra weights). `python -m networks.demo`
builds each network, plays a short rollout with it (forward pass), then computes the
PPO loss, backpropagates and prints the gradient reaching each part (backward pass).

```python
from networks import make_policy, env_config_for
from ball_catch import BallCatchEnv

policy = make_policy("temporal")
env = BallCatchEnv(env_config_for("temporal"))   # observation size matches the network
```

## Learning agent

`agent/` trains one of the three networks with PPO in the environment; `train.py` is its command
line. One update:

1. **Collect.** 16 environments run in lockstep for 128 control steps (2048 steps, about 38 throws),
   with one batched network call per step. A finished throw is reset at once and marked as an episode
   start, where the GRU clears its memory. Throws and memory carry over to the next rollout.
2. **Advantages.** GAE, no bootstrapping across the end of a throw.
3. **Learn.** 10 epochs of 4 minibatches. A minibatch is 4 environments with their whole
   sequences, and the GRU is re-run from the memory it had at the start of the rollout, exactly as
   during collection. Adam (learning rate 3e-4, decayed linearly to zero), clipped PPO loss,
   gradient norm clipped to 0.5.
4. **Inputs.** The first layer of each network (`ObservationNorm`) standardises the observation with
   running statistics, updated after the learning phase so that a rollout is collected and
   re-evaluated with the same statistics.

Every hyperparameter is in `TrainConfig` (`agent/ppo_agent.py`) and is saved with the model.

```
python train.py --out runs/new_temporal            # temporal network, jitter delay, 3M steps
python train.py --variant recurrent --seed 1 --out runs/new_recurrent
python train.py --steps 50000 --final-episodes 100 --out runs/smoke_test
python train.py --help                            # every option
```

`train.py` writes `runs/<variant>_<delay>_seed<seed>/`: `model.pt`, `train_log.csv` (one row per
update), `training_curve.png` and `evaluation.json` (configuration, evaluation seeds and per-throw results).
An existing results directory is refused; choose a fresh `--out` directory for another experiment.
Every 50 updates the policy is evaluated with its mean action on
200 held-out throws (seeds from 100000, never used to start a training environment). At the end it
is compared with the chase and oracle controllers on the same 1000 held-out throws. One million
steps take about 9 minutes on one CPU core for the recurrent networks and 2 minutes for the MLP.

Repeat the same protocol with independent training seeds (all three variants, 3M steps each):

```
python run_seeds.py --seeds 1 2 --workers 2
python report/summarize_seeds.py --seeds 0 1 2
```

The runner stores each run separately and reuses matching completed runs. Each job uses one CPU
thread; lower `--workers` on a machine with limited memory. Seeds change network initialisation,
action sampling and the training throws. Evaluation uses the same 1,000 throws (seeds 100000--100999)
for every trained policy. Compare means and sample standard deviations across training seeds;
these describe training variability, not a confidence interval or proof of a timestamp advantage.
The summary script validates all checkpoint configurations, recreates missing evaluation files for
legacy runs, and writes per-run CSV, summary JSON and a mean-with-standard-deviation learning curve
to `report/figures/`. The held-out set is also used for monitoring/tuning; it is not an untouched test set.

Results across three training seeds (0, 1, 2), 3M steps each, jitter delay, on the same 1,000 held-out
throws. Seed 0 uses the original saved models; seeds 1 and 2 were trained with the unchanged PPO
settings in the pinned environment. Values are mean ± sample standard deviation across training seeds:

| Controller | Uses | Caught | Mean return | Mean miss distance |
|---|---|---|---|---|
| Stationary | nothing | 5.3% | -0.41 | 0.70 m |
| Chase | latest frame | 10.5% | -0.27 | 0.55 m |
| `single_frame` network | latest frame, net state | 13.4% ± 1.2 pp | -0.18 ± 0.03 | 0.51 ± 0.02 m |
| `recurrent` network | + memory | 22.6% ± 3.0 pp | 0.00 ± 0.08 | 0.43 ± 0.05 m |
| `temporal` network | + memory, capture interval, age | 28.0% ± 3.2 pp | +0.11 ± 0.05 | 0.37 ± 0.02 m |
| Oracle (privileged, delayed start) | true crossing point | 97.4% | +1.37 | 0.03 m |

The temporal network exceeds the recurrent network by 4.9, 5.8 and 5.3 percentage points in seeds
0, 1 and 2, respectively (mean gap 5.3 points). This is preliminary evidence on the current benchmark;
three seeds and an evaluation set also used for monitoring/tuning do not establish general superiority.
The final checkpoint is used for every run, rather than the best intermediate validation score.

Per-run success rates (seeds 0 / 1 / 2): `single_frame` 14.8% / 12.6% / 12.7%;
`recurrent` 23.7% / 25.0% / 19.2%; `temporal` 28.6% / 30.8% / 24.5%.
The exact metrics are in [`multiseed_results.csv`](report/figures/multiseed_results.csv) and
[`multiseed_summary.json`](report/figures/multiseed_summary.json).
Regenerate the mean-and-SD figure with `python report/summarize_seeds.py --seeds 0 1 2`.

A trained network plays through the same interface as the other controllers
(`agent.NetworkController`), so `demo.py --model` shows or evaluates it exactly like them:

```
python demo.py --model runs/temporal_jitter_seed0/model.pt                    # live window
python demo.py --model runs/temporal_jitter_seed0/model.pt --no-render --episodes 500
```

```python
from agent import PPOAgent, TrainConfig, load_policy, NetworkController

agent = PPOAgent(TrainConfig(variant="temporal", total_steps=1_000_000))
agent.train()
agent.save("runs/temporal/model.pt")
print(agent.evaluate(episodes=500)["success_rate"])
```

## Additional experiments and saved results

The repository contains saved experimental artifacts. These are **existing outputs**, not results
that are automatically regenerated by the setup or demo commands. Run scripts from the repository
root, and use a new output prefix to avoid overwriting the supplied files.

### Delay-condition evaluation

`evaluate_delays.py` evaluates the stationary, chase, predictive and privileged oracle baselines,
plus all completed PPO runs it discovers in `runs/`. It includes the original `single_frame`,
`recurrent` and `temporal` variants and the recurrent/temporal models trained with an auxiliary
crossing-point prediction objective (`recurrent_pred_*` and `temporal_pred_*`). The prediction
models use the corresponding recurrent/temporal policy architectures, not separate entries in
`networks/policies.py`'s `VARIANTS` mapping.

```bash
python evaluate_delays.py --episodes 1000 --workers 4 --out report/figures/new_delay_sweep
```

The default conditions are `none`, fixed delays of 0.05, 0.10, 0.15 and 0.20 seconds, `jitter`,
and `burst`. Outputs are `<out>.csv`, `<out>.json`, `<out>.png` and `<out>.pdf`.
The supplied main evaluation files are `report/figures/delay_sweep.*`. The repository also includes `report/figures/final_test.*` (CSV, JSON, PNG and PDF), retained as a separate final-test output set. Consult the accompanying JSON for its recorded evaluation settings.

### Delay-pattern and stall-stage tests

Saved results are grouped under:

- `report/figures/additional_experiments/delay_pattern/`: `delay_pattern`, `delay_pattern_test`
  and `delay_sweep` CSV/JSON data and PNG/PDF plots.
- `report/figures/additional_experiments/stall_stage/`: `stall_stage`, `stall_stage_test`
  and `delay_sweep` data and plots, alongside multi-seed summaries and learning-curve figures.

These directories document separate experimental output sets. Files with `_test` in their names
are retained separately from the corresponding non-`_test` outputs; do not assume they represent
identical protocols without inspecting their metadata.

### Movement analysis

`analyze_movement.py` measures when the net begins moving (default speed threshold 0.05 m/s),
counts per-axis velocity sign reversals above a configurable threshold, and records catch rate
and miss distance. By default, it compares `burst_early`, `burst_middle`, and `burst_late` to test whether delays occurring at different stages of a catching attempt affect the agent's movement. These are comparison conditions, not requirements of the movement analysis; other delay conditions can be selected using `--delays`.

```bash
python analyze_movement.py --episodes 100 --out report/figures/additional_experiments/movement_tests/new_movement
```

It writes `<out>.csv`, `<out>_summary.csv`, `<out>_onset.png` and
`<out>_reversals.png`. Existing `movement.*` and `movement_test.*` results are stored in
`report/figures/additional_experiments/movement_tests/`. Velocity reversals are a specific
sign-change metric, **not** a count of all steering corrections.

### Auxiliary crossing-point prediction training

The `--aux-coef` option in `train.py` sets the weight of an additional crossing-point prediction loss during PPO training. The default is `0` (disabled). A positive value enables auxiliary prediction for the `recurrent` or `temporal` variant; it is not supported for `single_frame`. The true crossing point is used as a training target, not as a policy observation at evaluation time.

```bash
python train.py --variant temporal --aux-coef 0.5 --out runs/new_temporal_pred
```

### Saved trained checkpoints

`runs/` contains the original three-policy seed-0/1/2 checkpoints, prediction-auxiliary
recurrent and temporal checkpoints for seeds 0–4, and
`runs/temporal_reproduction_seed42/`. A typical run contains `model.pt`, `train_log.csv`,
`evaluation.json`, `training_curve.png` and `train_output.txt`; the reproduction folder does
not include `train_output.txt`. The primary three-seed aggregate remains in
`report/figures/multiseed_results.csv` and `multiseed_summary.json`.

The supplied outputs do not, by themselves, establish statistical significance or generalisation
outside the evaluated seeds and delay conditions.

## Code map

| File | Role |
|---|---|
| `ball_catch/config.py` | `EnvConfig`: every physical, sensing, delay and reward parameter |
| `ball_catch/dynamics.py` | Ball trajectory (closed form), launch sampling, net integration with limits |
| `ball_catch/camera.py` | Pinhole camera: projected ball centre and apparent area, with detector noise |
| `ball_catch/delay.py` | Delay models and the frame queue (arrival, acceptance, discarding) |
| `ball_catch/env.py` | `BallCatchEnv` (`gym.Env`): spaces, reset, step, reward, info |
| `ball_catch/rendering.py` | Matplotlib visualisation (`render_mode="human"` or `"rgb_array"`) |
| `ball_catch/controllers.py` | Reference controllers: random, stationary, chase, predictive, oracle |
| `ball_catch/evaluation.py` | Running and summarising episodes |
| `demo.py` | Demo script |
| `calibrate.py` | Pilot calibration: feasibility and sensitivity to delay |
| `tests/` | Unit tests |
| `report/` | Three project reports (PDF and LaTeX), evaluation figures and CSV/JSON results, plus plotting and seed-summary scripts |
| `runs/` | Saved baseline and prediction-auxiliary PPO checkpoints across training seeds, logs, curves, console output and evaluation results; also a temporal reproduction run (seed 42) |
| `networks/policies.py` | Actor-critic networks of the three PPO variants (MLP, GRU, GRU with timing), input standardisation |
| `networks/ppo.py` | Rollout collection in parallel environments, advantages (GAE) and the clipped PPO loss |
| `networks/demo.py` | Neural-network demo: forward and backward pass on environment data |
| `networks/tests/` | Unit tests of the networks |
| `agent/ppo_agent.py` | `TrainConfig` and `PPOAgent`: the PPO training loop, evaluation, saving and loading |
| `agent/controller.py` | `NetworkController`: a trained network behind the other controllers' interface |
| `agent/tests/` | Unit tests of the learning agent |
| `train.py` | Training script: log, learning curve, checkpoint and held-out comparison |
| `run_seeds.py` | Repeat the same training protocol with independent seeds and separate output folders |
| `report/summarize_seeds.py` | Validate completed runs, aggregate training-seed variability and plot the comparison |
| `evaluate_delays.py` | Compare reference controllers and discovered trained checkpoints across delay conditions; export CSV, JSON, PNG and PDF |
| `analyze_movement.py` | Measure movement onset and per-axis velocity reversals; export detailed/summary CSVs and PNGs |
| `tune_predictive.py` | Grid search the predictive controller PD gains and minimum accepted frames |
| `ME5418_Project_Demo.pptx` | Project demonstration presentation |
| `me5418_project_demo.ipynb` | Project demonstration notebook |
| `report/Gang_Yang_A0353152M_Final_Report.pdf` | Individual final project report included in this archive |

## Using the environment from code

```python
import gym
import ball_catch  # registers BallCatch-v0

env = gym.make("BallCatch-v0", delay="jitter")           # or: ball_catch.BallCatchEnv(...)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## AI tool usage declaration

We used Anthropic’s Claude and OpenAI's ChatGPT to help write and debug code, analyze results, and draft and edit parts of the report and presentation. We are responsible for the content and quality of the submitted work.
