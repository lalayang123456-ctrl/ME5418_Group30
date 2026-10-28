# Ball catching under visual delay

A Gym environment for the ME5418 project *Time-Aware Reinforcement Learning for Ball Catching
under Visual Delay*. A ball is thrown towards a net that can slide in a vertical plane. The agent
commands the net's acceleration. It knows the net's own state instantly, but it sees the ball only
through camera frames that arrive late, at irregular times and sometimes out of order.

The research question the environment is built for: does telling a recurrent policy *when* each
frame was captured (capture interval and observation age) help it beyond memory alone?

**Milestone reports:**

- **OpenAI Gym milestone:** [OpenAI Gym Code Report](report/OpenAI%20Gym%20Code%20Report.pdf)
  (2 pages), covering the environment, visual delay model and pilot calibration.
- **Neural Network milestone:** [Neural Network Code Report](report/Neural%20Network%20Code%20Report.pdf)
  (2 pages), covering the three actor-critic networks, PPO components and forward/backward demo.

The LaTeX sources and supporting figure are in `report/`.

## Setup

```
conda env create -f environment.yml
conda activate ball-catch
```

Python 3.9 with `gym 0.26.2`, `numpy`, `matplotlib`, `pillow` and `torch 2.5.1` (CPU is enough).
No external simulator is needed.
Without conda, create a Python 3.9 virtual environment and `pip install` the packages pinned in
`environment.yml`. On Linux, pip installs the CUDA build of PyTorch (about 2 GB); it also runs on a
CPU-only machine, it is just a larger download.

### Running on Google Colab

Colab ships NumPy 2, which gym 0.26.2 does not support (its environment checker uses the removed
`np.bool8`). Patching that one line in Colab's copy of gym is enough; PyTorch is pre-installed.

```python
!git clone https://github.com/lalayang123456-ctrl/ME5418_Group30.git
%cd ME5418_Group30
!pip install -q gym==0.26.2
import gym, os
!sed -i 's/np\.bool8/np.bool_/g' {os.path.dirname(gym.__file__)}/utils/passive_env_checker.py
!python -m unittest
!python -m networks.demo
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
python -m unittest                             # all unit tests (environment and networks)
```

`python demo.py --help` lists every option. Policies: `random`, `stationary`, `chase` (uses the
observation only), `oracle` (privileged, reads the true interception point, but only starts moving
after the typical frame delay plus two camera periods, as in the calibration). Delays: `none`,
`fixed`, `jitter`, `burst`, or a number of seconds.

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

Success rate of three hand-written controllers. *Stationary* never moves. *Chase* is reactive and
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

## Code map

| File | Role |
|---|---|
| `ball_catch/config.py` | `EnvConfig`: every physical, sensing, delay and reward parameter |
| `ball_catch/dynamics.py` | Ball trajectory (closed form), launch sampling, net integration with limits |
| `ball_catch/camera.py` | Pinhole camera: projected ball centre and apparent area, with detector noise |
| `ball_catch/delay.py` | Delay models and the frame queue (arrival, acceptance, discarding) |
| `ball_catch/env.py` | `BallCatchEnv` (`gym.Env`): spaces, reset, step, reward, info |
| `ball_catch/rendering.py` | Matplotlib visualisation (`render_mode="human"` or `"rgb_array"`) |
| `ball_catch/controllers.py` | Hand-written reference controllers: random, stationary, chase, oracle |
| `ball_catch/evaluation.py` | Running and summarising episodes |
| `demo.py` | Demo script |
| `calibrate.py` | Pilot calibration: feasibility and sensitivity to delay |
| `tests/` | Unit tests |
| `report/` | Report for the OpenAI Gym milestone (PDF, LaTeX source, figure) |
| `networks/policies.py` | Actor-critic networks of the three PPO variants (MLP, GRU, GRU with timing) |
| `networks/ppo.py` | Rollout collection, advantages (GAE) and the clipped PPO loss |
| `networks/demo.py` | Neural-network demo: forward and backward pass on environment data |
| `networks/tests/` | Unit tests of the networks |

## Using the environment from code

```python
import gym
import ball_catch  # registers BallCatch-v0

env = gym.make("BallCatch-v0", delay="jitter")           # or: ball_catch.BallCatchEnv(...)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```
