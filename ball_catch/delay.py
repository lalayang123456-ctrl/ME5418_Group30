"""Visual delay: how long each camera frame takes to reach the controller.

Frame i is captured at time S_i and delivered at S_i + d_i, where the delivery delay d_i >= 0 is
drawn from a `DelayModel`. Frames can overtake each other, so `FrameQueue` hands the controller
only frames that are newer than everything it has already seen and discards the rest.
"""

import heapq
import math
import numbers
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np

from ball_catch.config import EnvConfig

_TIME_EPS = 1e-9  # tolerance for comparing times that are multiples of the control step


# ------------------------------------------------------------------------------ delay models
def _check_delay(value, name: str = "delay") -> float:
    """Return `value` as a float, or raise if it is not a finite, non-negative number of seconds."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number of seconds, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0 s, got {value!r}")
    return value


class DelayModel:
    """Draws the delivery delay of each frame of an episode."""

    def reset(self, rng: np.random.Generator, flight_time: float) -> None:
        """Called once per episode. `flight_time` is only used to place test conditions (such as
        a stall in the middle of the flight); it is never shown to a policy."""

    def sample(self, frame_index: int, capture_time: float, rng: np.random.Generator) -> float:
        raise NotImplementedError


class FixedDelay(DelayModel):
    """Every frame is delayed by the same amount."""

    def __init__(self, delay: float):
        self.delay = _check_delay(delay)

    def sample(self, frame_index, capture_time, rng):
        return self.delay


class JitterDelay(DelayModel):
    """Independent delays from a Gaussian truncated to [0, d_max]."""

    def __init__(self, mean: float, std: float, d_max: float):
        self.mean = _check_delay(mean, "mean")
        self.std = _check_delay(std, "std")
        self.d_max = _check_delay(d_max, "d_max")

    def sample(self, frame_index, capture_time, rng):
        # Rejection sampling keeps the Gaussian shape inside the bounds instead of piling
        # probability mass onto them, which clipping would do.
        for _ in range(1000):
            delay = rng.normal(self.mean, self.std)
            if 0.0 <= delay <= self.d_max:
                return float(delay)
        return float(np.clip(self.mean, 0.0, self.d_max))


class BurstDelay(DelayModel):
    """A stall: `n_frames` consecutive frames get a high delay, the others follow `base`.

    The stall starts with the first frame captured at or after `start_fraction` of the flight.
    With `start_fraction=None` the start is drawn uniformly from [0.1, 0.7] every episode.
    """

    def __init__(
        self,
        base: DelayModel,
        start_fraction: Optional[float],
        n_frames: int,
        stall_delay: float,
    ):
        self.base = base
        self.start_fraction = start_fraction
        self.n_frames = int(n_frames)
        self.stall_delay = float(stall_delay)
        self._stall_start_time = 0.0
        self._first_stalled_frame: Optional[int] = None

    def reset(self, rng, flight_time):
        self.base.reset(rng, flight_time)
        fraction = rng.uniform(0.1, 0.7) if self.start_fraction is None else self.start_fraction
        self._stall_start_time = fraction * flight_time
        self._first_stalled_frame = None

    def sample(self, frame_index, capture_time, rng):
        # The base delay is always drawn so that the frames outside the stall are identical to
        # those of a stall-free episode with the same seed.
        delay = self.base.sample(frame_index, capture_time, rng)
        if self._first_stalled_frame is None and capture_time >= self._stall_start_time - _TIME_EPS:
            self._first_stalled_frame = frame_index
        first = self._first_stalled_frame
        if first is not None and first <= frame_index < first + self.n_frames:
            return self.stall_delay
        return delay


class SequenceDelay(DelayModel):
    """Replays a given list of delays (cyclically).

    Used to present the same delay values in a different order, e.g. dispersed versus clustered.
    """

    def __init__(self, delays: Sequence[float]):
        self.delays = [_check_delay(d) for d in delays]
        if not self.delays:
            raise ValueError("SequenceDelay needs at least one delay")

    def sample(self, frame_index, capture_time, rng):
        return self.delays[frame_index % len(self.delays)]


def make_delay_model(spec: Union[str, float, DelayModel], cfg: EnvConfig) -> DelayModel:
    """Build a delay model from a preset name, a number (fixed delay in seconds) or an instance.

    Presets: "none", "fixed" (the jitter mean, for a like-for-like comparison with "jitter"),
    "jitter" and "burst" (jitter plus one stall of maximum-delay frames at a random time).
    """
    if isinstance(spec, DelayModel):
        return spec
    if isinstance(spec, numbers.Real) and not isinstance(spec, bool):
        return FixedDelay(spec)  # checks that the delay is finite and non-negative
    if spec == "none":
        return FixedDelay(0.0)
    if spec == "fixed":
        return FixedDelay(cfg.jitter_mean)
    if spec == "jitter":
        return JitterDelay(cfg.jitter_mean, cfg.jitter_std, cfg.delay_max)
    if spec == "burst":
        base = JitterDelay(cfg.jitter_mean, cfg.jitter_std, cfg.delay_max)
        return BurstDelay(base, start_fraction=None, n_frames=4, stall_delay=cfg.delay_max)
    raise ValueError(f"unknown delay model: {spec!r}")


def nominal_delay(spec: Union[str, float, DelayModel], cfg: EnvConfig) -> float:
    """Typical delivery delay (s) of a delay specification, as accepted by `make_delay_model`.

    Used to give the privileged oracle a fair reaction time. For the random presets it is the
    jitter mean; a burst's stall is ignored because it does not affect the first frames.
    """
    if isinstance(spec, numbers.Real) and not isinstance(spec, bool):
        return _check_delay(spec)
    if isinstance(spec, str):
        if spec == "none":
            return 0.0
        if spec in ("fixed", "jitter", "burst"):
            return cfg.jitter_mean
        raise ValueError(f"unknown delay model: {spec!r}")
    if isinstance(spec, FixedDelay):
        return spec.delay
    if isinstance(spec, JitterDelay):
        return spec.mean
    if isinstance(spec, BurstDelay):
        return nominal_delay(spec.base, cfg)
    if isinstance(spec, SequenceDelay):
        return float(np.mean(spec.delays))
    raise ValueError(f"no nominal delay for {type(spec).__name__}")


# ------------------------------------------------------------------------------ frame delivery
@dataclass
class Frame:
    """One camera frame on its way to the controller."""

    index: int
    capture_time: float  # S_i
    arrival_time: float  # S_i + d_i
    features: np.ndarray  # [u, v, area] detected in this frame
    valid: bool  # whether the ball was detected
    status: str = "in_transit"  # "in_transit" -> "accepted" or "discarded"

    @property
    def delay(self) -> float:
        return self.arrival_time - self.capture_time


class FrameQueue:
    """Delivers frames at their arrival time and keeps only the newest information.

    A frame is *accepted* when it is shown to the controller. A frame is *discarded* when, on
    arrival, a newer capture has already been accepted or arrives in the same control step.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.frames: List[Frame] = []  # every frame of the episode, for analysis and rendering
        self._pending: list = []  # heap ordered by arrival time
        self.latest: Optional[Frame] = None  # most recent accepted frame
        self.capture_interval = 0.0  # capture-time gap between the last two accepted frames
        self.accepted_count = 0
        self.discarded_count = 0

    def push(self, frame: Frame) -> None:
        self.frames.append(frame)
        heapq.heappush(self._pending, (frame.arrival_time, frame.index, frame))

    def deliver(self, t: float) -> Optional[Frame]:
        """Process every frame that has arrived by time `t`.

        Returns the newly accepted frame, or None if nothing new reached the controller.
        """
        arrived = []
        while self._pending and self._pending[0][0] <= t + _TIME_EPS:
            arrived.append(heapq.heappop(self._pending)[2])
        if not arrived:
            return None

        newest = max(arrived, key=lambda f: f.capture_time)
        accepted = None
        if self.latest is None or newest.capture_time > self.latest.capture_time:
            if self.latest is not None:
                self.capture_interval = newest.capture_time - self.latest.capture_time
            self.latest = accepted = newest
            newest.status = "accepted"
            self.accepted_count += 1
        for frame in arrived:
            if frame is not accepted:
                frame.status = "discarded"
                self.discarded_count += 1
        return accepted
