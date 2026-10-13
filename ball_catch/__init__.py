"""Ball catching under visual delay: a Gym environment for time-aware reinforcement learning.

Importing this package registers the environment, so that it can also be created with
`gym.make("BallCatch-v0")`.
"""

from gym.envs.registration import register

from ball_catch.config import EnvConfig
from ball_catch.env import BallCatchEnv

register(id="BallCatch-v0", entry_point="ball_catch.env:BallCatchEnv")

__all__ = ["BallCatchEnv", "EnvConfig"]
