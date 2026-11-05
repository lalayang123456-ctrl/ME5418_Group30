"""The learning agent of the ME5418 ball-catching project.

`PPOAgent` trains one of the three networks of `networks` (single_frame, recurrent, temporal) on
the environment of `ball_catch` with PPO. `NetworkController` plays a trained network through the
same interface as the hand-written controllers. `python train.py` is the command-line front end.
"""

from agent.controller import NetworkController
from agent.ppo_agent import EVAL_SEED, PPOAgent, TrainConfig, load_policy

__all__ = ["EVAL_SEED", "NetworkController", "PPOAgent", "TrainConfig", "load_policy"]
