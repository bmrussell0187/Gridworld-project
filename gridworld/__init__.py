"""GridWorld: a configurable finite-MDP environment for RL dissertations.

Quick start
-----------
    from gridworld.examples import make_easy_gridworld
    from gridworld.dynamic_programming import value_iteration

    env = make_easy_gridworld()
    V, policy, history = value_iteration(env)

See README.md for the full workflow (dynamic programming, Q-learning,
plotting, animation, and optional Stable-Baselines3 training).
"""

from .config import GridWorldConfig
from .env import GridWorldEnv

__all__ = ["GridWorldConfig", "GridWorldEnv"]
