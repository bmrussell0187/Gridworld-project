"""GridWorld: a configurable finite-MDP environment for RL dissertations.

Quick start
-----------
    from gridworld.examples import make_easy_gridworld
    from gridworld.dynamic_programming import value_iteration

    env = make_easy_gridworld()
    V, policy, history = value_iteration(env)

The mining extension (MineWorld) adds a fifth action, MINE, and per-node
mining memory:

    from gridworld.examples import make_mining_gridworld

    env = make_mining_gridworld()
    V, policy, history = value_iteration(env)

Importing this package also registers the example environments with
Gymnasium, so ``gym.make("MineWorld-v0")`` works.

See README.md for the full workflow (dynamic programming, Q-learning,
plotting, animation, and optional Stable-Baselines3 training).
"""

from .config import GridWorldConfig
from .mining_config import MineWorldConfig, ContinuousRewardModelError
from .env import GridWorldEnv
from .mining_env import MineWorldEnv, MiningTransition
from .registration import ENVIRONMENT_IDS, register_envs

__version__ = "0.1.0"

register_envs()

__all__ = [
    "GridWorldConfig",
    "GridWorldEnv",
    "MineWorldConfig",
    "MineWorldEnv",
    "MiningTransition",
    "ContinuousRewardModelError",
    "ENVIRONMENT_IDS",
    "register_envs",
    "__version__",
]
