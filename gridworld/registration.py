"""Gymnasium registration for the ready-made GridWorld / MineWorld MDPs.

Registering the example environments means they can be created the standard
Gymnasium way, which is what most third-party RL code (including
Stable-Baselines3 scripts and benchmark harnesses) expects::

    import gymnasium as gym
    import gridworld            # importing the package registers the ids

    env = gym.make("MineWorld-v0")
    env = gym.make("MineWorld-v0", seed=123, render_mode="rgb_array")

The ``entry_point`` of each id is one of the factory functions in
``examples.py``, so a registered environment and a directly constructed one
are the same object with the same configuration.

Note that these environments implement their own truncation via
``config.max_steps``, so no ``max_episode_steps`` is declared here (which
would wrap them in a second, redundant ``TimeLimit``).
"""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

# id -> factory in gridworld.examples
ENVIRONMENT_IDS = {
    "GridWorld-v0": "gridworld.examples:make_easy_gridworld",
    "GridWorldWalled-v0": "gridworld.examples:make_walled_gridworld",
    "GridWorldSlippery-v0": "gridworld.examples:make_slippery_gridworld",
    "GridWorldShaping-v0": "gridworld.examples:make_reward_shaping_gridworld",
    "GridWorldMaze-v0": "gridworld.examples:make_maze_gridworld",
    "MineWorld-v0": "gridworld.examples:make_mining_gridworld",
    "MineWorldRisky-v0": "gridworld.examples:make_risky_mining_gridworld",
    "MineWorldContinuous-v0": "gridworld.examples:make_continuous_mining_gridworld",
}


def register_envs() -> None:
    """Register every example environment with Gymnasium (idempotent)."""
    for env_id, entry_point in ENVIRONMENT_IDS.items():
        if env_id in registry:
            continue
        register(id=env_id, entry_point=entry_point, disable_env_checker=False)
