"""
Trains 30 seeded agents on maze gridworld with shaping rewards.
Saves Q, learned policy, per-episode returns, per-episode length, and greedy rollout return for each seed.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

import numpy as np


from gridworld.maze_gridworld_shaping import make_maze_gridworld
from gridworld.q_learning import q_learning
from gridworld.plotting import plot_q_values, plot_policy, plot_learning_curve

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

#Values for base maze gridworld with initial q values of zero

os.makedirs(OUTPUT_DIR, exist_ok=True)


save_path = os.path.join(OUTPUT_DIR, "maze_multiseed_shaping_init.npz")

q_shaping_maze = []
policy_shaping_maze = []
episode_returns_shaping_maze = []
episode_lengths_shaping_maze = []
greedy_shaping_maze = []

for seed in range (0,30):
    print(seed)
    env = make_maze_gridworld(seed=seed)
    Q, policy, episode_returns, episode_lengths = q_learning(
            env,
            episodes=300000,
            alpha=0.2,
            gamma=0.999,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.9999,
            seed=seed,
            q_init=0.0,
        )
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action = int(policy[obs])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    q_shaping_maze.append(Q)
    policy_shaping_maze.append(policy)
    episode_returns_shaping_maze.append(episode_returns)
    episode_lengths_shaping_maze.append(episode_lengths)
    greedy_shaping_maze.append(total_reward)

N=20
res=greedy_shaping_maze[:N]
print(str(res))


np.savez_compressed(save_path, q_null_maze=np.array(q_shaping_maze), policy_1=np.array(policy_shaping_maze), episode_returns_null_maze=np.array(episode_returns_shaping_maze),greedy_null_maze=np.array(greedy_shaping_maze))

print(greedy_shaping_maze)