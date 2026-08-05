"""
Trains 30 seeded agents on maze gridworld with Q initialised to zero.
Saves Q, learned policy, per-episode returns, per-episode length, and greedy rollout return for each seed.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

import numpy as np


from gridworld.maze_gridworld import make_maze_gridworld
from gridworld.q_learning import q_learning
from gridworld.plotting import plot_q_values, plot_policy, plot_learning_curve

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

#Values for base maze gridworld with initial q values of zero

os.makedirs(OUTPUT_DIR, exist_ok=True)


save_path_1 = os.path.join(OUTPUT_DIR, "maze_multiseed_null_init.npz")

q_null_maze = []
policy_null_maze = []
episode_returns_null_maze = []
episode_lengths_null_maze = []
greedy_null_maze = []

N_SEEDS = 30 
N_EPISODES = 10_000

for seed in range (0,N_SEEDS):
    print(seed)
    env = make_maze_gridworld(seed=seed)
    
    Q, policy, episode_returns, episode_lengths = q_learning(
            env,
            episodes=N_EPISODES,
            alpha=0.2,
            gamma=0.99,
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

    q_null_maze.append(Q)
    policy_null_maze.append(policy)
    episode_returns_null_maze.append(episode_returns)
    episode_lengths_null_maze.append(episode_lengths)
    greedy_null_maze.append(total_reward)

N=20
res=greedy_null_maze[:N]
print(str(res))

print("EPISODE RETURNS NULL MAZE:", np.array(episode_returns_null_maze).shape)


np.savez_compressed(save_path_1, q_values=np.array(q_null_maze), policy=np.array(policy_null_maze), returns=np.array(episode_returns_null_maze),greedy=np.array(greedy_null_maze))

print(greedy_null_maze)