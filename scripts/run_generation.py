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

value_1 = []
policy_1 = []
episode_returns_1 = []
episode_lengths_1 = []
greedy_1 = []

for seed in range (0,30):
    print(seed)
    env = make_maze_gridworld(seed=seed)
    
    Q, policy, episode_returns, episode_lengths = q_learning(
            env,
            episodes=500000,
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

    value_1.append(Q)
    policy_1.append(policy)
    episode_returns_1.append(episode_returns)
    episode_lengths_1.append(episode_lengths)
    greedy_1.append(total_reward)

N=20
res=greedy_1[:N]
print(str(res))


np.savez_compressed(save_path_1, q_values_1=np.array(value_1), policy_1=np.array(policy_1), episode_returns_1=np.array(episode_returns_1),greedy_1=np.array(greedy_1))

print(greedy_1)