"""
Value iteration on base maze gridworld.
Returns number of sweeps, change in max(V) on final sweep, optimal policy length, and optimal policy return.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

from gridworld.maze_gridworld import make_maze_gridworld
from gridworld.dynamic_programming import value_iteration
from gridworld.plotting import plot_gridworld, plot_value_function, plot_policy
from gridworld.animation import animate_episode

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = make_maze_gridworld(seed=0)

    # Plot the raw MDP layout (state space, rewards, terminals).
    ax = plot_gridworld(env, title="Maze GridWorld: layout")
    ax.get_figure().savefig(os.path.join(OUTPUT_DIR, "maze_gridworld.png"), dpi=150)
    plt.close(ax.get_figure())

    # Solve the MDP with value iteration.
    V, policy, history = value_iteration(env, gamma=0.99, theta=1e-8)
    print(f"Value iteration converged in {len(history)} sweeps "
          f"(final delta={history[-1]:.2e})")

    ax = plot_value_function(env, V, title="Value iteration: V*(s)")
    ax.get_figure().savefig(os.path.join(OUTPUT_DIR, "value_function_maze.png"), dpi=150)
    plt.close(ax.get_figure())

    ax = plot_policy(env, policy, title="Value iteration: optimal policy")
    ax.get_figure().savefig(os.path.join(OUTPUT_DIR, "optimal_policy_maze.png"), dpi=150)
    plt.close(ax.get_figure())

    # Roll out one episode following the optimal (greedy) policy and animate it.
    obs, info = env.reset(seed=0)
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action = int(policy[obs])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    print(f"Optimal policy episode: steps={info['step']}, total_reward={total_reward:.3f}")

    save_path = os.path.join(OUTPUT_DIR, "value_iteration_maze.gif")
    animate_episode(env, save_path=save_path, fps=2, show_policy=policy)
    print(f"Saved animation to {save_path}")



if __name__ == "__main__":
    main()