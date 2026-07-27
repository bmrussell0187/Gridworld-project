"""Solve the easy GridWorld MDP exactly using value iteration.

Value iteration is a *model-based* dynamic-programming method: it uses the
environment's known transition model P(s'|s,a) (via
``env.transition_probabilities``) to repeatedly apply the Bellman
optimality equation until the value function converges, then extracts the
greedy (optimal) policy from it.

Run from the repository root:
    python scripts/run_value_iteration.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

from gridworld.examples import make_easy_gridworld
from gridworld.dynamic_programming import value_iteration
from gridworld.plotting import plot_gridworld, plot_value_function, plot_policy
from gridworld.animation import animate_episode

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = make_easy_gridworld(seed=0)

    # Plot the raw MDP layout (state space, rewards, terminals).
    ax = plot_gridworld(env, title="Easy GridWorld: layout")
    ax.get_figure().savefig(os.path.join(OUTPUT_DIR, "easy_gridworld.png"), dpi=150)
    plt.close(ax.get_figure())

    # Solve the MDP with value iteration.
    V, policy, history = value_iteration(env, gamma=0.95, theta=1e-8)
    print(f"Value iteration converged in {len(history)} sweeps "
          f"(final delta={history[-1]:.2e})")

    ax = plot_value_function(env, V, title="Value iteration: V*(s)")
    ax.get_figure().savefig(os.path.join(OUTPUT_DIR, "value_function.png"), dpi=150)
    plt.close(ax.get_figure())

    ax = plot_policy(env, policy, title="Value iteration: optimal policy")
    ax.get_figure().savefig(os.path.join(OUTPUT_DIR, "optimal_policy.png"), dpi=150)
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

    save_path = os.path.join(OUTPUT_DIR, "value_iteration_agent.gif")
    animate_episode(env, save_path=save_path, fps=2, show_policy=policy)
    print(f"Saved animation to {save_path}")


if __name__ == "__main__":
    main()
