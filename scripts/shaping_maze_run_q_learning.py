"""
Runs q-learning on the maze gridworld with shaping rewards. 
Returns a greedy policy return and step length, average return over final 100 training runs.
Also returns 4 image files: A gif of agent following learned policy, a curve displaying moving average and 
    per-episode return by episode, final learned policy, and learned state-action values.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

from gridworld.maze_gridworld_shaping import make_maze_gridworld
from gridworld.q_learning import q_learning
from gridworld.plotting import plot_q_values, plot_policy, plot_learning_curve
from gridworld.animation import animate_episode

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = make_maze_gridworld(seed=0)

    Q, policy, episode_returns, episode_lengths = q_learning(
        env,
    
        episodes=30000,
        alpha=0.2,
        gamma=0.999,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9999,
        seed=0,
        q_init=0.0,
    )

    print("")
    print(
        f"Q-learning finished {len(episode_returns)} episodes; "
        f"mean return over last 100 episodes = "
        f"{sum(episode_returns[-100:]) / min(100, len(episode_returns)):.3f}"
    )

    plot_learning_curve(
        episode_returns,
        save_path=os.path.join(OUTPUT_DIR, "shaping_maze_q_learning_curve_.png"),
        title="Q-learning: episode return on shaping maze GridWorld",
    )

    ax = plot_policy(env, policy, title="Q-learning: greedy policy")
    ax.get_figure().savefig(os.path.join(OUTPUT_DIR, "shaping_maze_q_learning_policy.png"), dpi=150)
    plt.close(ax.get_figure())

    # Also save a Q-value grid (action-values in each state).
    ax = plot_q_values(env, Q, title="Q-learning: Q(s, a)")
    ax.get_figure().savefig(os.path.join(OUTPUT_DIR, "shaping_maze_q_learning_q_values.png"), dpi=150)
    plt.close(ax.get_figure())

    # Roll out one episode with the learned greedy policy and animate it.
    obs, info = env.reset(seed=0)
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action = int(policy[obs])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    print(f"Greedy rollout: steps={info['step']}, total_reward={total_reward:.3f}")

    print("shaping rewards in config:", env.config.rewards)
    print("trajectory rewards recorded:", len(env.trajectory["rewards"]))
    save_path = os.path.join(OUTPUT_DIR, "shaping_maze_q_learning_agent.gif")
    animate_episode(env, save_path=save_path, fps=2, show_policy=policy)
    print(f"Saved animation to {save_path}")

if __name__ == "__main__":
    main()
