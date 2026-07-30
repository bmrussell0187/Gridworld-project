"""Train tabular Q-learning on the slippery GridWorld and visualise the results.

Q-learning is *model-free*: unlike value_iteration/policy_iteration, it
never sees the transition probabilities directly -- it learns Q(s, a)
purely from sampled (state, action, reward, next_state) experience while
following an epsilon-greedy behaviour policy. Running it on the slippery
(stochastic) gridworld shows that model-free learning still converges to a
sensible policy even when transitions are noisy.

Run from the repository root:
    python scripts/run_q_learning.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

from gridworld.maze_gridworld import make_maze_gridworld
from gridworld.q_learning import q_learning
from gridworld.plotting import plot_q_values, plot_policy, plot_learning_curve
from gridworld.animation import animate_episode

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = make_maze_gridworld(seed=0) #make_slippery_gridworld(seed=0)

    Q, policy, episode_returns, episode_lengths = q_learning(
        env,
        episodes=100000,
        alpha=0.2,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9999,
        seed=0,
    )

    print("")
    print(
        f"Q-learning finished {len(episode_returns)} episodes; "
        f"mean return over last 100 episodes = "
        f"{sum(episode_returns[-100:]) / min(100, len(episode_returns)):.3f}"
    )

    plot_learning_curve(
        episode_returns,
        save_path=os.path.join(OUTPUT_DIR, "q_learning_curve.png"),
        title="Q-learning: episode return on maze GridWorld",
    )

    ax = plot_policy(env, policy, title="Q-learning: greedy policy")
    ax.get_figure().savefig(os.path.join(OUTPUT_DIR, "q_learning_policy.png"), dpi=150)
    plt.close(ax.get_figure())

    # Also save a Q-value grid (action-values in each state).
    ax = plot_q_values(env, Q, title="Q-learning: Q(s, a)")
    ax.get_figure().savefig(os.path.join(OUTPUT_DIR, "q_learning_q_values.png"), dpi=150)
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
    save_path = os.path.join(OUTPUT_DIR, "q_learning_agent.gif")
    animate_episode(env, save_path=save_path, fps=2, show_policy=policy)
    print(f"Saved animation to {save_path}")
    print("config start:", env.config.start)
    print("env start:   ", env.state_to_coord(env.reset()[0]))
    print("loaded from: ", type(env).__module__)

if __name__ == "__main__":
    main()

