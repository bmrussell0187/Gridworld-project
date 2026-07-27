"""Train a Stable-Baselines3 DQN agent on the GridWorld environment.

This script demonstrates that GridWorldEnv is a fully-conformant Gymnasium
environment by (1) running it through Stable-Baselines3's ``check_env``
sanity checker, and (2) training a standard deep-RL algorithm (DQN) on it.

This is *optional* scaffolding for students who want to compare tabular
methods (value iteration, policy iteration, Q-learning) against a
function-approximation-based deep RL algorithm. The rest of the
dissertation project does not depend on this script or on Stable-Baselines3
being installed correctly/at all.

Run from the repository root:
    python scripts/run_sb3_dqn.py
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from gridworld.examples import make_easy_gridworld
from gridworld.plotting import plot_policy
from gridworld.animation import animate_episode

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = make_easy_gridworld(seed=0)

    # 1. Verify Gymnasium/SB3 compatibility.
    from stable_baselines3.common.env_checker import check_env

    check_env(env, warn=True)
    print("check_env passed: GridWorldEnv is Stable-Baselines3 compatible.")

    # 2. Train a DQN agent.
    from stable_baselines3 import DQN

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=20_000,
        learning_starts=500,
        batch_size=64,
        train_freq=4,
        target_update_interval=250,
        exploration_fraction=0.5,
        exploration_final_eps=0.05,
        gamma=0.95,
        seed=0,
        verbose=1,
    )
    model.learn(total_timesteps=30_000)

    # 3. Run one episode with the trained (greedy) policy.
    obs, info = env.reset(seed=0)
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
    print(f"Trained DQN rollout: steps={info['step']}, total_reward={total_reward:.3f}")

    # Derive a full state->action policy table (for plotting only) by
    # querying the trained model at every state.
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        action, _state = model.predict(s, deterministic=True)
        policy[s] = int(action)
    ax = plot_policy(env, policy, title="Stable-Baselines3 DQN: learned policy")
    ax.get_figure().savefig(os.path.join(OUTPUT_DIR, "sb3_dqn_policy.png"), dpi=150)
    plt.close(ax.get_figure())

    # 4. Save an animation of the trained agent's episode.
    save_path = os.path.join(OUTPUT_DIR, "sb3_dqn_agent.gif")
    animate_episode(env, save_path=save_path, fps=2, show_policy=policy)
    print(f"Saved animation to {save_path}")


if __name__ == "__main__":
    main()
