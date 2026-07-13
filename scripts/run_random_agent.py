"""Run a uniformly random policy for one episode and animate it.

This is the simplest possible baseline: at each state, pick an action from
{up, right, down, left} uniformly at random. It is useful as a sanity check
(does the environment run at all?) and as a point of comparison against the
value-iteration / Q-learning policies produced by the other scripts.

Run from the repository root:
    python scripts/run_random_agent.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gridworld.examples import make_easy_gridworld
from gridworld.animation import animate_episode

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = make_easy_gridworld(seed=0)

    obs, info = env.reset(seed=0)
    terminated = truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(f"Random agent finished episode: steps={info['step']}, total_reward={total_reward:.3f}")

    save_path = os.path.join(OUTPUT_DIR, "random_agent.gif")
    animate_episode(env, save_path=save_path, fps=2)
    print(f"Saved animation to {save_path}")


if __name__ == "__main__":
    main()
