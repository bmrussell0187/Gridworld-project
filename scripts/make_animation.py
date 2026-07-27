"""Minimal example: animate an arbitrary trajectory on a GridWorld.

This shows the smallest possible use of ``animate_episode`` -- run any
policy for one episode (here, a fixed sequence of actions is used just to
illustrate the API) and turn the resulting trajectory into a GIF.

You can adapt this script to animate a trajectory produced anywhere else in
the project, by passing it explicitly via the ``trajectory=`` argument
instead of relying on ``env.trajectory``.

Run from the repository root:
    python scripts/make_animation.py
"""

from __future__ import annotations

import os

from gridworld.examples import make_easy_gridworld
from gridworld.animation import animate_episode

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = make_easy_gridworld(seed=0)

    # Run one short, fixed episode: right, right, right, right, up, up, up, up
    # (a direct diagonal path from (0,0) to the goal at (4,4)).
    obs, info = env.reset(seed=0)
    actions = [1, 1, 1, 1, 0, 0, 0, 0]
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # env.trajectory now holds the recorded episode; animate_episode uses
    # it automatically when no `trajectory=` argument is given.
    save_path = os.path.join(OUTPUT_DIR, "example_trajectory.gif")
    animate_episode(env, save_path=save_path, fps=2)
    print(f"Saved animation to {save_path}")


if __name__ == "__main__":
    main()
