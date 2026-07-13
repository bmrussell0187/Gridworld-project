"""Animate a recorded GridWorld episode as a GIF or MP4.

An episode trajectory (a sequence of states, actions, rewards visited under
some policy) is the basic unit of empirical evidence in reinforcement
learning. This module turns a trajectory into a short video so that a
dissertation can *show* how a random policy, a value-iteration-derived
policy, or a learned Q-learning policy actually behaves, frame by frame.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless: no display / X server required

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np

from .env import ACTION_NAMES
from .plotting import plot_gridworld, plot_policy


def animate_episode(
    env,
    trajectory: dict | None = None,
    save_path: str = "outputs/gridworld_episode.gif",
    fps: int = 2,
    output_format: str | None = None,
    show_rewards: bool = True,
    show_policy: np.ndarray | None = None,
) -> str:
    """Render a recorded episode trajectory as an animation and save it.

    Parameters
    ----------
    env:
        The GridWorldEnv the trajectory was recorded on (used for the
        static background: layout, walls, rewards, terminals).
    trajectory:
        A dict with keys "positions", "actions", "rewards" (as produced by
        ``env.trajectory``). If ``None``, the environment's own most
        recently recorded trajectory (``env.trajectory``) is used.
    save_path:
        Output file path. The extension (``.gif`` or ``.mp4``) determines
        the writer used, unless overridden by ``output_format``.
    fps:
        Frames per second of the output animation.
    output_format:
        Optional explicit override ("gif" or "mp4") ignoring the file
        extension.
    show_rewards:
        Whether to annotate reward values on the background grid.
    show_policy:
        Optional policy array (n_states,) to overlay as static arrows,
        e.g. to show the trajectory alongside the policy that produced it.

    Returns
    -------
    The path the animation was saved to.
    """
    trajectory = trajectory if trajectory is not None else env.trajectory
    positions = trajectory["positions"]
    actions = trajectory.get("actions", [])
    rewards = trajectory.get("rewards", [])

    if len(positions) == 0:
        raise ValueError("Trajectory is empty; run an episode before animating it.")

    n_frames = len(positions)
    # cumulative[i] = total reward accumulated by the time we reach positions[i]
    cumulative = np.concatenate([[0.0], np.cumsum(rewards)]) if rewards else np.zeros(n_frames)

    fig, ax = plt.subplots(figsize=(6, 6))
    if show_policy is not None:
        plot_policy(env, show_policy, ax=ax, show_rewards=show_rewards)
    else:
        plot_gridworld(env, ax=ax, show_rewards=show_rewards)

    agent_marker, = ax.plot([], [], marker="o", markersize=22, color="tab:blue", zorder=6)
    path_line, = ax.plot(
        [], [], marker="o", markersize=6, color="tab:blue", alpha=0.35,
        linestyle="-", linewidth=1.5, zorder=4,
    )
    title = ax.set_title("t=0")

    def init():
        agent_marker.set_data([], [])
        path_line.set_data([], [])
        title.set_text("t=0")
        return agent_marker, path_line, title

    def update(frame: int):
        pos = positions[frame]
        agent_marker.set_data([pos[0]], [pos[1]])

        xs = [p[0] for p in positions[: frame + 1]]
        ys = [p[1] for p in positions[: frame + 1]]
        path_line.set_data(xs, ys)

        if frame == 0:
            title.set_text(f"t=0 | start at {pos} | cumulative reward=0.00")
        else:
            action_name = ACTION_NAMES[actions[frame - 1]]
            reward = rewards[frame - 1]
            title.set_text(
                f"t={frame} | action={action_name} | reward={reward:+.2f} "
                f"| cumulative reward={cumulative[frame]:+.2f}"
            )
        return agent_marker, path_line, title

    anim = mpl_animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init, blit=False,
        interval=1000.0 / fps,
    )

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fmt = (output_format or os.path.splitext(save_path)[1].lstrip(".")).lower()

    if fmt == "gif":
        writer = mpl_animation.PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
    elif fmt in ("mp4", "mov"):
        if not mpl_animation.writers.is_available("ffmpeg"):
            plt.close(fig)
            print(
                "[animate_episode] ffmpeg was not found on PATH, so an MP4 "
                "cannot be written.\n"
                "  Install ffmpeg (e.g. `brew install ffmpeg` or `apt install "
                "ffmpeg`), or call animate_episode(..., save_path='...gif') "
                "to use the GIF writer instead (no external dependency)."
            )
            raise RuntimeError("ffmpeg is not available for MP4 export")
        writer = mpl_animation.FFMpegWriter(fps=fps)
        anim.save(save_path, writer=writer)
    else:
        plt.close(fig)
        raise ValueError(f"Unsupported animation format '{fmt}'; use 'gif' or 'mp4'.")

    plt.close(fig)
    return save_path
