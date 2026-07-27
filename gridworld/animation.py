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
from .plotting import (
    add_mining_count_labels,
    default_mining_counts,
    mining_nodes_of,
    plot_gridworld,
    plot_policy,
)


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
        recently recorded trajectory (``env.trajectory``) is used. A
        MineWorld trajectory additionally carries "mining_counts" (one
        immutable snapshot per frame) and "infos", which are used to animate
        the per-node attempt counts and to flag a mine collapse.
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
    infos = trajectory.get("infos", [])

    if len(positions) == 0:
        raise ValueError("Trajectory is empty; run an episode before animating it.")

    n_frames = len(positions)
    # cumulative[i] = total reward accumulated by the time we reach positions[i]
    cumulative = np.concatenate([[0.0], np.cumsum(rewards)]) if rewards else np.zeros(n_frames)

    # MineWorld only: one immutable count snapshot per frame, so each frame
    # shows the counts as they were *then* rather than the final totals.
    mining_nodes = mining_nodes_of(env)
    mining_counts = trajectory.get("mining_counts") or []
    if mining_nodes and len(mining_counts) != n_frames:
        raise ValueError(
            "MineWorld trajectory must contain one 'mining_counts' snapshot per "
            f"frame; got {len(mining_counts)} snapshots for {n_frames} positions."
        )
    # Action names: MineWorld adds a fifth action, so prefer the env's own map.
    action_names = getattr(env, "action_names", ACTION_NAMES)

    fig, ax = plt.subplots(figsize=(6, 6))
    if show_policy is not None:
        plot_policy(
            env, show_policy, ax=ax, show_rewards=show_rewards, show_mining_counts=False
        )
    else:
        # Draw the node markers but not the counts: the counts are dynamic and
        # get their own text artists, updated frame by frame below.
        plot_gridworld(env, ax=ax, show_rewards=show_rewards, show_mining_counts=False)
    count_labels = add_mining_count_labels(env, ax, show_counts=False)

    agent_marker, = ax.plot([], [], marker="o", markersize=22, color="tab:blue", zorder=6)
    path_line, = ax.plot(
        [], [], marker="o", markersize=6, color="tab:blue", alpha=0.35,
        linestyle="-", linewidth=1.5, zorder=4,
    )
    title = ax.set_title("t=0")

    def set_counts(frame: int) -> None:
        if not count_labels:
            return
        counts = mining_counts[frame] if mining_counts else default_mining_counts(env)
        for node, label in count_labels.items():
            label.set_text(str(counts[mining_nodes.index(node)]))

    def init():
        agent_marker.set_data([], [])
        agent_marker.set_color("tab:blue")
        path_line.set_data([], [])
        set_counts(0)
        title.set_text("t=0")
        return agent_marker, path_line, title

    def update(frame: int):
        pos = positions[frame]
        agent_marker.set_data([pos[0]], [pos[1]])

        xs = [p[0] for p in positions[: frame + 1]]
        ys = [p[1] for p in positions[: frame + 1]]
        path_line.set_data(xs, ys)
        set_counts(frame)

        if frame == 0:
            agent_marker.set_color("tab:blue")
            title.set_text(f"t=0 | start at {pos} | cumulative reward=0.00")
            return agent_marker, path_line, title

        action_name = action_names[actions[frame - 1]]
        reward = rewards[frame - 1]
        info = infos[frame - 1] if frame - 1 < len(infos) else {}
        text = (
            f"t={frame} | action={action_name} | reward={reward:+.2f} "
            f"| cumulative reward={cumulative[frame]:+.2f}"
        )
        if info.get("mining_failure"):
            # The episode ended because the mine collapsed: make that
            # unmistakable in the final frame.
            agent_marker.set_color("tab:red")
            text += f"\nMINE COLLAPSED at {info.get('mining_node')} -- episode over"
        else:
            agent_marker.set_color("tab:blue")
            if info.get("mining_success"):
                text += f"\nmined {info.get('mining_node')} (attempt " \
                        f"{info.get('mining_attempt_number')})"
        title.set_text(text)
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
