"""Static plotting utilities for GridWorld: layout, policies, and values.

All functions here operate on a headless ("Agg") matplotlib backend so they
work on servers / CI without a display, and every ``plot_*`` function
accepts an optional ``ax`` so plots can be composed into subplot grids (or
reused frame-by-frame by ``animation.py``).

These plots are the main way a dissertation reader "sees" the MDP:

* :func:`plot_gridworld` shows the state space S itself (cells), the
  reward function R (numbers), and terminal/wall structure.
* :func:`plot_policy` visualises a policy pi(s) -> a as arrows: this is
  what value iteration / policy iteration / Q-learning are trying to find.
* :func:`plot_value_function` visualises V(s), the expected discounted
  return from each state under a policy -- the central object in the
  Bellman equations.
* :func:`plot_q_values` visualises Q(s, a), the action-value function used
  directly by Q-learning and epsilon-greedy action selection.
* :func:`plot_learning_curve` shows how episode return evolves during
  training, the standard empirical evidence of learning progress.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: no display / X server required

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np

# Arrow direction for each action, matching env.py's action encoding
# (0=up, 1=right, 2=down, 3=left) and the (x, y)-with-y-up convention.
_ACTION_ARROWS = {
    0: (0.0, 0.32),   # up
    1: (0.32, 0.0),   # right
    2: (0.0, -0.32),  # down
    3: (-0.32, 0.0),  # left
}
_ACTION_LABELS = {0: "^", 1: ">", 2: "v", 3: "<"}


def _new_ax(ax: Axes | None, figsize=(6, 6)) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _setup_grid_axes(env, ax: Axes) -> None:
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(-0.5, env.height - 0.5)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, which="major", color="gray", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)


def plot_gridworld(
    env,
    ax: Axes | None = None,
    show_rewards: bool = True,
    show_start: bool = True,
    show_terminals: bool = True,
    show_walls: bool = True,
    title: str | None = None,
) -> Axes:
    """Draw the static layout of a GridWorld: walls, start, terminals, rewards.

    This depicts the state space S and reward function R of the finite MDP.
    """
    ax = _new_ax(ax)
    cfg = env.config
    _setup_grid_axes(env, ax)

    for x in range(env.width):
        for y in range(env.height):
            cell = (x, y)
            face_color = "white"
            edge_color = "black"
            text = None
            text_color = "black"

            if show_walls and cell in env.walls:
                face_color = "dimgray"
            elif show_terminals and cell in cfg.terminal_states:
                reward = cfg.terminal_states[cell]
                face_color = "#8fd19e" if reward > 0 else "#e28b8b"
                if show_rewards:
                    text = f"{reward:+.2f}"
            elif cell in cfg.rewards:
                reward = cfg.rewards[cell]
                face_color = "#cfe8ff" if reward >= 0 else "#ffe0b3"
                if show_rewards:
                    text = f"{reward:+.2f}"

            ax.add_patch(
                Rectangle(
                    (x - 0.5, y - 0.5), 1.0, 1.0,
                    facecolor=face_color, edgecolor=edge_color, linewidth=0.8,
                )
            )
            if text is not None:
                ax.text(x, y, text, ha="center", va="center", fontsize=9, color=text_color)

    if show_start:
        sx, sy = cfg.start
        ax.text(
            sx, sy, "S", ha="center", va="center", fontsize=14,
            fontweight="bold", color="tab:blue",
        )

    if title:
        ax.set_title(title)
    return ax


def plot_policy(
    env,
    policy: np.ndarray,
    ax: Axes | None = None,
    title: str | None = None,
    show_rewards: bool = True,
) -> Axes:
    """Draw arrows showing the greedy action pi(s) chosen in each state.

    ``policy`` is an array of shape (n_states,) mapping a state index to an
    action in {0, 1, 2, 3}. Terminal and wall cells are skipped since no
    action is ever taken from them.
    """
    ax = plot_gridworld(env, ax=ax, show_rewards=show_rewards, title=title)
    cfg = env.config

    for state in range(env.n_states):
        cell = env.state_to_coord(state)
        if cell in env.walls or cell in cfg.terminal_states:
            continue
        action = int(policy[state])
        dx, dy = _ACTION_ARROWS[action]
        ax.annotate(
            "", xy=(cell[0] + dx, cell[1] + dy), xytext=cell,
            arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.8),
        )
    return ax


def plot_value_function(env, V: np.ndarray, ax: Axes | None = None, title: str | None = None) -> Axes:
    """Draw the state-value function V(s) as a heatmap with numeric labels.

    V represents the expected discounted return from each state, i.e. the
    solution to the Bellman equation V(s) = max_a sum_s' P(s'|s,a)[R + gamma*V(s')].
    """
    ax = _new_ax(ax)
    _setup_grid_axes(env, ax)

    grid = np.full((env.height, env.width), np.nan)
    for state in range(env.n_states):
        x, y = env.state_to_coord(state)
        grid[y, x] = V[state]

    im = ax.imshow(
        grid, origin="lower", extent=(-0.5, env.width - 0.5, -0.5, env.height - 0.5),
        cmap="viridis",
    )
    for x in range(env.width):
        for y in range(env.height):
            cell = (x, y)
            if cell in env.walls:
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1.0, 1.0, facecolor="dimgray"))
                continue
            value = grid[y, x]
            ax.text(
                x, y, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color="white" if value < np.nanmedian(grid) else "black",
            )

    fig = ax.get_figure()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V(s)")
    if title:
        ax.set_title(title)
    return ax


def plot_q_values(env, Q: np.ndarray, ax: Axes | None = None, title: str | None = None) -> Axes:
    """Show Q(s, a) for every action, as small directional labels within each cell.

    Q is shape (n_states, 4). In each non-wall cell we print the value of
    each action next to an arrow glyph pointing in that action's direction,
    stacked vertically for readability. The best action's text is bolded.
    """
    ax = plot_gridworld(env, ax=ax, show_rewards=False, title=title)
    cfg = env.config

    offsets = {0: (0, 0.30), 1: (0.30, 0.05), 2: (0, -0.30), 3: (-0.30, 0.05)}
    for state in range(env.n_states):
        cell = env.state_to_coord(state)
        if cell in env.walls or cell in cfg.terminal_states:
            continue
        q_row = Q[state]
        best_action = int(np.argmax(q_row))
        for action in range(4):
            dx, dy = offsets[action]
            weight = "bold" if action == best_action else "normal"
            color = "tab:red" if action == best_action else "black"
            ax.text(
                cell[0] + dx, cell[1] + dy,
                f"{_ACTION_LABELS[action]}{q_row[action]:.2f}",
                ha="center", va="center", fontsize=6, fontweight=weight, color=color,
            )
    return ax


def plot_learning_curve(
    episode_returns: list[float] | np.ndarray,
    save_path: str | None = None,
    window: int = 20,
    title: str = "Learning curve",
) -> Axes:
    """Plot episode return vs. episode number, with a moving-average overlay.

    This is the standard empirical curve used to show that a model-free
    method (e.g. Q-learning) is learning: as more episodes are experienced,
    the epsilon-greedy policy should collect increasing return.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    episode_returns = np.asarray(episode_returns, dtype=float)
    episodes = np.arange(1, len(episode_returns) + 1)

    ax.plot(episodes, episode_returns, color="lightgray", linewidth=1, label="episode return")
    if len(episode_returns) >= window:
        moving_avg = np.convolve(episode_returns, np.ones(window) / window, mode="valid")
        ax.plot(
            episodes[window - 1:], moving_avg, color="tab:blue", linewidth=2,
            label=f"{window}-episode moving average",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return ax
