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
* :func:`plot_mining_schedule` shows the MineWorld risk/reward gamble as a
  function of the attempt number m.

MineWorld support
-----------------
:class:`~gridworld.mining_env.MineWorldEnv` has an *augmented* state,
``(position, mining_counts)``, so a policy/value/Q array has one entry per
(cell, count-vector) pair rather than one per cell. Every function here that
indexes such an array therefore takes an optional ``mining_counts`` argument
selecting which slice of the augmented state space to draw (defaulting to the
all-zeros vector, i.e. "nothing mined yet"). Plain
:class:`~gridworld.env.GridWorldEnv` objects ignore the argument entirely.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: no display / X server required

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import numpy as np

# Arrow direction for each movement action, matching env.py's action encoding
# (0=up, 1=right, 2=down, 3=left) and the (x, y)-with-y-up convention. The
# MineWorld action 4 ("mine") does not move the agent, so it is drawn as a
# marker rather than an arrow (see _MINE_ACTION).
_ACTION_ARROWS = {
    0: (0.0, 0.32),   # up
    1: (0.32, 0.0),   # right
    2: (0.0, -0.32),  # down
    3: (-0.32, 0.0),  # left
}
_ACTION_LABELS = {0: "^", 1: ">", 2: "v", 3: "<", 4: "M"}

# MineWorld's fifth action, and how mining nodes are drawn.
_MINE_ACTION = 4
_MINING_FACE_COLOR = "#f2dfa0"     # sandy fill for an un-exhausted mining node
_MINING_MARKER_COLOR = "#7a5c12"   # dark gold for the node marker + count label
# Offset of the node marker/count label inside its cell. Placing them in the
# top-left corner keeps them readable even when the (centred) agent marker is
# standing on the node.
_MINING_MARKER_OFFSET = (-0.32, 0.32)
_MINING_LABEL_OFFSET = (-0.12, 0.32)


def _new_ax(ax: Axes | None, figsize=(6, 6)) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


# ----------------------------------------------------------------------
# GridWorld / MineWorld compatibility helpers
# ----------------------------------------------------------------------
def mining_nodes_of(env) -> tuple[tuple[int, int], ...]:
    """The ordered mining nodes of ``env``, or ``()`` for a plain GridWorld."""
    return tuple(getattr(env, "mining_nodes", ()))


def default_mining_counts(env) -> tuple[int, ...]:
    """The all-zeros count vector for ``env`` (empty for a plain GridWorld)."""
    return (0,) * len(mining_nodes_of(env))


def state_index_of(env, cell: tuple[int, int], mining_counts: tuple[int, ...] | None = None) -> int:
    """Flattened state index of ``cell`` for the given mining counts.

    Works for both environments: ``GridWorldEnv`` has one state per cell,
    while ``MineWorldEnv`` has one per (cell, count-vector) pair.
    """
    if hasattr(env, "components_to_state"):
        counts = default_mining_counts(env) if mining_counts is None else tuple(mining_counts)
        return env.components_to_state(cell, counts)
    return env.coord_to_state(cell)


def add_mining_count_labels(
    env,
    ax: Axes,
    mining_counts: tuple[int, ...] | None = None,
    show_counts: bool = True,
) -> dict[tuple[int, int], object]:
    """Add the attempt-count label next to every mining node.

    Returns the text artists keyed by node, so ``animation.py`` can update
    them frame by frame instead of redrawing the whole grid.
    """
    labels: dict[tuple[int, int], object] = {}
    nodes = mining_nodes_of(env)
    if not nodes:
        return labels

    counts = default_mining_counts(env) if mining_counts is None else tuple(mining_counts)
    label_dx, label_dy = _MINING_LABEL_OFFSET
    for index, (x, y) in enumerate(nodes):
        labels[(x, y)] = ax.text(
            x + label_dx, y + label_dy, str(counts[index]) if show_counts else "",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color=_MINING_MARKER_COLOR, zorder=7,
        )
    return labels


def _draw_mining_nodes(
    env, ax: Axes, mining_counts: tuple[int, ...] | None, show_counts: bool
) -> dict[tuple[int, int], object]:
    """Draw a diamond marker (and optionally an attempt count) on every node."""
    nodes = mining_nodes_of(env)
    if not nodes:
        return {}

    marker_dx, marker_dy = _MINING_MARKER_OFFSET
    for x, y in nodes:
        ax.plot(
            [x + marker_dx], [y + marker_dy], marker="D", markersize=7,
            color=_MINING_MARKER_COLOR, linestyle="none", zorder=7,
        )
    return add_mining_count_labels(env, ax, mining_counts, show_counts=show_counts)


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
    mining_counts: tuple[int, ...] | None = None,
    show_mining_counts: bool = True,
) -> Axes:
    """Draw the static layout of a GridWorld: walls, start, terminals, rewards.

    This depicts the state space S and reward function R of the finite MDP.

    For a :class:`~gridworld.mining_env.MineWorldEnv`, mining nodes are also
    shaded and marked with a diamond. ``mining_counts`` supplies the attempt
    count to print beside each node (defaulting to all zeros); pass
    ``show_mining_counts=False`` to draw the markers without labels, which is
    what ``animation.py`` does before adding its own per-frame labels.
    """
    ax = _new_ax(ax)
    cfg = env.config
    _setup_grid_axes(env, ax)
    mining_nodes = set(mining_nodes_of(env))

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
            elif cell in mining_nodes:
                # Mining nodes have no fixed reward to print: their payoff
                # depends on the attempt number, drawn as a count instead.
                face_color = _MINING_FACE_COLOR
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

    _draw_mining_nodes(env, ax, mining_counts, show_counts=show_mining_counts)

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
    mining_counts: tuple[int, ...] | None = None,
    show_mining_counts: bool = True,
) -> Axes:
    """Draw arrows showing the greedy action pi(s) chosen in each state.

    ``policy`` is an array of shape (n_states,) mapping a state index to an
    action. Terminal and wall cells are skipped since no action is ever taken
    from them. For MineWorld, the ``MINE`` action is drawn as a cross marker
    (the agent stays put), and ``mining_counts`` chooses which slice of the
    augmented state space is shown.
    """
    ax = plot_gridworld(
        env, ax=ax, show_rewards=show_rewards, title=title,
        mining_counts=mining_counts, show_mining_counts=show_mining_counts,
    )
    cfg = env.config

    for x in range(env.width):
        for y in range(env.height):
            cell = (x, y)
            if cell in env.walls or cell in cfg.terminal_states:
                continue
            action = int(policy[state_index_of(env, cell, mining_counts)])
            if action == _MINE_ACTION:
                ax.plot(
                    [x], [y], marker="X", markersize=13, color="tab:purple",
                    linestyle="none", zorder=5,
                )
                continue
            dx, dy = _ACTION_ARROWS[action]
            ax.annotate(
                "", xy=(x + dx, y + dy), xytext=cell,
                arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.8),
            )
    return ax


def plot_value_function(
    env,
    V: np.ndarray,
    ax: Axes | None = None,
    title: str | None = None,
    mining_counts: tuple[int, ...] | None = None,
) -> Axes:
    """Draw the state-value function V(s) as a heatmap with numeric labels.

    V represents the expected discounted return from each state, i.e. the
    solution to the Bellman equation V(s) = max_a sum_s' P(s'|s,a)[R + gamma*V(s')].
    For MineWorld, ``mining_counts`` selects which slice of the augmented
    state space to show.
    """
    ax = _new_ax(ax)
    _setup_grid_axes(env, ax)

    grid = np.full((env.height, env.width), np.nan)
    for x in range(env.width):
        for y in range(env.height):
            grid[y, x] = V[state_index_of(env, (x, y), mining_counts)]

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

    _draw_mining_nodes(env, ax, mining_counts, show_counts=True)

    fig = ax.get_figure()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="V(s)")
    if title:
        ax.set_title(title)
    return ax


def plot_q_values(
    env,
    Q: np.ndarray,
    ax: Axes | None = None,
    title: str | None = None,
    mining_counts: tuple[int, ...] | None = None,
) -> Axes:
    """Show Q(s, a) for every action, as small directional labels within each cell.

    Q is shape (n_states, n_actions). In each non-wall cell we print the value
    of each action next to a glyph for that action (an arrow for the four
    moves, ``M`` in the centre for MineWorld's ``MINE``). The best action's
    text is bolded. ``mining_counts`` selects which slice of a MineWorld
    augmented state space to show.
    """
    ax = plot_gridworld(
        env, ax=ax, show_rewards=False, title=title, mining_counts=mining_counts
    )
    cfg = env.config

    offsets = {
        0: (0.0, 0.30),    # up
        1: (0.30, 0.05),   # right
        2: (0.0, -0.30),   # down
        3: (-0.30, 0.05),  # left
        4: (0.0, -0.08),   # mine (centre: it does not move the agent)
    }
    n_actions = Q.shape[1]
    for x in range(env.width):
        for y in range(env.height):
            cell = (x, y)
            if cell in env.walls or cell in cfg.terminal_states:
                continue
            q_row = Q[state_index_of(env, cell, mining_counts)]
            best_action = int(np.argmax(q_row))
            for action in range(n_actions):
                dx, dy = offsets[action]
                weight = "bold" if action == best_action else "normal"
                color = "tab:red" if action == best_action else "black"
                ax.text(
                    x + dx, y + dy,
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


def plot_mining_schedule(
    config,
    ax: Axes | None = None,
    title: str = "Mining gamble by attempt number",
) -> Axes:
    """Plot the MineWorld risk/reward trade-off against the attempt number m.

    Shows the success probability p(m) on the left axis and, on the right
    axis, the expected positive payout E[R+(m)], the failure penalty, and the
    overall expected value of one mining attempt. Where the expected value
    crosses zero is exactly where "mine again" stops being a free lunch --
    the decision the agent has to learn.

    ``config`` is a :class:`~gridworld.mining_config.MineWorldConfig`.
    """
    ax = _new_ax(ax, figsize=(7, 4))
    attempts = list(range(1, config.max_mining_count + 2))

    probabilities = [config.positive_reward_probability(m) for m in attempts]
    ax.plot(attempts, probabilities, marker="o", color="tab:blue", label="p(m) = P(success)")
    ax.set_xlabel("attempt number m at a single node")
    ax.set_ylabel("probability", color="tab:blue")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(attempts)
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(True, alpha=0.3)

    reward_ax = ax.twinx()
    reward_ax.plot(
        attempts, [config.expected_positive_reward(m) for m in attempts],
        marker="s", color="tab:green", label="E[R+(m)] if successful",
    )
    reward_ax.plot(
        attempts, [config.mining_failure_reward_at(m) for m in attempts],
        marker="v", color="tab:red", label="penalty if the mine collapses",
    )
    reward_ax.plot(
        attempts, [config.expected_mining_reward(m) for m in attempts],
        marker="D", color="black", linestyle="--", label="E[reward] of the attempt",
    )
    reward_ax.axhline(0.0, color="gray", linewidth=0.8)
    reward_ax.set_ylabel("reward")

    handles, labels = ax.get_legend_handles_labels()
    reward_handles, reward_labels = reward_ax.get_legend_handles_labels()
    ax.legend(handles + reward_handles, labels + reward_labels, fontsize=8, loc="center left")

    # The last attempt number is the *saturated* level: counts stop growing at
    # max_mining_count, so every further attempt is evaluated there.
    ax.set_title(f"{title}\n(m={attempts[-1]} repeats forever once the cap is reached)")
    return ax
