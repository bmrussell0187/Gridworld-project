"""A handful of ready-made GridWorld MDPs for dissertation experiments.

Each ``make_*`` function returns a fully-constructed :class:`GridWorldEnv`.
They are deliberately small and easy to reason about by hand, so that a
student can verify dynamic-programming / Q-learning results against
intuition (or against a hand-drawn transition diagram).

Grid coordinate convention: (x, y), with (0, 0) at the bottom-left and y
increasing upwards -- so "top-right" means large x, large y.
"""

from __future__ import annotations

from .config import GridWorldConfig
from .env import GridWorldEnv


def make_easy_gridworld(seed: int | None = 0) -> GridWorldEnv:
    """Example 1: small 5x5 deterministic gridworld.

    - Start: bottom-left, (0, 0).
    - Goal: top-right, (4, 4), reward +1.
    - Trap: bottom-right, (4, 0), reward -1.
    - Deterministic transitions (slip_probability = 0).

    This is the simplest possible non-trivial finite MDP: 25 states, 4
    actions, no stochasticity, one good absorbing state and one bad one.
    A good first target for value iteration / policy iteration by hand.
    """
    config = GridWorldConfig(
        width=5,
        height=5,
        start=(0, 0),
        terminal_states={(4, 4): 1.0, (4, 0): -1.0},
        rewards={},
        walls=set(),
        step_reward=-0.01,
        invalid_move_reward=-0.05,
        slip_probability=0.0,
        max_steps=100,
        seed=seed,
    )
    return GridWorldEnv(config)


def make_walled_gridworld(seed: int | None = 0) -> GridWorldEnv:
    """Example 2: 6x6 gridworld with obstacles separating start and goal.

    The agent must navigate around a wall of blocked cells to reach the
    goal, so the optimal policy is no longer a straight line -- useful for
    illustrating that value iteration correctly handles non-trivial
    topology / reachability.
    """
    walls = {(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)}  # a vertical wall with a gap at y=5
    config = GridWorldConfig(
        width=6,
        height=6,
        start=(0, 0),
        terminal_states={(5, 5): 1.0, (5, 0): -1.0},
        rewards={},
        walls=walls,
        step_reward=-0.01,
        invalid_move_reward=-0.05,
        slip_probability=0.0,
        max_steps=150,
        seed=seed,
    )
    return GridWorldEnv(config)


def make_slippery_gridworld(seed: int | None = 0) -> GridWorldEnv:
    """Example 3: the easy gridworld made stochastic (slip_probability=0.2).

    Identical layout to :func:`make_easy_gridworld`, but 20% of the time
    the agent's action is replaced by a random different one. This is
    useful for showing how introducing transition stochasticity changes
    the optimal policy (e.g. the agent may prefer to stay further from the
    trap even if that path is nominally longer), and for contrasting
    model-based dynamic programming (which uses the true P(s'|s,a)) with
    model-free Q-learning (which must estimate its effects from samples).
    """
    config = GridWorldConfig(
        width=5,
        height=5,
        start=(0, 0),
        terminal_states={(4, 4): 1.0, (4, 0): -1.0},
        rewards={},
        walls=set(),
        step_reward=-0.01,
        invalid_move_reward=-0.05,
        slip_probability=0.2,
        max_steps=100,
        seed=seed,
    )
    return GridWorldEnv(config)


def make_reward_shaping_gridworld(seed: int | None = 0) -> GridWorldEnv:
    """Example 4: reward shaping via intermediate reward cells.

    Same 5x5 layout and goal/trap as :func:`make_easy_gridworld`, but with
    small positive "breadcrumb" rewards placed along a corridor toward the
    goal, plus a small negative reward in an area the designer wants the
    agent to avoid. Useful for studying how shaping rewards can speed up
    learning but also risk changing the optimal policy if not
    potential-based (a classic RL discussion point).
    """
    shaping_rewards = {
        (1, 1): 0.02,
        (2, 2): 0.05,
        (3, 3): 0.08,
        (1, 3): -0.05,  # a mildly discouraged detour cell
    }
    config = GridWorldConfig(
        width=5,
        height=5,
        start=(0, 0),
        terminal_states={(4, 4): 1.0, (4, 0): -1.0},
        rewards=shaping_rewards,
        walls=set(),
        step_reward=-0.01,
        invalid_move_reward=-0.05,
        slip_probability=0.0,
        max_steps=100,
        seed=seed,
    )
    return GridWorldEnv(config)


def make_maze_gridworld(seed: int | None = 0) -> GridWorldEnv:
    walls: set[tuple[int, int]] = set()
    walls |= {(0, y) for y in (3, 4, 5)}
    walls |= {(1, y) for y in (3, 8)}
    walls |= {(2, y) for y in (1, 3, 5, 6, 7, 8)}
    walls |= {(3, y) for y in (1, 5)}
    walls |= {(4, y) for y in (1, 2, 3, 4, 5, 7, 9)}
    walls |= {(5, y) for y in (1, 4, 5, 7, 9)}
    walls |= {(6, y) for y in (1, 2, 4, 7)}
    walls |= {(7, y) for y in (1, 4, 6)}
    walls |= {(8, y) for y in (3, 4, 6, 8)}
    walls |= {(9, y) for y in (0, 1, 2, 3, 6)}

    shaping_rewards = {
        (1, 5): 0.03,
        (3, 0): 0.05,
        (8, 0): 0.08,
    }

    config = GridWorldConfig(
        width=10,
        height=10,
        start=(0, 9),
        terminal_states={
            (1, 1): -1.0,
            (1, 7): -1.0,
            (5, 2): 10.0,
            (7, 7): 1.0,
            (7, 8): -1.0,
            (9, 4): -4.0,
        },
        rewards=shaping_rewards,
        walls=walls,
        step_reward=-0.01,
        invalid_move_reward=-0.05,
        slip_probability=0.0,
        max_steps=200,
        seed=seed,
    )

    return GridWorldEnv(config)