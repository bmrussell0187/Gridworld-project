"""Configuration for the GridWorld environment.

A GridWorld is a classic way of representing a *finite Markov Decision
Process* (MDP). An MDP is defined by the tuple (S, A, P, R, gamma):

    S     - a finite set of states        -> here, grid cells (x, y)
    A     - a finite set of actions       -> here, {up, right, down, left}
    P     - transition probabilities      -> P(s' | s, a)
    R     - reward function               -> R(s, a, s')
    gamma - discount factor (set later, when solving the MDP)

This module defines ``GridWorldConfig``, a plain dataclass that fully
specifies one particular finite MDP instance: the size of the grid, where
the agent starts, which cells are terminal (absorbing) and what reward they
give, which cells are blocked, and how "slippery" the transitions are.

Everything a student needs to change to define a new MDP for their
dissertation experiments lives in this one file (or in ``examples.py``,
which builds ``GridWorldConfig`` objects for a few standard scenarios).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GridWorldConfig:
    """Full specification of a GridWorld finite MDP.

    Coordinates use ``(x, y)`` with ``x`` increasing to the right and ``y``
    increasing upwards, and ``(0, 0)`` at the bottom-left of the grid. This
    matches how the grid is drawn in ``plotting.py``.

    Attributes:
        width: Number of columns in the grid (state space size = width * height).
        height: Number of rows in the grid.
        start: The (x, y) cell the agent starts in at the beginning of each episode.
        terminal_states: Mapping from (x, y) cell to the reward received when
            the agent enters that cell. Entering a terminal state ends the
            episode (``terminated = True``) -- these are "absorbing states"
            in MDP terminology.
        rewards: Mapping from (x, y) cell to an *additional* reward received
            when the agent enters that (non-terminal) cell. Useful for
            reward shaping experiments.
        walls: Set of (x, y) cells the agent cannot enter. Attempting to
            move into a wall (or off the edge of the grid) leaves the agent
            in place and yields ``invalid_move_reward``.
        step_reward: Reward received on every transition, on top of any
            terminal/shaping reward. Typically small and negative to
            encourage reaching the goal quickly.
        invalid_move_reward: Reward for an attempted move into a wall or off
            the grid (the agent does not move).
        slip_probability: Probability that the environment ignores the
            agent's chosen action and instead executes a random *different*
            action. This is what makes the MDP's transition function
            P(s' | s, a) stochastic rather than deterministic -- with
            probability ``slip_probability`` the outcome is not the one the
            policy intended, a key idea in dynamic programming and RL.
        max_steps: Episode truncation limit (``truncated = True`` after this
            many steps if the agent has not reached a terminal state).
        seed: Optional default random seed for reproducibility.
    """

    width: int
    height: int
    start: tuple[int, int]
    terminal_states: dict[tuple[int, int], float]
    rewards: dict[tuple[int, int], float] = field(default_factory=dict)
    walls: set[tuple[int, int]] = field(default_factory=set)
    step_reward: float = -0.01
    invalid_move_reward: float = -0.05
    slip_probability: float = 0.0
    max_steps: int = 100
    seed: int | None = None

    def __post_init__(self) -> None:
        # Basic sanity checks so misconfigurations fail fast and loudly,
        # rather than producing a silently-broken MDP.
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive integers")

        if not self._in_bounds(self.start):
            raise ValueError(f"start {self.start} is outside the grid")

        if self.start in self.walls:
            raise ValueError(f"start {self.start} cannot be a wall")

        for cell in self.terminal_states:
            if not self._in_bounds(cell):
                raise ValueError(f"terminal state {cell} is outside the grid")
            if cell in self.walls:
                raise ValueError(f"terminal state {cell} cannot be a wall")

        for cell in self.rewards:
            if not self._in_bounds(cell):
                raise ValueError(f"reward cell {cell} is outside the grid")

        for cell in self.walls:
            if not self._in_bounds(cell):
                raise ValueError(f"wall cell {cell} is outside the grid")

        if not (0.0 <= self.slip_probability <= 1.0):
            raise ValueError("slip_probability must be in [0, 1]")

    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height
