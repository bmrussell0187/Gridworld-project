"""GridWorldEnv: a configurable Gymnasium environment for a finite MDP.

Relation to finite MDP theory
------------------------------
An episodic finite MDP is the tuple (S, A, P, R, gamma). This environment
implements exactly that:

* **State space S**: every non-wall cell (x, y) of the grid, flattened to a
  single integer via ``coord_to_state``. ``observation_space`` is
  ``Discrete(width * height)`` so it is directly compatible with tabular
  dynamic-programming/Q-learning code *and* with Stable-Baselines3.
* **Action space A**: ``Discrete(4)`` = {up, right, down, left}.
* **Transition function P(s' | s, a)**: implemented twice, in two forms that
  must always agree:
    - imperatively, in :meth:`step` (used for actually running episodes);
    - explicitly, in :meth:`transition_probabilities` (used by the dynamic
      programming algorithms in ``dynamic_programming.py``, which need the
      full P(s'|s,a) distribution rather than a single sample from it).
  When ``slip_probability > 0`` the environment implements a *stochastic*
  transition function: with probability ``slip_probability`` a uniformly
  random different action is executed instead of the one chosen by the
  agent, exactly as in Sutton & Barto's "slippery" gridworld / FrozenLake.
* **Reward function R**: a combination of a per-step cost, a penalty for
  bumping into a wall/boundary, an optional shaping reward for visiting a
  particular cell, and a terminal reward for reaching a goal/trap.
* **Terminal (absorbing) states**: cells in ``terminal_states``. Once
  entered, the episode ends (``terminated=True``); by convention their
  *value* V(terminal) = 0, since no further reward can be accrued.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import GridWorldConfig

# Action encoding, fixed by the assignment brief.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = (UP, RIGHT, DOWN, LEFT)
ACTION_NAMES = {UP: "up", RIGHT: "right", DOWN: "down", LEFT: "left"}

# (dx, dy) for each action. y increases *upwards*, matching plotting.py.
ACTION_DELTAS = {
    UP: (0, 1),
    RIGHT: (1, 0),
    DOWN: (0, -1),
    LEFT: (-1, 0),
}


class GridWorldEnv(gym.Env):
    """A configurable, fully-observable GridWorld MDP.

    Parameters
    ----------
    config:
        A :class:`~gridworld.config.GridWorldConfig` describing the MDP.
    render_mode:
        Either ``None`` or ``"rgb_array"``. If ``"rgb_array"``, :meth:`render`
        returns an RGB image of the current grid state (via ``plotting.py``).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, config: GridWorldConfig, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        self.width = config.width
        self.height = config.height
        self.n_states = self.width * self.height

        # Flattened discrete observation/action spaces -> compatible with
        # both tabular RL code and Stable-Baselines3.
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(4)  # up, right, down, left

        # Cells the agent can never occupy.
        self.walls = set(config.walls)

        # Internal episode state.
        self._agent_pos: tuple[int, int] = config.start
        self._step_count: int = 0
        self._rng = np.random.default_rng(config.seed)

        # Trajectory recorded during the current/most recent episode, used
        # by animation.py. Each entry corresponds to one environment step
        # (the initial reset position is stored as step 0 with no action).
        self.trajectory: dict[str, list[Any]] = self._empty_trajectory()

    # ------------------------------------------------------------------
    # Coordinate <-> state index helpers
    # ------------------------------------------------------------------
    def coord_to_state(self, coord: tuple[int, int]) -> int:
        """Convert an (x, y) grid cell to a flattened state index."""
        x, y = coord
        return y * self.width + x

    def state_to_coord(self, state: int) -> tuple[int, int]:
        """Convert a flattened state index back to an (x, y) grid cell."""
        y, x = divmod(state, self.width)
        return (x, y)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._agent_pos = self.config.start
        self._step_count = 0
        self.trajectory = self._empty_trajectory()
        self.trajectory["positions"].append(self._agent_pos)

        obs = self.coord_to_state(self._agent_pos)
        info = {
            "position": self._agent_pos,
            "step": self._step_count,
            "is_terminal": self._is_terminal(self._agent_pos),
        }
        return obs, info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected one of {ACTIONS}")

        intended_action = int(action)
        actual_action = self._maybe_slip(intended_action)

        new_pos, invalid_move = self._attempt_move(self._agent_pos, actual_action)
        reward = self._reward(new_pos, invalid_move)

        self._agent_pos = new_pos
        self._step_count += 1

        terminated = self._is_terminal(self._agent_pos)
        truncated = self._step_count >= self.config.max_steps and not terminated

        info = {
            "position": self._agent_pos,
            "step": self._step_count,
            "intended_action": intended_action,
            "actual_action": actual_action,
            "invalid_move": invalid_move,
            "is_terminal": terminated,
        }

        # Record this transition for later animation/analysis.
        self.trajectory["positions"].append(self._agent_pos)
        self.trajectory["actions"].append(actual_action)
        self.trajectory["rewards"].append(reward)
        self.trajectory["dones"].append(terminated or truncated)
        self.trajectory["infos"].append(info)

        obs = self.coord_to_state(self._agent_pos)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        # Local import so plotting/matplotlib is only required if rendering
        # is actually used.
        from .plotting import plot_gridworld
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        plot_gridworld(self, ax=ax)
        ax.scatter(*self._agent_pos, s=400, c="tab:blue", zorder=5, marker="o")
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return image

    def close(self) -> None:
        return None

    # ------------------------------------------------------------------
    # MDP model, exposed for dynamic programming
    # ------------------------------------------------------------------
    def transition_probabilities(
        self, state: int, action: int
    ) -> list[tuple[float, int, float, bool]]:
        """Return the full transition distribution P(s' | s, a).

        This is the explicit model-based view of the MDP dynamics used by
        value iteration / policy iteration. It must stay consistent with the
        sampling logic in :meth:`step`.

        Returns a list of ``(probability, next_state, reward, terminated)``
        tuples. Probabilities sum to 1. Terminal states are treated as
        absorbing: taking any action from a terminal state deterministically
        "stays" with reward 0, so that V(terminal) = 0 is a fixed point.
        """
        pos = self.state_to_coord(state)

        if self._is_terminal(pos):
            return [(1.0, state, 0.0, True)]

        slip = self.config.slip_probability
        outcomes: dict[int, list] = {}

        def add_outcome(prob: float, act: int) -> None:
            new_pos, invalid = self._attempt_move(pos, act)
            reward = self._reward(new_pos, invalid)
            next_state = self.coord_to_state(new_pos)
            terminated = self._is_terminal(new_pos)
            key = (next_state, reward, terminated)
            outcomes[key] = outcomes.get(key, 0.0) + prob

        if slip <= 0.0:
            add_outcome(1.0, action)
        else:
            add_outcome(1.0 - slip, action)
            other_actions = [a for a in ACTIONS if a != action]
            for a in other_actions:
                add_outcome(slip / len(other_actions), a)

        return [(p, s, r, t) for (s, r, t), p in outcomes.items()]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _empty_trajectory(self) -> dict[str, list[Any]]:
        return {"positions": [], "actions": [], "rewards": [], "dones": [], "infos": []}

    def _maybe_slip(self, action: int) -> int:
        """With probability slip_probability, execute a random different action."""
        if self.config.slip_probability <= 0.0:
            return action
        if self._rng.random() < self.config.slip_probability:
            other_actions = [a for a in ACTIONS if a != action]
            return int(self._rng.choice(other_actions))
        return action

    def _attempt_move(
        self, pos: tuple[int, int], action: int
    ) -> tuple[tuple[int, int], bool]:
        """Apply an action to a position, returning (new_pos, invalid_move)."""
        dx, dy = ACTION_DELTAS[action]
        x, y = pos
        new_pos = (x + dx, y + dy)
        if not self._in_bounds(new_pos) or new_pos in self.walls:
            return pos, True
        return new_pos, False

    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_terminal(self, pos: tuple[int, int]) -> bool:
        return pos in self.config.terminal_states

    def _reward(self, new_pos: tuple[int, int], invalid_move: bool) -> float:
        """Reward function R(s, a, s').

        Composition (all additive except invalid moves, which are treated
        as a standalone penalty since the agent does not actually move):
            - invalid move  -> invalid_move_reward only
            - terminal cell -> step_reward + terminal_states[new_pos]
            - shaped cell   -> step_reward + rewards[new_pos]
            - otherwise     -> step_reward
        """
        if invalid_move:
            return self.config.invalid_move_reward

        reward = self.config.step_reward
        if new_pos in self.config.terminal_states:
            reward += self.config.terminal_states[new_pos]
        elif new_pos in self.config.rewards:
            reward += self.config.rewards[new_pos]
        return reward
