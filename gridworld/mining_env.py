"""MineWorldEnv: a GridWorld MDP with a fifth action, ``MINE``, and memory.

Relation to finite MDP theory
------------------------------
MineWorld is still an episodic finite MDP (S, A, P, R, gamma), but the state
is no longer just the agent's cell:

* **State space S**: ``(agent_position, mining_counts)`` where
  ``mining_counts`` is a vector holding, for each configured mining node, how
  many mining attempts have already been made there *during this episode*
  (capped at ``max_mining_count``). Because the payoff of the next attempt
  depends on those counts, they must be part of the state for the process to
  be Markov -- a single scalar "streak" would not be enough once several
  nodes can be mined independently. The pair is flattened to a single integer
  by :meth:`MineWorldEnv.components_to_state`, so the observation space is a
  plain ``Discrete`` and all the existing tabular code keeps working.
* **Action space A**: ``Discrete(5)`` = {up, right, down, left, mine}.
  ``MINE`` never changes the agent's cell.
* **Transition function P(s'|s,a)**: enumerated once, by the pure helper
  :meth:`MineWorldEnv._transition_branches`, and consumed in two ways:
    - :meth:`step` *samples* one branch (and, for a successful mine, one
      reward draw);
    - :meth:`transition_probabilities` *enumerates* all branches for the
      dynamic-programming code.
  Because both go through the same helper, the sampled environment and the
  explicit model can never drift apart.
* **Reward function R**: the ordinary GridWorld composition (step cost,
  invalid-move penalty, shaping rewards, terminal rewards) plus the mining
  gamble described in ``mining_config.py``.
* **Termination**: entering an ordinary terminal cell *or* a mine collapse.
  A collapse is signalled purely by the transition's ``terminated`` flag --
  the mining node itself stays non-terminal, because a *successful* mine
  must leave the agent free to keep playing from the same cell.

Order of operations in a single step
------------------------------------
1. slip: the intended action may be replaced by a uniformly random
   *different* action (this can turn a move into a ``MINE`` and vice versa);
2. the **actual** action decides everything else -- in particular, mining
   only happens when the actual action is ``MINE``;
3. movement (pure, spatial) or the mining gamble (which consumes one attempt
   at the node the agent is standing on).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .mining_config import (
    EXACT_REWARD_DISTRIBUTIONS,
    ContinuousRewardModelError,
    MineWorldConfig,
)

# Action encoding, fixed by the assignment brief.
UP, RIGHT, DOWN, LEFT, MINE = 0, 1, 2, 3, 4
ACTIONS = (UP, RIGHT, DOWN, LEFT, MINE)
MOVEMENT_ACTIONS = (UP, RIGHT, DOWN, LEFT)
ACTION_NAMES = {UP: "up", RIGHT: "right", DOWN: "down", LEFT: "left", MINE: "mine"}

# (dx, dy) for each action. y increases *upwards*, matching plotting.py.
# MINE is included with a zero delta so that "the agent never moves when
# mining" is stated once, here, rather than special-cased in the movement
# helper. The mining *reward* logic lives entirely in _mining_branches.
ACTION_DELTAS = {
    UP: (0, 1),
    RIGHT: (1, 0),
    DOWN: (0, -1),
    LEFT: (-1, 0),
    MINE: (0, 0),
}

# Mining outcome labels used by MiningTransition.
NO_MINING = "none"
MINING_POSITIVE = "positive"
MINING_FAILURE = "failure"


@dataclass(frozen=True)
class MiningTransition:
    """One branch of P(s' | s, a): an immutable, fully-described outcome.

    A branch is *not* a full ``(probability, next_state, reward, terminated)``
    tuple, because the reward of a successful mine may still be random. It
    records the deterministic part of the reward (``base_reward``) plus
    enough information (``mining_outcome``, ``attempt_number``) for either
    consumer to finish the job: :meth:`MineWorldEnv.step` draws a sample,
    :meth:`MineWorldEnv.transition_probabilities` enumerates the finite
    support.

    Attributes
    ----------
    probability:
        Probability of this branch given the *intended* action (already
        including the slip probability).
    actual_action:
        The action the environment really executed on this branch.
    next_position, next_mining_counts:
        The successor state's two components.
    base_reward:
        Everything in the reward that is known exactly: ``step_reward``,
        ``invalid_move_reward``, shaping and terminal rewards, and the
        deterministic mining-failure penalty.
    mining_outcome:
        ``"none"``, ``"positive"`` or ``"failure"``.
    attempt_number:
        ``m``, the 1-based attempt index used to evaluate the mining
        schedules, or ``None`` when no mining was attempted.
    positive_reward_probability:
        ``p(m)`` for this attempt, or ``None`` when no mining was attempted.
    terminated:
        True if the episode ends on this branch (ordinary terminal cell or
        mine collapse).
    invalid_move:
        True if a movement action was blocked by a wall or the grid edge.
    mining_node:
        The node that was mined, or ``None``.
    """

    probability: float
    actual_action: int
    next_position: tuple[int, int]
    next_mining_counts: tuple[int, ...]
    base_reward: float
    mining_outcome: str
    attempt_number: int | None
    positive_reward_probability: float | None
    terminated: bool
    invalid_move: bool
    mining_node: tuple[int, int] | None

    @property
    def mining_attempted(self) -> bool:
        return self.mining_outcome != NO_MINING

    @property
    def mining_success(self) -> bool:
        return self.mining_outcome == MINING_POSITIVE

    @property
    def mining_failure(self) -> bool:
        return self.mining_outcome == MINING_FAILURE


class MineWorldEnv(gym.Env):
    """A configurable, fully-observable MineWorld MDP.

    Parameters
    ----------
    config:
        A :class:`~gridworld.mining_config.MineWorldConfig` describing the MDP.
    render_mode:
        Either ``None`` or ``"rgb_array"``. If ``"rgb_array"``, :meth:`render`
        returns an RGB image of the current grid state (via ``plotting.py``).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    # Used by animation.py to label actions in frame titles.
    action_names = ACTION_NAMES

    def __init__(self, config: MineWorldConfig, render_mode: str | None = None) -> None:
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        self.width = config.width
        self.height = config.height

        # Mining nodes in a fixed order: index i of every mining-count vector
        # refers to self.mining_nodes[i].
        self.mining_nodes: tuple[tuple[int, int], ...] = config.mining_nodes_ordered
        self.n_mining_nodes = len(self.mining_nodes)
        self._node_index = {node: i for i, node in enumerate(self.mining_nodes)}

        # Mixed-radix encoding of the count vector: each count is one digit in
        # base (max_mining_count + 1).
        self._count_radix = config.n_mining_levels
        self._n_count_vectors = self._count_radix**self.n_mining_nodes
        self.n_states = self.width * self.height * self._n_count_vectors

        # Flattened discrete observation/action spaces -> compatible with both
        # tabular RL code and Stable-Baselines3.
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(len(ACTIONS))  # up, right, down, left, mine

        # Cells the agent can never occupy.
        self.walls = set(config.walls)

        # Internal episode state. Mining counts are an immutable tuple, so a
        # snapshot handed to a caller (or stored in the trajectory) can never
        # be mutated behind the environment's back.
        self._agent_pos: tuple[int, int] = config.start
        self._mining_counts: tuple[int, ...] = self._zero_counts()
        self._step_count: int = 0
        self._rng = np.random.default_rng(config.seed)

        # Trajectory recorded during the current/most recent episode, used by
        # animation.py. The reset state is stored as frame 0 (no action).
        self.trajectory: dict[str, list[Any]] = self._empty_trajectory()

    # ------------------------------------------------------------------
    # Augmented state <-> components helpers
    # ------------------------------------------------------------------
    def components_to_state(
        self,
        position: tuple[int, int],
        mining_counts: tuple[int, ...],
    ) -> int:
        """Encode ``((x, y), (c_1, ..., c_K))`` as a single integer state.

        The encoding is mixed-radix: the position index ``y * width + x`` is
        the most significant digit, followed by one digit per mining node in
        base ``max_mining_count + 1``. With ``K = 0`` mining nodes this
        degenerates to the plain GridWorld encoding.
        """
        self._validate_position(position)
        counts = self._validate_counts(mining_counts)

        x, y = position
        position_index = y * self.width + x

        count_index = 0
        for count in counts:
            count_index = count_index * self._count_radix + count
        return position_index * self._n_count_vectors + count_index

    def state_to_components(self, state: int) -> tuple[tuple[int, int], tuple[int, ...]]:
        """Decode an integer state into ``((x, y), (c_1, ..., c_K))``.

        Exact inverse of :meth:`components_to_state` over the whole finite
        state space.
        """
        state = int(state)
        if not 0 <= state < self.n_states:
            raise ValueError(f"state must be in [0, {self.n_states - 1}], got {state}")

        position_index, count_index = divmod(state, self._n_count_vectors)
        y, x = divmod(position_index, self.width)

        counts = [0] * self.n_mining_nodes
        for i in range(self.n_mining_nodes - 1, -1, -1):
            count_index, counts[i] = divmod(count_index, self._count_radix)
        return (x, y), tuple(counts)

    def state_to_coord(self, state: int) -> tuple[int, int]:
        """Only the spatial coordinate of a state (used by ``plotting.py``)."""
        position, _counts = self.state_to_components(state)
        return position

    def state_to_mining_counts(self, state: int) -> tuple[int, ...]:
        """Only the mining-count vector of a state."""
        _position, counts = self.state_to_components(state)
        return counts

    def mining_node_index(self, cell: tuple[int, int]) -> int | None:
        """Index of ``cell`` in the ordered mining-node list, or ``None``."""
        return self._node_index.get(cell)

    def skip_state_indices(self) -> set[int]:
        """States the DP sweeps should never update: terminal and wall cells.

        Every mining-count vector at such a cell is skipped, since the cell is
        either absorbing (V = 0 by convention) or unreachable.
        """
        skip: set[int] = set()
        blocked = set(self.config.terminal_states) | self.walls
        for cell in blocked:
            for count_index in range(self._n_count_vectors):
                x, y = cell
                skip.add((y * self.width + x) * self._n_count_vectors + count_index)
        return skip

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
        self._mining_counts = self._zero_counts()
        self._step_count = 0

        self.trajectory = self._empty_trajectory()
        self.trajectory["positions"].append(self._agent_pos)
        self.trajectory["mining_counts"].append(self._mining_counts)

        obs = self.components_to_state(self._agent_pos, self._mining_counts)
        info = {
            "position": self._agent_pos,
            "mining_counts": self._mining_counts,
            "current_node_mining_count": self._current_node_count(),
            "step": self._step_count,
            "is_terminal": self._is_terminal(self._agent_pos),
        }
        return obs, info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected one of {ACTIONS}")

        intended_action = int(action)
        # 1. Slip first: everything downstream keys off the *actual* action.
        actual_action = self._maybe_slip(intended_action)

        # 2. Enumerate the branches available given that actual action, then
        #    sample one. Using the same enumerator as
        #    transition_probabilities() is what keeps the two views consistent.
        branches = self._branches_for_actual_action(
            self._agent_pos, self._mining_counts, actual_action, weight=1.0
        )
        branch = self._sample_branch(branches)

        # 3. Finish the reward: only a *successful* mine is still random.
        reward = branch.base_reward
        if branch.mining_success:
            assert branch.attempt_number is not None  # set on every mining branch
            reward += self.config.sample_positive_reward(branch.attempt_number, self._rng)

        self._agent_pos = branch.next_position
        self._mining_counts = branch.next_mining_counts
        self._step_count += 1

        terminated = branch.terminated
        truncated = self._step_count >= self.config.max_steps and not terminated

        info = self._step_info(branch, intended_action, terminated)

        # Record this transition for later animation/analysis. Counts are
        # tuples, so each frame keeps its own immutable snapshot.
        self.trajectory["positions"].append(self._agent_pos)
        self.trajectory["mining_counts"].append(self._mining_counts)
        self.trajectory["actions"].append(branch.actual_action)
        self.trajectory["rewards"].append(reward)
        self.trajectory["dones"].append(terminated or truncated)
        self.trajectory["infos"].append(info)

        obs = self.components_to_state(self._agent_pos, self._mining_counts)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        # Local import so plotting/matplotlib is only required if rendering is
        # actually used.
        from .plotting import plot_gridworld
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        # Rendering is a pure read of the current state: nothing below mutates
        # the environment.
        plot_gridworld(self, ax=ax, mining_counts=self._mining_counts)
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

        This is the explicit model-based view of the dynamics used by value
        iteration / policy iteration. It is built from the same branch
        enumerator as :meth:`step`, so the two can never disagree, and it is
        completely side-effect free: calling it does not touch the agent
        position, the mining counts, the step counter or the RNG.

        Returns a list of ``(probability, next_state, reward, terminated)``
        tuples whose probabilities sum to 1. Ordinary terminal states are
        absorbing: any action from them stays put with reward 0, so that
        V(terminal) = 0 is a fixed point.

        Raises
        ------
        ContinuousRewardModelError
            If the configured positive-reward distribution is continuous. A
            finite list of ``(probability, reward)`` pairs cannot represent a
            continuous law, and silently substituting its expectation would
            make this model describe a *different* MDP from the one
            :meth:`step` samples. The check is made up front, for every state
            and action, so that "this environment has no exact tabular model"
            is a property of the environment rather than a surprise that only
            surfaces once a sweep happens to reach a mining node.
        """
        self._require_exact_reward_model()
        position, counts = self.state_to_components(state)

        if self._is_terminal(position):
            return [(1.0, int(state), 0.0, True)]

        outcomes: dict[tuple[int, float, bool], float] = {}

        def add(probability: float, next_state: int, reward: float, terminated: bool) -> None:
            key = (next_state, reward, terminated)
            outcomes[key] = outcomes.get(key, 0.0) + probability

        for branch in self._transition_branches(position, counts, int(action)):
            next_state = self.components_to_state(
                branch.next_position, branch.next_mining_counts
            )
            if branch.mining_success:
                assert branch.attempt_number is not None  # set on every mining branch
                # Expand the finite support of R_positive(m). Raises for a
                # continuous distribution.
                for reward_probability, value in self.config.positive_reward_support(
                    branch.attempt_number
                ):
                    add(
                        branch.probability * reward_probability,
                        next_state,
                        branch.base_reward + value,
                        branch.terminated,
                    )
            else:
                add(branch.probability, next_state, branch.base_reward, branch.terminated)

        return [(p, s, r, t) for (s, r, t), p in outcomes.items()]

    # ------------------------------------------------------------------
    # The shared, side-effect-free transition model
    # ------------------------------------------------------------------
    def _transition_branches(
        self,
        position: tuple[int, int],
        mining_counts: tuple[int, ...],
        intended_action: int,
    ) -> tuple[MiningTransition, ...]:
        """All branches reachable from ``(position, counts)`` for an intended action.

        Pure: it reads the configuration and its arguments only. Slip is
        applied first (so a slip can turn a move into a mine, or a mine into a
        move), then each *actual* action is expanded into its own outcomes.
        """
        slip = self.config.slip_probability
        branches: list[MiningTransition] = []

        if slip <= 0.0:
            branches.extend(
                self._branches_for_actual_action(
                    position, mining_counts, intended_action, weight=1.0
                )
            )
        else:
            branches.extend(
                self._branches_for_actual_action(
                    position, mining_counts, intended_action, weight=1.0 - slip
                )
            )
            other_actions = [a for a in ACTIONS if a != intended_action]
            share = slip / len(other_actions)
            for other in other_actions:
                branches.extend(
                    self._branches_for_actual_action(
                        position, mining_counts, other, weight=share
                    )
                )
        return tuple(branches)

    def _branches_for_actual_action(
        self,
        position: tuple[int, int],
        mining_counts: tuple[int, ...],
        actual_action: int,
        weight: float,
    ) -> tuple[MiningTransition, ...]:
        """Branches for an action the environment has already decided to execute.

        Movement actions give exactly one branch. ``MINE`` gives one branch off
        a mining node, and two (success / collapse) on one. ``weight`` scales
        the branch probabilities, so the caller can fold in the slip
        probability.
        """
        if actual_action == MINE:
            return self._mining_branches(position, mining_counts, weight)
        return self._movement_branches(position, mining_counts, actual_action, weight)

    def _movement_branches(
        self,
        position: tuple[int, int],
        mining_counts: tuple[int, ...],
        actual_action: int,
        weight: float,
    ) -> tuple[MiningTransition, ...]:
        """The single branch produced by an ordinary movement action.

        Mining counts are untouched: entering, leaving or bumping into a
        mining node is not a mining attempt.
        """
        new_position, invalid_move = self._attempt_move(position, actual_action)
        return (
            MiningTransition(
                probability=weight,
                actual_action=actual_action,
                next_position=new_position,
                next_mining_counts=mining_counts,
                base_reward=self._movement_reward(new_position, invalid_move),
                mining_outcome=NO_MINING,
                attempt_number=None,
                positive_reward_probability=None,
                terminated=self._is_terminal(new_position),
                invalid_move=invalid_move,
                mining_node=None,
            ),
        )

    def _mining_branches(
        self,
        position: tuple[int, int],
        mining_counts: tuple[int, ...],
        weight: float,
    ) -> tuple[MiningTransition, ...]:
        """The branches produced by actually executing ``MINE`` at ``position``.

        Off a mining node this is a no-op that pays ``invalid_mining_reward``
        (or just ``step_reward``). On a mining node it is the gamble: attempt
        number ``m = count + 1``, success with probability ``p(m)``, collapse
        otherwise. The stored count is incremented (and capped) on *both*
        branches, because it counts *attempts*.
        """
        node_index = self._node_index.get(position)

        if node_index is None:
            invalid_mining_reward = self.config.invalid_mining_reward
            base_reward = (
                self.config.step_reward
                if invalid_mining_reward is None
                else invalid_mining_reward
            )
            return (
                MiningTransition(
                    probability=weight,
                    actual_action=MINE,
                    next_position=position,
                    next_mining_counts=mining_counts,
                    base_reward=base_reward,
                    mining_outcome=NO_MINING,
                    attempt_number=None,
                    positive_reward_probability=None,
                    # MINE never moves the agent, so it can never *enter* an
                    # ordinary terminal cell, and never re-triggers a shaping
                    # or terminal reward.
                    terminated=False,
                    invalid_move=False,
                    mining_node=None,
                ),
            )

        attempt_number = mining_counts[node_index] + 1
        success_probability = self.config.positive_reward_probability(attempt_number)
        next_counts = self._incremented_counts(mining_counts, node_index)
        step_reward = self.config.step_reward

        success = MiningTransition(
            probability=weight * success_probability,
            actual_action=MINE,
            next_position=position,
            next_mining_counts=next_counts,
            # The random positive payout is added by the caller; only the
            # deterministic part lives here.
            base_reward=step_reward,
            mining_outcome=MINING_POSITIVE,
            attempt_number=attempt_number,
            positive_reward_probability=success_probability,
            terminated=False,
            invalid_move=False,
            mining_node=position,
        )
        failure = MiningTransition(
            probability=weight * (1.0 - success_probability),
            actual_action=MINE,
            next_position=position,
            next_mining_counts=next_counts,
            base_reward=step_reward + self.config.mining_failure_reward_at(attempt_number),
            mining_outcome=MINING_FAILURE,
            attempt_number=attempt_number,
            positive_reward_probability=success_probability,
            # A collapse ends the episode from a cell that is *not* an
            # ordinary terminal state: termination is carried by this flag.
            terminated=True,
            invalid_move=False,
            mining_node=position,
        )

        branches = [b for b in (success, failure) if b.probability > 0.0]
        return tuple(branches) if branches else (success,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _empty_trajectory(self) -> dict[str, list[Any]]:
        return {
            "positions": [],
            "mining_counts": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "infos": [],
        }

    def _zero_counts(self) -> tuple[int, ...]:
        return (0,) * self.n_mining_nodes

    def _incremented_counts(
        self, mining_counts: tuple[int, ...], node_index: int
    ) -> tuple[int, ...]:
        """A new count vector with one node's attempt count bumped (and capped)."""
        counts = list(mining_counts)
        counts[node_index] = min(counts[node_index] + 1, self.config.max_mining_count)
        return tuple(counts)

    def _maybe_slip(self, action: int) -> int:
        """With probability slip_probability, execute a random *different* action."""
        if self.config.slip_probability <= 0.0:
            return action
        if self._rng.random() < self.config.slip_probability:
            other_actions = [a for a in ACTIONS if a != action]
            return int(self._rng.choice(other_actions))
        return action

    def _sample_branch(self, branches: tuple[MiningTransition, ...]) -> MiningTransition:
        """Draw one branch according to its (conditional) probability."""
        if len(branches) == 1:
            return branches[0]
        threshold = float(self._rng.random())
        cumulative = 0.0
        for branch in branches:
            cumulative += branch.probability
            if threshold < cumulative:
                return branch
        return branches[-1]  # guards against floating-point shortfall

    def _attempt_move(
        self, position: tuple[int, int], action: int
    ) -> tuple[tuple[int, int], bool]:
        """Pure spatial movement: ``(new_position, invalid_move)``.

        This helper knows nothing about mining and never mutates anything.
        """
        dx, dy = ACTION_DELTAS[action]
        x, y = position
        new_position = (x + dx, y + dy)
        if not self._in_bounds(new_position) or new_position in self.walls:
            return position, True
        return new_position, False

    def _in_bounds(self, position: tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_terminal(self, position: tuple[int, int]) -> bool:
        """Whether ``position`` is an *ordinary* terminal cell.

        Mining collapses are deliberately not represented here: a mining node
        must stay non-terminal so that a successful mine can be followed by
        another action from the same cell.
        """
        return position in self.config.terminal_states

    def _movement_reward(self, new_position: tuple[int, int], invalid_move: bool) -> float:
        """Reward for a movement action, identical to the base GridWorld's R.

        Note this is *only* used for movement: mining rewards depend on the
        action taken, not on the cell entered, so they cannot be derived from
        ``new_position`` alone.

            - invalid move  -> invalid_move_reward only
            - terminal cell -> step_reward + terminal_states[new_position]
            - shaped cell   -> step_reward + rewards[new_position]
            - otherwise     -> step_reward
        """
        if invalid_move:
            return self.config.invalid_move_reward

        reward = self.config.step_reward
        if new_position in self.config.terminal_states:
            reward += self.config.terminal_states[new_position]
        elif new_position in self.config.rewards:
            reward += self.config.rewards[new_position]
        return reward

    def _current_node_count(self) -> int | None:
        """Attempt count at the node the agent is standing on, if any."""
        node_index = self._node_index.get(self._agent_pos)
        return None if node_index is None else self._mining_counts[node_index]

    def _step_info(
        self, branch: MiningTransition, intended_action: int, terminated: bool
    ) -> dict[str, Any]:
        """Per-step diagnostics. Only immutable values leave the environment."""
        return {
            "position": self._agent_pos,
            "mining_counts": self._mining_counts,
            "current_node_mining_count": self._current_node_count(),
            "intended_action": intended_action,
            "actual_action": branch.actual_action,
            "mining_attempted": branch.mining_attempted,
            "mining_success": branch.mining_success,
            "mining_failure": branch.mining_failure,
            "mining_node": branch.mining_node,
            "mining_attempt_number": branch.attempt_number,
            "positive_reward_probability": branch.positive_reward_probability,
            "step": self._step_count,
            "invalid_move": branch.invalid_move,
            "is_terminal": terminated,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _require_exact_reward_model(self) -> None:
        """Guard the exact-model API against continuous reward distributions."""
        if not self.config.is_exact_reward_model:
            raise ContinuousRewardModelError(
                f"positive_reward_distribution="
                f"{self.config.positive_reward_distribution!r} is continuous, so this "
                "MineWorld has no exact finite tabular model: "
                "transition_probabilities() cannot enumerate it, and value/policy "
                "iteration cannot be run on it. Silently replacing the random payout "
                "by its expectation would make the model describe a different MDP "
                f"from the one step() samples. Use one of "
                f"{EXACT_REWARD_DISTRIBUTIONS} for model-based dynamic programming, "
                "or stick to sampled methods such as q_learning()."
            )

    def _validate_position(self, position: tuple[int, int]) -> None:
        if len(position) != 2:
            raise ValueError(f"position must be an (x, y) pair, got {position!r}")
        if not self._in_bounds(position):
            raise ValueError(f"position {position} is outside the {self.width}x{self.height} grid")

    def _validate_counts(self, mining_counts: tuple[int, ...]) -> tuple[int, ...]:
        counts = tuple(int(c) for c in mining_counts)
        if len(counts) != self.n_mining_nodes:
            raise ValueError(
                f"mining_counts must have one entry per mining node "
                f"({self.n_mining_nodes}), got {len(counts)}"
            )
        for count in counts:
            if not 0 <= count <= self.config.max_mining_count:
                raise ValueError(
                    f"mining counts must be in [0, {self.config.max_mining_count}], "
                    f"got {counts}"
                )
        return counts
