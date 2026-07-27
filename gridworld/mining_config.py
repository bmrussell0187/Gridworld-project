"""Configuration for the MineWorld environment (GridWorld + mining nodes).

MineWorld extends the plain GridWorld MDP with a fifth action, ``MINE``, and
a set of *mining nodes*. Standing on a mining node and executing ``MINE``
is a gamble: with probability ``p(m)`` the agent receives a positive reward
drawn from a configurable distribution, and with probability ``1 - p(m)`` the
mine collapses -- the agent receives a large negative reward and the episode
ends.

The attempt number ``m`` is what makes this interesting: ``p(m)`` is
*non-increasing* in ``m`` while the positive reward *grows* with ``m``, so a
greedy "mine forever" policy eventually destroys itself. The agent therefore
faces a genuine risk/reward decision at every node.

Design notes
------------
* Everything here is plain data (floats, ints, tuples, sets, dicts) -- no
  Python callables -- so a configuration stays easy to print, diff, serialise
  to JSON/YAML and reason about in a dissertation appendix. The reward and
  probability *schedules* are selected by short string names (``"linear"``,
  ``"categorical"``, ...) rather than by arbitrary lambdas.
* Some reward distributions have finite support (``"deterministic"``,
  ``"categorical"``) and can therefore be enumerated exactly by
  :meth:`~gridworld.mining_env.MineWorldEnv.transition_probabilities` for
  model-based dynamic programming. The continuous ones (``"normal"``,
  ``"lognormal"``, ``"gamma"``) can only be *sampled*; asking for an exact
  model with those raises :class:`ContinuousRewardModelError` rather than
  silently substituting an expectation (which would make ``step()`` and
  ``transition_probabilities()`` describe two different MDPs).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from .config import GridWorldConfig

# Schedules for p_positive(m), the probability that the m-th mining attempt
# at a node succeeds. Every schedule must be non-increasing in m.
POSITIVE_PROBABILITY_SCHEDULES = ("constant", "linear", "geometric")

# Reward distributions with finite support: usable by exact dynamic programming.
EXACT_REWARD_DISTRIBUTIONS = ("deterministic", "categorical")
# Continuous reward distributions: sampled ``step()`` only.
SAMPLING_ONLY_REWARD_DISTRIBUTIONS = ("normal", "lognormal", "gamma")
REWARD_DISTRIBUTIONS = EXACT_REWARD_DISTRIBUTIONS + SAMPLING_ONLY_REWARD_DISTRIBUTIONS

# A tabular state space has size width * height * (max_mining_count + 1) ** K,
# i.e. it grows *exponentially* in the number of mining nodes K. These
# thresholds turn an accidental combinatorial explosion into a clear
# warning/error rather than an out-of-memory crash inside value iteration.
WARN_STATE_SPACE_SIZE = 100_000
MAX_STATE_SPACE_SIZE = 2_000_000

# How many times a non-positive draw from a continuous "positive reward"
# distribution is resampled before being clipped (see ``sample_positive_reward``).
_MAX_POSITIVE_RESAMPLES = 100


class ContinuousRewardModelError(NotImplementedError):
    """Raised when an exact P(s'|s,a) is requested for a continuous reward law.

    ``transition_probabilities`` returns a *finite* list of
    ``(probability, next_state, reward, terminated)`` tuples, which cannot
    represent a continuous reward distribution. Rather than quietly replacing
    the random reward by its mean -- which would make the dynamic-programming
    model disagree with what ``step()`` actually pays out -- MineWorld raises
    this error.
    """


@dataclass
class MineWorldConfig(GridWorldConfig):
    """Full specification of a MineWorld finite MDP.

    Inherits every field of :class:`~gridworld.config.GridWorldConfig` (grid
    size, start, terminal states, shaping rewards, walls, step reward,
    invalid-move reward, slip probability, max steps, seed) and adds the
    mining layer below.

    Attributes
    ----------
    mining_nodes:
        Cells at which the ``MINE`` action is productive. Anywhere else
        ``MINE`` simply pays ``invalid_mining_reward`` (or ``step_reward``).
        Mining nodes must lie inside the grid, must not be walls, must not be
        ordinary terminal states, and must be unique.
    max_mining_count:
        Cap ``C`` on each node's stored attempt count, so every count lives in
        ``{0, 1, ..., C}`` and the state space stays finite. Once a node's
        count reaches ``C`` it stops growing and every further attempt is
        evaluated at attempt number ``m = C + 1``: the schedule *saturates* at
        its final (hardest, most valuable) level rather than resetting.
    positive_probability_schedule:
        One of ``POSITIVE_PROBABILITY_SCHEDULES``:

        * ``"constant"``  -> ``p(m) = initial_positive_probability``
        * ``"linear"``    -> ``p(m) = max(p_min, p_0 - (m - 1) * decrement)``
        * ``"geometric"`` -> ``p(m) = max(p_min, p_0 * decay ** (m - 1))``
    initial_positive_probability:
        ``p_0``: probability that the *first* attempt at a fresh node succeeds.
    positive_probability_decrement:
        Per-attempt decrease used by the ``"linear"`` schedule (>= 0).
    positive_probability_decay:
        Per-attempt multiplier used by the ``"geometric"`` schedule, in (0, 1].
    minimum_positive_probability:
        Floor ``p_min``, so success never becomes literally impossible.
    positive_reward_distribution:
        One of ``REWARD_DISTRIBUTIONS``. ``"deterministic"`` and
        ``"categorical"`` have finite support and support exact dynamic
        programming; ``"normal"``, ``"lognormal"`` and ``"gamma"`` are
        simulation-only.
    positive_reward_base_mean, positive_reward_mean_increment:
        The *scale* of the m-th positive reward,
        ``mu(m) = base_mean + (m - 1) * mean_increment``. The increment is
        required to be >= 0, so the payoff grows with the attempt number.
    positive_reward_base_std, positive_reward_std_increment:
        The *spread* ``sigma(m) = base_std + (m - 1) * std_increment``, used
        by the continuous distributions.
    positive_reward_multipliers, positive_reward_multiplier_probabilities:
        Finite support of the ``"categorical"`` distribution, expressed as
        multipliers of ``mu(m)``. Multipliers (rather than absolute values)
        keep the "payoff grows with m" property automatic. All multipliers
        must be strictly positive and the probabilities must sum to 1.
    positive_reward_floor:
        Smallest reward a "positive" outcome may pay; used to clip a
        pathological draw from the (unbounded-below) normal distribution.
    mining_failure_reward:
        Reward paid when the mine collapses. Must be <= 0.
    mining_failure_reward_increment:
        Optional extra penalty per attempt: the m-th failure pays
        ``mining_failure_reward - (m - 1) * mining_failure_reward_increment``.
        Defaults to 0.0 (a flat penalty), which keeps teaching examples simple.
    invalid_mining_reward:
        Reward for executing ``MINE`` while *not* standing on a mining node.
        ``None`` (the default) means "just the ordinary ``step_reward``".

    Reward conventions
    ------------------
    ``step_reward`` is *additive* with the mining outcome, exactly as it is
    additive with terminal and shaping rewards in the base GridWorld:

        successful mine -> ``step_reward + R_positive(m)``
        collapsed mine  -> ``step_reward + mining_failure_reward_at(m)``

    Executing ``MINE`` never moves the agent, so it never re-triggers a
    terminal or shaping reward (those fire on *entering* a cell).
    """

    # --- the mining layer -------------------------------------------------
    mining_nodes: set[tuple[int, int]] = field(default_factory=set)
    max_mining_count: int = 3

    # --- p_positive(m) ----------------------------------------------------
    positive_probability_schedule: str = "linear"
    initial_positive_probability: float = 0.95
    positive_probability_decrement: float = 0.15
    positive_probability_decay: float = 0.7
    minimum_positive_probability: float = 0.05

    # --- R_positive(m) ----------------------------------------------------
    positive_reward_distribution: str = "deterministic"
    positive_reward_base_mean: float = 1.0
    positive_reward_mean_increment: float = 1.5
    positive_reward_base_std: float = 0.2
    positive_reward_std_increment: float = 0.0
    positive_reward_multipliers: tuple[float, ...] = (0.5, 1.0, 1.5)
    positive_reward_multiplier_probabilities: tuple[float, ...] = (0.25, 0.5, 0.25)
    positive_reward_floor: float = 1e-3

    # --- the failure branch ----------------------------------------------
    mining_failure_reward: float = -10.0
    mining_failure_reward_increment: float = 0.0

    # --- mining where there is nothing to mine ----------------------------
    invalid_mining_reward: float | None = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        super().__post_init__()  # grid-level checks (bounds, walls, slip, ...)
        self._validate_mining_nodes()
        self._validate_probability_schedule()
        self._validate_reward_distribution()
        self._validate_state_space_size()

    def _validate_mining_nodes(self) -> None:
        nodes = self.mining_nodes
        if not isinstance(nodes, set):
            # Accept any iterable, but reject duplicates explicitly: silently
            # de-duplicating would quietly change the size of the state space.
            listed = list(nodes)
            if len(listed) != len(set(listed)):
                raise ValueError(f"mining_nodes contains duplicates: {listed}")
            self.mining_nodes = nodes = set(listed)

        for node in nodes:
            if not self._in_bounds(node):
                raise ValueError(f"mining node {node} is outside the grid")
            if node in self.walls:
                raise ValueError(f"mining node {node} cannot be a wall")
            if node in self.terminal_states:
                raise ValueError(
                    f"mining node {node} cannot also be an ordinary terminal state: "
                    "entering the cell would end the episode before MINE could ever "
                    "be executed there. A mining failure is represented by the "
                    "'terminated' flag of the transition, not by a terminal cell."
                )

        if self.max_mining_count < 0:
            raise ValueError("max_mining_count must be >= 0")

    def _validate_probability_schedule(self) -> None:
        if self.positive_probability_schedule not in POSITIVE_PROBABILITY_SCHEDULES:
            raise ValueError(
                f"positive_probability_schedule must be one of "
                f"{POSITIVE_PROBABILITY_SCHEDULES}, got "
                f"{self.positive_probability_schedule!r}"
            )
        if not 0.0 <= self.initial_positive_probability <= 1.0:
            raise ValueError("initial_positive_probability must be in [0, 1]")
        if not 0.0 <= self.minimum_positive_probability <= 1.0:
            raise ValueError("minimum_positive_probability must be in [0, 1]")
        if self.minimum_positive_probability > self.initial_positive_probability:
            raise ValueError(
                "minimum_positive_probability must not exceed "
                "initial_positive_probability (p(m) must be non-increasing)"
            )
        if self.positive_probability_decrement < 0.0:
            raise ValueError(
                "positive_probability_decrement must be >= 0 so that p(m) is "
                "non-increasing in the attempt number m"
            )
        if not 0.0 < self.positive_probability_decay <= 1.0:
            raise ValueError("positive_probability_decay must be in (0, 1]")

    def _validate_reward_distribution(self) -> None:
        if self.positive_reward_distribution not in REWARD_DISTRIBUTIONS:
            raise ValueError(
                f"positive_reward_distribution must be one of {REWARD_DISTRIBUTIONS}, "
                f"got {self.positive_reward_distribution!r}"
            )
        if self.positive_reward_base_mean <= 0.0:
            raise ValueError("positive_reward_base_mean must be > 0")
        if self.positive_reward_mean_increment < 0.0:
            raise ValueError(
                "positive_reward_mean_increment must be >= 0 so that the payoff "
                "grows with the attempt number m"
            )
        if self.positive_reward_base_std < 0.0 or self.positive_reward_std_increment < 0.0:
            raise ValueError("reward standard deviations/increments must be >= 0")
        if self.positive_reward_floor <= 0.0:
            raise ValueError("positive_reward_floor must be > 0 (rewards must be positive)")

        if self.positive_reward_distribution == "categorical":
            multipliers = tuple(float(v) for v in self.positive_reward_multipliers)
            probabilities = tuple(
                float(p) for p in self.positive_reward_multiplier_probabilities
            )
            if len(multipliers) == 0:
                raise ValueError("positive_reward_multipliers must not be empty")
            if len(multipliers) != len(probabilities):
                raise ValueError(
                    "positive_reward_multipliers and "
                    "positive_reward_multiplier_probabilities must have equal length"
                )
            if any(v <= 0.0 for v in multipliers):
                raise ValueError(
                    "positive_reward_multipliers must all be > 0 so that a "
                    "'positive outcome' really does pay a positive reward"
                )
            if any(p < 0.0 for p in probabilities):
                raise ValueError("positive_reward_multiplier_probabilities must be >= 0")
            if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    "positive_reward_multiplier_probabilities must sum to 1, got "
                    f"{sum(probabilities)}"
                )
            self.positive_reward_multipliers = multipliers
            self.positive_reward_multiplier_probabilities = probabilities

        if (
            self.positive_reward_distribution in ("gamma", "lognormal")
            and self.positive_reward_base_std <= 0.0
        ):
            raise ValueError(
                f"the {self.positive_reward_distribution} distribution requires "
                "positive_reward_base_std > 0"
            )

        if self.mining_failure_reward > 0.0:
            raise ValueError("mining_failure_reward should be <= 0 (it is a penalty)")
        if self.mining_failure_reward_increment < 0.0:
            raise ValueError("mining_failure_reward_increment must be >= 0")

    def _validate_state_space_size(self) -> None:
        size = self.state_space_size
        if size > MAX_STATE_SPACE_SIZE:
            raise ValueError(
                f"this configuration has {size:,} tabular states "
                f"({self.width} x {self.height} x {self.n_mining_levels}^"
                f"{self.n_mining_nodes}), which exceeds MAX_STATE_SPACE_SIZE="
                f"{MAX_STATE_SPACE_SIZE:,}. The state space grows *exponentially* "
                "in the number of mining nodes: reduce mining_nodes or "
                "max_mining_count, or move to function approximation."
            )
        if size > WARN_STATE_SPACE_SIZE:
            warnings.warn(
                f"MineWorldConfig has {size:,} tabular states; value iteration and "
                "tabular Q-learning will be slow. The state space grows "
                "exponentially in the number of mining nodes.",
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Derived structure
    # ------------------------------------------------------------------
    @property
    def mining_nodes_ordered(self) -> tuple[tuple[int, int], ...]:
        """Mining nodes in a fixed, deterministic order.

        Entry ``i`` of a mining-count vector always refers to node ``i`` of
        this tuple, so the state encoding is stable across runs and machines.
        """
        return tuple(sorted(self.mining_nodes))

    @property
    def n_mining_nodes(self) -> int:
        return len(self.mining_nodes)

    @property
    def n_mining_levels(self) -> int:
        """Number of values a single node's count can take: ``C + 1``."""
        return self.max_mining_count + 1

    @property
    def state_space_size(self) -> int:
        """``width * height * (max_mining_count + 1) ** number_of_mining_nodes``."""
        return self.width * self.height * self.n_mining_levels**self.n_mining_nodes

    @property
    def is_exact_reward_model(self) -> bool:
        """True when the positive-reward law has finite support (DP-friendly)."""
        return self.positive_reward_distribution in EXACT_REWARD_DISTRIBUTIONS

    # ------------------------------------------------------------------
    # Schedules: pure functions of the attempt number m
    # ------------------------------------------------------------------
    def _check_attempt_number(self, m: int) -> int:
        m = int(m)
        if not 1 <= m <= self.max_mining_count + 1:
            raise ValueError(
                f"attempt number must be in [1, {self.max_mining_count + 1}] "
                f"(counts are capped at max_mining_count={self.max_mining_count}), "
                f"got {m}"
            )
        return m

    def positive_reward_probability(self, m: int) -> float:
        """``p(m)``: probability that the m-th attempt at a node succeeds.

        Non-increasing in ``m`` for every supported schedule.
        """
        m = self._check_attempt_number(m)
        p0 = self.initial_positive_probability
        if self.positive_probability_schedule == "constant":
            p = p0
        elif self.positive_probability_schedule == "linear":
            p = p0 - (m - 1) * self.positive_probability_decrement
        else:  # "geometric"
            p = p0 * self.positive_probability_decay ** (m - 1)
        return float(max(self.minimum_positive_probability, p))

    def positive_reward_scale(self, m: int) -> float:
        """``mu(m) = base_mean + (m - 1) * mean_increment`` (non-decreasing)."""
        m = self._check_attempt_number(m)
        return float(
            self.positive_reward_base_mean + (m - 1) * self.positive_reward_mean_increment
        )

    def positive_reward_spread(self, m: int) -> float:
        """``sigma(m) = base_std + (m - 1) * std_increment``."""
        m = self._check_attempt_number(m)
        return float(
            self.positive_reward_base_std + (m - 1) * self.positive_reward_std_increment
        )

    def mining_failure_reward_at(self, m: int) -> float:
        """The (deterministic) penalty paid when the m-th attempt collapses."""
        m = self._check_attempt_number(m)
        return float(
            self.mining_failure_reward - (m - 1) * self.mining_failure_reward_increment
        )

    def positive_reward_support(self, m: int) -> tuple[tuple[float, float], ...]:
        """Finite support of ``R_positive(m)`` as ``((probability, reward), ...)``.

        Raises
        ------
        ContinuousRewardModelError
            If ``positive_reward_distribution`` is continuous, in which case
            no finite support exists.
        """
        m = self._check_attempt_number(m)
        if self.positive_reward_distribution == "deterministic":
            return ((1.0, self.positive_reward_scale(m)),)
        if self.positive_reward_distribution == "categorical":
            scale = self.positive_reward_scale(m)
            return tuple(
                (probability, scale * multiplier)
                for multiplier, probability in zip(
                    self.positive_reward_multipliers,
                    self.positive_reward_multiplier_probabilities,
                )
                if probability > 0.0
            )
        raise ContinuousRewardModelError(
            f"positive_reward_distribution={self.positive_reward_distribution!r} is "
            "continuous, so it has no finite support and cannot be enumerated exactly "
            "by transition_probabilities(). Use a 'deterministic' or 'categorical' "
            "distribution for model-based dynamic programming, or restrict yourself "
            "to sampled step()/Q-learning experiments."
        )

    def sample_positive_reward(self, m: int, rng: np.random.Generator) -> float:
        """Draw one positive mining reward for attempt ``m``.

        The result is guaranteed to be strictly positive:

        * ``deterministic``, ``categorical``, ``lognormal`` and ``gamma`` have
          positive support by construction (multipliers are validated > 0);
        * ``normal`` is unbounded below, so a non-positive draw is *resampled*
          up to ``_MAX_POSITIVE_RESAMPLES`` times and, failing that, clipped to
          ``positive_reward_floor``. Clipping is a documented last resort; it
          only bites for configurations whose ``sigma(m)`` is large relative to
          ``mu(m)``.
        """
        m = self._check_attempt_number(m)
        scale = self.positive_reward_scale(m)
        spread = self.positive_reward_spread(m)
        distribution = self.positive_reward_distribution

        if distribution in EXACT_REWARD_DISTRIBUTIONS:
            support = self.positive_reward_support(m)
            probabilities = [p for p, _ in support]
            values = [v for _, v in support]
            index = int(rng.choice(len(values), p=probabilities))
            return float(values[index])

        if distribution == "normal":
            for _ in range(_MAX_POSITIVE_RESAMPLES):
                draw = float(rng.normal(scale, spread))
                if draw > 0.0:
                    return draw
            return float(self.positive_reward_floor)

        if distribution == "lognormal":
            # Parameterised so the *median* draw is mu(m): the underlying
            # normal has mean log(mu(m)) and standard deviation sigma(m).
            return float(rng.lognormal(mean=math.log(scale), sigma=spread))

        # "gamma", matched to mean mu(m) and standard deviation sigma(m) via
        # shape k = (mu / sigma) ** 2 and scale theta = sigma ** 2 / mu.
        shape = (scale / spread) ** 2
        theta = spread**2 / scale
        return float(rng.gamma(shape=shape, scale=theta))

    def expected_positive_reward(self, m: int) -> float:
        """``E[R_positive(m)]``, for reporting/plotting only (never used by step)."""
        m = self._check_attempt_number(m)
        scale = self.positive_reward_scale(m)
        if self.positive_reward_distribution in EXACT_REWARD_DISTRIBUTIONS:
            return float(sum(p * v for p, v in self.positive_reward_support(m)))
        if self.positive_reward_distribution == "lognormal":
            return float(scale * math.exp(self.positive_reward_spread(m) ** 2 / 2.0))
        return float(scale)  # normal and gamma are parameterised by their mean

    def expected_mining_reward(self, m: int) -> float:
        """``E[reward | MINE at a node on attempt m]``, excluding ``step_reward``.

        Handy when choosing example parameters, to check that mining is a
        genuine gamble rather than a free lunch (or an obvious mistake).
        """
        p = self.positive_reward_probability(m)
        return float(
            p * self.expected_positive_reward(m)
            + (1.0 - p) * self.mining_failure_reward_at(m)
        )

    def describe_mining_schedule(self) -> str:
        """A short human-readable table of the mining gamble, attempt by attempt."""
        lines = [
            f"mining schedule ({self.positive_probability_schedule} probability, "
            f"{self.positive_reward_distribution} reward, cap C="
            f"{self.max_mining_count})",
            "  m   p(m)    E[R+(m)]   fail(m)    E[reward]",
        ]
        for m in range(1, self.max_mining_count + 2):
            label = f"{m}" if m <= self.max_mining_count else f"{m}+"
            lines.append(
                f"  {label:<3} {self.positive_reward_probability(m):<7.3f} "
                f"{self.expected_positive_reward(m):<10.3f} "
                f"{self.mining_failure_reward_at(m):<10.3f} "
                f"{self.expected_mining_reward(m):+.3f}"
            )
        return "\n".join(lines)
