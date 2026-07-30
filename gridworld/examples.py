"""A handful of ready-made GridWorld MDPs for dissertation experiments.

Each ``make_*`` function returns a fully-constructed :class:`GridWorldEnv`
(or, for the mining variants, a :class:`MineWorldEnv`). They are deliberately
small and easy to reason about by hand, so that a student can verify
dynamic-programming / Q-learning results against intuition (or against a
hand-drawn transition diagram).

Grid coordinate convention: (x, y), with (0, 0) at the bottom-left and y
increasing upwards -- so "top-right" means large x, large y.
"""

from __future__ import annotations

from .config import GridWorldConfig
from .env import GridWorldEnv
from .mining_config import MineWorldConfig
from .mining_env import MineWorldEnv


def make_easy_gridworld(seed: int | None = 0, render_mode: str | None = None) -> GridWorldEnv:
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
    return GridWorldEnv(config, render_mode=render_mode)


def make_walled_gridworld(seed: int | None = 0, render_mode: str | None = None) -> GridWorldEnv:
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
    return GridWorldEnv(config, render_mode=render_mode)


def make_slippery_gridworld(seed: int | None = 0, render_mode: str | None = None) -> GridWorldEnv:
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
    return GridWorldEnv(config, render_mode=render_mode)


def make_reward_shaping_gridworld(seed: int | None = 0, render_mode: str | None = None) -> GridWorldEnv:
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
    return GridWorldEnv(config, render_mode=render_mode)


def make_maze_gridworld(seed: int | None = 0, render_mode: str | None = None) -> GridWorldEnv:
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
        rewards={},
        walls=walls,
        step_reward=-0.01,
        invalid_move_reward=0,
        slip_probability=0.0,
        max_steps=200,
        seed=seed,
    )

    return GridWorldEnv(config, render_mode=render_mode)


# ----------------------------------------------------------------------
# MineWorld: GridWorld + a MINE action with per-node memory
# ----------------------------------------------------------------------
def make_mining_gridworld(
    seed: int | None = 0, render_mode: str | None = None
) -> MineWorldEnv:
    """Example 6: a 5x5 mining world with two independent mining nodes.

    Layout::

        start (0, 0)      goal (4, 4) = +1.0     trap (4, 0) = -1.0
        mining nodes at (1, 3) and (3, 1)

    Mining schedule (linear probability decay, deterministic payout, cap C=3,
    failure = -10.0). ``m`` is the attempt number *at that node*; because
    counts are capped at C=3, ``m = 4`` repeats forever once a node has been
    mined three times::

        m   p(m)    R+(m)    E[reward of the attempt]
        1   0.95     1.0      0.95*1.0 + 0.05*(-10) = +0.45
        2   0.80     2.5      0.80*2.5 + 0.20*(-10) =  0.00
        3   0.65     4.0      0.65*4.0 + 0.35*(-10) = -0.90
        4+  0.50     5.5      0.50*5.5 + 0.50*(-10) = -2.25

    This is a genuine risk/reward decision rather than a free lunch: the
    first attempt at a fresh node is clearly worth taking, the second is
    exactly break-even in immediate terms (and therefore *negative* once the
    forgone goal reward and the step cost are discounted in), and everything
    beyond that destroys value. An optimal agent mines each node once and
    then walks to the goal.

    The payout distribution is ``"deterministic"``, so the MDP has finite
    reward support and ``transition_probabilities`` can enumerate it exactly:
    this example is solvable by value/policy iteration. The state space is
    5 * 5 * 4**2 = 400 states.
    """
    config = MineWorldConfig(
        width=5,
        height=5,
        start=(0, 0),
        terminal_states={(4, 4): 1.0, (4, 0): -1.0},
        rewards={},
        walls=set(),
        step_reward=-0.05,
        invalid_move_reward=-0.10,
        slip_probability=0.0,
        max_steps=100,
        seed=seed,
        mining_nodes={(1, 3), (3, 1)},
        max_mining_count=3,
        positive_probability_schedule="linear",
        initial_positive_probability=0.95,
        positive_probability_decrement=0.15,
        minimum_positive_probability=0.05,
        positive_reward_distribution="deterministic",
        positive_reward_base_mean=1.0,
        positive_reward_mean_increment=1.5,
        mining_failure_reward=-10.0,
    )
    return MineWorldEnv(config, render_mode=render_mode)


def make_risky_mining_gridworld(
    seed: int | None = 0, render_mode: str | None = None
) -> MineWorldEnv:
    """Example 7: the mining world with a random payout *and* slippery moves.

    Same layout and probability schedule as :func:`make_mining_gridworld`,
    but two extra sources of randomness:

    * the payout of a successful mine is ``"categorical"`` -- it pays
      0.5x, 1.0x or 1.5x of ``mu(m) = 1.0 + (m - 1) * 1.5`` with
      probabilities 0.25 / 0.5 / 0.25, so the expected payout matches the
      deterministic example while the variance grows with ``m``;
    * ``slip_probability = 0.1``, so one action in ten is replaced by a
      uniformly random *different* action -- which can turn an intended move
      into an unintended ``MINE`` (and vice versa).

    The categorical payout still has finite support, so this example remains
    exactly solvable by dynamic programming; it is the interesting middle
    ground between a deterministic teaching example and a fully continuous
    reward model.
    """
    config = MineWorldConfig(
        width=5,
        height=5,
        start=(0, 0),
        terminal_states={(4, 4): 1.0, (4, 0): -1.0},
        rewards={},
        walls=set(),
        step_reward=-0.05,
        invalid_move_reward=-0.10,
        slip_probability=0.1,
        max_steps=100,
        seed=seed,
        mining_nodes={(1, 3), (3, 1)},
        max_mining_count=3,
        positive_probability_schedule="linear",
        initial_positive_probability=0.95,
        positive_probability_decrement=0.15,
        minimum_positive_probability=0.05,
        positive_reward_distribution="categorical",
        positive_reward_base_mean=1.0,
        positive_reward_mean_increment=1.5,
        positive_reward_multipliers=(0.5, 1.0, 1.5),
        positive_reward_multiplier_probabilities=(0.25, 0.5, 0.25),
        mining_failure_reward=-10.0,
    )
    return MineWorldEnv(config, render_mode=render_mode)


def make_continuous_mining_gridworld(
    seed: int | None = 0, render_mode: str | None = None
) -> MineWorldEnv:
    """Example 8: mining with a *continuous* (log-normal) payout.

    Identical to :func:`make_mining_gridworld` apart from the payout law: a
    successful m-th mine pays ``LogNormal(log mu(m), sigma(m))``, whose
    median is ``mu(m)`` and whose spread widens with ``m``.

    This example is **simulation-only**: a continuous reward distribution has
    no finite support, so ``transition_probabilities`` raises
    :class:`~gridworld.mining_config.ContinuousRewardModelError` rather than
    silently substituting an expectation. Use it with model-free methods
    (``q_learning``, Stable-Baselines3) and contrast it with the exactly
    solvable examples above.
    """
    config = MineWorldConfig(
        width=5,
        height=5,
        start=(0, 0),
        terminal_states={(4, 4): 1.0, (4, 0): -1.0},
        rewards={},
        walls=set(),
        step_reward=-0.05,
        invalid_move_reward=-0.10,
        slip_probability=0.0,
        max_steps=100,
        seed=seed,
        mining_nodes={(1, 3), (3, 1)},
        max_mining_count=3,
        positive_probability_schedule="linear",
        initial_positive_probability=0.95,
        positive_probability_decrement=0.15,
        minimum_positive_probability=0.05,
        positive_reward_distribution="lognormal",
        positive_reward_base_mean=1.0,
        positive_reward_mean_increment=1.5,
        positive_reward_base_std=0.3,
        positive_reward_std_increment=0.05,
        mining_failure_reward=-10.0,
    )
    return MineWorldEnv(config, render_mode=render_mode)

def make_deep_mining_gridworld(
    seed: int | None = 0, render_mode: str | None = None
) -> MineWorldEnv:
    
    config = MineWorldConfig(
        width=7,
        height=7,
        start=(0, 0),
        terminal_states={(6, 6): 10},
        rewards={},
        walls=set(),
        step_reward=-0.01,
        invalid_move_reward=-0.05,
        slip_probability=0.0,
        max_steps=200,
        seed=seed,
        mining_nodes={(2, 4), (5, 1)},
        max_mining_count=19,
        positive_probability_schedule="linear",
        initial_positive_probability=1,
        positive_probability_decrement=0.03,
        minimum_positive_probability=0.00,
        positive_reward_distribution="deterministic",
        positive_reward_base_mean=10,
        positive_reward_mean_increment=20,
        mining_failure_reward=-100.0,
        mining_failure_reward_proportion=1.0,
        
    )
    return MineWorldEnv(config, render_mode=render_mode)

    