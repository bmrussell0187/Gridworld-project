"""Shared fixtures and helpers for the GridWorld / MineWorld test-suite."""

from __future__ import annotations

import itertools

import pytest

from gridworld.mining_config import MineWorldConfig
from gridworld.mining_env import MineWorldEnv

# Action encoding, repeated here so the tests assert against literal values
# rather than importing whatever the implementation happens to define.
UP, RIGHT, DOWN, LEFT, MINE = 0, 1, 2, 3, 4


def make_config(**overrides) -> MineWorldConfig:
    """A small, fully-specified MineWorld config that tests can tweak.

    4x4 grid, two mining nodes at (1, 1) and (3, 3), one terminal goal, a
    deterministic payout and no slip -- i.e. everything is exactly
    predictable unless a test asks for otherwise.
    """
    defaults = dict(
        width=4,
        height=4,
        start=(0, 0),
        terminal_states={(3, 0): 1.0},
        rewards={},
        walls=set(),
        step_reward=-0.1,
        invalid_move_reward=-0.5,
        slip_probability=0.0,
        max_steps=50,
        seed=0,
        mining_nodes={(1, 1), (3, 3)},
        max_mining_count=2,
        positive_probability_schedule="linear",
        initial_positive_probability=0.9,
        positive_probability_decrement=0.3,
        minimum_positive_probability=0.1,
        positive_reward_distribution="deterministic",
        positive_reward_base_mean=1.0,
        positive_reward_mean_increment=2.0,
        mining_failure_reward=-5.0,
    )
    defaults.update(overrides)
    return MineWorldConfig(**defaults)


def make_env(**overrides) -> MineWorldEnv:
    """A MineWorldEnv built from :func:`make_config`, already reset."""
    env = MineWorldEnv(make_config(**overrides))
    env.reset(seed=0)
    return env


def always_succeeds(**overrides) -> MineWorldEnv:
    """An env whose mining always pays out (p(m) = 1 for every m)."""
    return make_env(
        initial_positive_probability=1.0,
        minimum_positive_probability=1.0,
        positive_probability_decrement=0.0,
        **overrides,
    )


def always_fails(**overrides) -> MineWorldEnv:
    """An env whose mining always collapses (p(m) = 0 for every m)."""
    return make_env(
        initial_positive_probability=0.0,
        minimum_positive_probability=0.0,
        **overrides,
    )


def put_agent(env: MineWorldEnv, position, counts) -> None:
    """Force the environment into a chosen state.

    Tests need to probe states that are awkward to reach by walking; this
    writes the two state components directly, exactly as ``step`` would.
    """
    env._agent_pos = tuple(position)
    env._mining_counts = tuple(counts)


def all_component_pairs(env: MineWorldEnv):
    """Every valid ``(position, mining_counts)`` pair of an environment."""
    cells = [(x, y) for y in range(env.height) for x in range(env.width)]
    count_vectors = itertools.product(
        range(env.config.n_mining_levels), repeat=env.n_mining_nodes
    )
    return itertools.product(cells, count_vectors)


def env_state_snapshot(env: MineWorldEnv):
    """Everything a model query must leave untouched."""
    return (
        env._agent_pos,
        env._mining_counts,
        env._step_count,
        env._rng.bit_generator.state,
        {k: list(v) for k, v in env.trajectory.items()},
    )


@pytest.fixture
def env() -> MineWorldEnv:
    return make_env()
