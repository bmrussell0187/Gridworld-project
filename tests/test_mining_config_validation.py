"""Misconfigured mining worlds must fail loudly, not silently."""

from __future__ import annotations

import warnings

import pytest

from gridworld.mining_config import MAX_STATE_SPACE_SIZE, MineWorldConfig

from conftest import make_config


def test_a_mining_node_outside_the_grid_is_rejected():
    with pytest.raises(ValueError, match="outside the grid"):
        make_config(mining_nodes={(9, 9)})


def test_a_mining_node_on_a_wall_is_rejected():
    with pytest.raises(ValueError, match="cannot be a wall"):
        make_config(mining_nodes={(1, 1)}, walls={(1, 1)})


def test_a_mining_node_on_an_ordinary_terminal_state_is_rejected():
    with pytest.raises(ValueError, match="cannot also be an ordinary terminal state"):
        make_config(mining_nodes={(3, 0)})  # (3, 0) is the goal


def test_duplicate_mining_nodes_are_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        make_config(mining_nodes=[(1, 1), (1, 1)])


def test_an_iterable_of_unique_nodes_is_accepted_and_normalised():
    config = make_config(mining_nodes=[(3, 3), (1, 1)])
    assert config.mining_nodes == {(1, 1), (3, 3)}
    assert config.mining_nodes_ordered == ((1, 1), (3, 3))  # sorted, deterministic


def test_a_negative_cap_is_rejected():
    with pytest.raises(ValueError, match="max_mining_count"):
        make_config(max_mining_count=-1)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"positive_probability_schedule": "quadratic"}, "positive_probability_schedule"),
        ({"initial_positive_probability": 1.5}, "initial_positive_probability"),
        ({"minimum_positive_probability": -0.1}, "minimum_positive_probability"),
        ({"initial_positive_probability": 0.1, "minimum_positive_probability": 0.5}, "must not exceed"),
        ({"positive_probability_decrement": -0.1}, "non-increasing"),
        ({"positive_probability_decay": 0.0}, "positive_probability_decay"),
        ({"positive_probability_decay": 1.5}, "positive_probability_decay"),
    ],
)
def test_invalid_probability_schedules_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_config(**overrides)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"positive_reward_distribution": "cauchy"}, "positive_reward_distribution"),
        ({"positive_reward_base_mean": 0.0}, "positive_reward_base_mean"),
        ({"positive_reward_mean_increment": -1.0}, "grows with the attempt number"),
        ({"positive_reward_base_std": -0.1}, "standard deviations"),
        ({"positive_reward_floor": 0.0}, "positive_reward_floor"),
        ({"mining_failure_reward": 1.0}, "mining_failure_reward"),
        ({"mining_failure_reward_increment": -1.0}, "mining_failure_reward_increment"),
        ({"positive_reward_distribution": "gamma", "positive_reward_base_std": 0.0}, "requires"),
        ({"positive_reward_distribution": "lognormal", "positive_reward_base_std": 0.0}, "requires"),
    ],
)
def test_invalid_reward_parameters_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_config(**overrides)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"positive_reward_multipliers": (), "positive_reward_multiplier_probabilities": ()},
            "must not be empty",
        ),
        (
            {"positive_reward_multipliers": (1.0, 2.0), "positive_reward_multiplier_probabilities": (1.0,)},
            "equal length",
        ),
        (
            {"positive_reward_multipliers": (0.0, 1.0), "positive_reward_multiplier_probabilities": (0.5, 0.5)},
            "must all be > 0",
        ),
        (
            {"positive_reward_multipliers": (1.0, 2.0), "positive_reward_multiplier_probabilities": (0.5, 0.9)},
            "must sum to 1",
        ),
    ],
)
def test_invalid_categorical_supports_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_config(positive_reward_distribution="categorical", **overrides)


def test_state_space_size_formula_and_exactness_flag():
    config = make_config(mining_nodes={(1, 1), (3, 3)}, max_mining_count=2)
    assert config.state_space_size == 4 * 4 * 3**2
    assert config.n_mining_levels == 3
    assert config.is_exact_reward_model
    assert not make_config(
        positive_reward_distribution="lognormal", positive_reward_base_std=0.3
    ).is_exact_reward_model


# Every cell of the 4x4 test grid except the goal, which may not be a node.
MANY_NODES = {(x, y) for x in range(4) for y in range(4)} - {(3, 0)}


def test_an_impractically_large_state_space_is_rejected():
    """The state space grows exponentially in the number of mining nodes."""
    with pytest.raises(ValueError, match="exceeds MAX_STATE_SPACE_SIZE"):
        make_config(mining_nodes=MANY_NODES, max_mining_count=3)  # 16 * 4**15
    assert MAX_STATE_SPACE_SIZE > 0


def test_a_merely_large_state_space_warns():
    with pytest.warns(UserWarning, match="exponentially"):
        config = make_config(mining_nodes=MANY_NODES, max_mining_count=1)  # 16 * 2**15
    assert 100_000 < config.state_space_size < MAX_STATE_SPACE_SIZE


def test_small_configurations_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_config()


def test_the_schedule_description_covers_every_attempt_level():
    config = make_config(max_mining_count=2)
    lines = config.describe_mining_schedule().splitlines()
    # A title, a column header, and one row per attempt level m in 1..C+1.
    assert len(lines) == 2 + (config.max_mining_count + 1)
    assert lines[-1].strip().startswith("3+")  # the saturated level is flagged


def test_mineworld_config_is_a_gridworld_config():
    """MineWorldConfig extends GridWorldConfig, so the grid checks still run."""
    from gridworld.config import GridWorldConfig

    assert issubclass(MineWorldConfig, GridWorldConfig)
    with pytest.raises(ValueError, match="start"):
        make_config(start=(9, 9))
