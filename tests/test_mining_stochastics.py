"""The mining gamble: schedules, branch outcomes and reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from gridworld.mining_config import ContinuousRewardModelError
from gridworld.mining_env import MineWorldEnv

from conftest import MINE, RIGHT, UP, make_config, make_env, put_agent

NODE_A = (1, 1)


# ----------------------------------------------------------------------
# p_positive(m)
# ----------------------------------------------------------------------
def test_linear_probability_schedule_has_the_configured_values():
    config = make_config(
        max_mining_count=4,
        positive_probability_schedule="linear",
        initial_positive_probability=0.95,
        positive_probability_decrement=0.08,
        minimum_positive_probability=0.05,
    )
    expected = [0.95, 0.87, 0.79, 0.71, 0.63]
    assert [config.positive_reward_probability(m) for m in range(1, 6)] == pytest.approx(expected)


def test_the_probability_floor_is_respected():
    config = make_config(
        max_mining_count=9,
        positive_probability_schedule="linear",
        initial_positive_probability=0.9,
        positive_probability_decrement=0.4,
        minimum_positive_probability=0.05,
    )
    assert config.positive_reward_probability(4) == pytest.approx(0.05)
    assert config.positive_reward_probability(10) == pytest.approx(0.05)


def test_geometric_probability_schedule():
    config = make_config(
        max_mining_count=3,
        positive_probability_schedule="geometric",
        initial_positive_probability=0.8,
        positive_probability_decay=0.5,
        minimum_positive_probability=0.0,
    )
    assert [config.positive_reward_probability(m) for m in range(1, 5)] == pytest.approx(
        [0.8, 0.4, 0.2, 0.1]
    )


def test_constant_probability_schedule():
    config = make_config(
        positive_probability_schedule="constant", initial_positive_probability=0.6
    )
    values = [config.positive_reward_probability(m) for m in range(1, 4)]
    assert values == pytest.approx([0.6, 0.6, 0.6])


@pytest.mark.parametrize("schedule", ["constant", "linear", "geometric"])
def test_failure_probability_never_decreases_with_the_attempt_number(schedule):
    config = make_config(max_mining_count=6, positive_probability_schedule=schedule)
    probabilities = [config.positive_reward_probability(m) for m in range(1, 8)]
    assert all(a >= b for a, b in zip(probabilities, probabilities[1:]))


def test_attempt_numbers_outside_the_capped_range_are_rejected():
    config = make_config(max_mining_count=2)
    for bad in (0, -1, 4):
        with pytest.raises(ValueError, match="attempt number"):
            config.positive_reward_probability(bad)


# ----------------------------------------------------------------------
# R_positive(m)
# ----------------------------------------------------------------------
def test_expected_payout_never_decreases_with_the_attempt_number():
    config = make_config(max_mining_count=5)
    payouts = [config.expected_positive_reward(m) for m in range(1, 7)]
    assert all(a <= b for a, b in zip(payouts, payouts[1:]))
    assert payouts[0] < payouts[-1]


def test_deterministic_payout_has_a_single_support_point():
    config = make_config(positive_reward_distribution="deterministic")
    assert config.positive_reward_support(2) == ((1.0, config.positive_reward_scale(2)),)


def test_categorical_payout_support_sums_to_one_and_scales_with_m():
    config = make_config(
        positive_reward_distribution="categorical",
        positive_reward_multipliers=(0.5, 1.0, 2.0),
        positive_reward_multiplier_probabilities=(0.25, 0.5, 0.25),
    )
    for m in (1, 2, 3):
        support = config.positive_reward_support(m)
        assert sum(p for p, _ in support) == pytest.approx(1.0)
        assert all(value > 0.0 for _, value in support)
        assert [v for _, v in support] == pytest.approx(
            [0.5 * config.positive_reward_scale(m),
             1.0 * config.positive_reward_scale(m),
             2.0 * config.positive_reward_scale(m)]
        )
    # Every support point grows with m.
    first = [v for _, v in config.positive_reward_support(1)]
    second = [v for _, v in config.positive_reward_support(2)]
    assert all(a < b for a, b in zip(first, second))


@pytest.mark.parametrize("distribution", ["normal", "lognormal", "gamma"])
def test_continuous_payouts_have_no_finite_support(distribution):
    config = make_config(
        positive_reward_distribution=distribution, positive_reward_base_std=0.3
    )
    with pytest.raises(ContinuousRewardModelError):
        config.positive_reward_support(1)


@pytest.mark.parametrize("distribution", ["deterministic", "categorical", "normal", "lognormal", "gamma"])
def test_sampled_payouts_are_always_strictly_positive(distribution):
    config = make_config(
        positive_reward_distribution=distribution,
        positive_reward_base_std=0.5,
        positive_reward_base_mean=0.5,
    )
    rng = np.random.default_rng(0)
    draws = [config.sample_positive_reward(1, rng) for _ in range(500)]
    assert all(draw > 0.0 for draw in draws)


def test_sampled_payout_distribution_shifts_upward_with_m():
    config = make_config(
        max_mining_count=3,
        positive_reward_distribution="normal",
        positive_reward_base_mean=2.0,
        positive_reward_mean_increment=3.0,
        positive_reward_base_std=0.2,
    )
    rng = np.random.default_rng(0)
    means = [
        float(np.mean([config.sample_positive_reward(m, rng) for _ in range(2000)]))
        for m in (1, 2, 3)
    ]
    assert means[0] == pytest.approx(2.0, abs=0.05)
    assert means[1] == pytest.approx(5.0, abs=0.05)
    assert means[2] == pytest.approx(8.0, abs=0.05)


# ----------------------------------------------------------------------
# Outcome branches in the live environment
# ----------------------------------------------------------------------
def test_a_certain_success_is_never_terminal():
    env = make_env(initial_positive_probability=1.0, minimum_positive_probability=1.0)
    put_agent(env, NODE_A, (0, 0))
    for _ in range(3):
        _obs, reward, terminated, _truncated, info = env.step(MINE)
        assert info["mining_success"] and not terminated
        assert reward > 0.0


def test_a_certain_failure_terminates_with_the_configured_penalty():
    env = make_env(initial_positive_probability=0.0, minimum_positive_probability=0.0)
    put_agent(env, NODE_A, (0, 0))
    _obs, reward, terminated, _truncated, info = env.step(MINE)
    assert terminated and info["mining_failure"]
    assert reward == pytest.approx(env.config.step_reward + env.config.mining_failure_reward)


def test_reported_success_probability_matches_the_schedule():
    env = make_env()
    put_agent(env, NODE_A, (0, 0))
    _obs, _reward, _terminated, _truncated, info = env.step(MINE)
    assert info["positive_reward_probability"] == pytest.approx(
        env.config.positive_reward_probability(1)
    )


def test_empirical_success_rate_matches_p_of_m():
    """Many independent first attempts should succeed at rate p(1)."""
    config = make_config(initial_positive_probability=0.7, minimum_positive_probability=0.7)
    successes = 0
    trials = 3000
    env = MineWorldEnv(config)
    env.reset(seed=12345)
    for _ in range(trials):
        put_agent(env, NODE_A, (0, 0))
        _obs, _reward, _terminated, _truncated, info = env.step(MINE)
        successes += bool(info["mining_success"])
    assert successes / trials == pytest.approx(0.7, abs=0.03)


def test_failure_becomes_more_likely_at_higher_attempt_numbers():
    config = make_config(
        max_mining_count=3,
        initial_positive_probability=0.9,
        positive_probability_decrement=0.3,
        minimum_positive_probability=0.0,
    )
    env = MineWorldEnv(config)
    env.reset(seed=7)
    rates = []
    for counts in [(0, 0), (1, 0), (2, 0)]:
        failures = 0
        trials = 2000
        for _ in range(trials):
            put_agent(env, NODE_A, counts)
            _obs, _reward, _terminated, _truncated, info = env.step(MINE)
            failures += bool(info["mining_failure"])
        rates.append(failures / trials)
    assert rates[0] < rates[1] < rates[2]
    assert rates == pytest.approx([0.1, 0.4, 0.7], abs=0.04)


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def _rollout(env, actions):
    env.reset(seed=99)
    trace = []
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        trace.append((obs, reward, terminated, truncated, info["actual_action"]))
        if terminated or truncated:
            break
    return trace


@pytest.mark.parametrize("distribution", ["deterministic", "categorical", "lognormal"])
def test_the_same_seed_reproduces_slips_outcomes_and_payouts(distribution):
    actions = [UP, RIGHT, MINE, MINE, UP, MINE, RIGHT, MINE, MINE, MINE]
    kwargs = dict(
        slip_probability=0.25,
        positive_reward_distribution=distribution,
        positive_reward_base_std=0.4,
    )
    first = _rollout(make_env(**kwargs), actions)
    second = _rollout(make_env(**kwargs), actions)
    assert first == second


def test_different_seeds_give_different_rollouts():
    actions = [MINE] * 10
    env_a, env_b = make_env(slip_probability=0.3), make_env(slip_probability=0.3)
    put_agent(env_a, NODE_A, (0, 0))
    env_a.reset(seed=1)
    env_b.reset(seed=2)
    put_agent(env_a, NODE_A, (0, 0))
    put_agent(env_b, NODE_A, (0, 0))
    trace_a = [env_a.step(a)[1] for a in actions[:4]]
    trace_b = [env_b.step(a)[1] for a in actions[:4]]
    assert trace_a != trace_b
