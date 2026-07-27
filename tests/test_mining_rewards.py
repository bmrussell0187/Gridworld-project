"""Mining rewards must depend on the *action*, never on the cell entered.

The whole point of the MINE action is that walking over a mining node is
worthless: only executing MINE while standing on one pays out. These tests
pin that down, including the slippery case where the executed action is not
the one the agent chose.
"""

from __future__ import annotations

import pytest

from conftest import DOWN, LEFT, MINE, RIGHT, UP, always_fails, always_succeeds, put_agent

NODE = (1, 1)  # a mining node in the shared test config


def walk_to_node(env):
    """Move start -> (1, 1) with two ordinary moves, returning the rewards."""
    _obs, reward_up, *_ = _step(env, UP)
    _obs, reward_right, *_ = _step(env, RIGHT)
    return [reward_up, reward_right]


def _step(env, action):
    return env.step(action)


def test_moving_onto_a_mining_node_pays_only_the_step_reward():
    env = always_succeeds()
    rewards = walk_to_node(env)
    assert rewards == [pytest.approx(env.config.step_reward)] * 2
    assert env._agent_pos == NODE
    assert env._mining_counts == (0, 0)  # entering is not an attempt


def test_leaving_and_re_entering_a_node_never_pays_a_mining_reward():
    env = always_succeeds()
    walk_to_node(env)
    for action in (DOWN, UP, LEFT, RIGHT, DOWN, UP):
        _obs, reward, terminated, _truncated, info = env.step(action)
        assert reward == pytest.approx(env.config.step_reward)
        assert not info["mining_attempted"]
        assert not terminated
    assert env._agent_pos == NODE
    assert env._mining_counts == (0, 0)


def test_mine_on_a_node_invokes_the_mining_distribution():
    env = always_succeeds()
    walk_to_node(env)
    _obs, reward, terminated, _truncated, info = env.step(MINE)

    expected = env.config.step_reward + env.config.positive_reward_scale(1)
    assert reward == pytest.approx(expected)
    assert info["mining_attempted"] and info["mining_success"]
    assert not info["mining_failure"]
    assert info["mining_node"] == NODE
    assert info["mining_attempt_number"] == 1
    assert not terminated


def test_mine_away_from_a_node_pays_the_ordinary_step_reward():
    env = always_succeeds()
    _obs, reward, terminated, _truncated, info = env.step(MINE)  # still at (0, 0)

    assert reward == pytest.approx(env.config.step_reward)
    assert not info["mining_attempted"]
    assert info["mining_node"] is None
    assert not terminated
    assert env._agent_pos == (0, 0)      # MINE never moves the agent
    assert env._mining_counts == (0, 0)  # and never touches the counts


def test_mine_away_from_a_node_can_have_its_own_penalty():
    env = always_succeeds(invalid_mining_reward=-0.75)
    _obs, reward, _terminated, _truncated, info = env.step(MINE)
    assert reward == pytest.approx(-0.75)
    assert not info["mining_attempted"]


def test_mine_does_not_re_trigger_a_shaping_reward():
    # (1, 1) is both a mining node and a shaped cell: entering it pays the
    # shaping reward, but MINEing on it must not pay it again.
    env = always_succeeds(rewards={NODE: 0.25})
    _obs, _reward, *_ = env.step(UP)
    _obs, entry_reward, *_ = env.step(RIGHT)
    assert entry_reward == pytest.approx(env.config.step_reward + 0.25)

    _obs, mine_reward, *_ = env.step(MINE)
    assert mine_reward == pytest.approx(
        env.config.step_reward + env.config.positive_reward_scale(1)
    )


def test_a_failed_mine_pays_the_configured_penalty_and_terminates():
    env = always_fails()
    walk_to_node(env)
    _obs, reward, terminated, truncated, info = env.step(MINE)

    expected = env.config.step_reward + env.config.mining_failure_reward_at(1)
    assert reward == pytest.approx(expected)
    assert terminated and not truncated
    assert info["mining_failure"] and not info["mining_success"]
    # The collapse is signalled by the flag, not by turning the cell terminal.
    assert env._agent_pos == NODE
    assert NODE not in env.config.terminal_states


def test_an_ordinary_terminal_cell_still_terminates_normally():
    env = always_succeeds()
    for action in (RIGHT, RIGHT):
        env.step(action)
    _obs, reward, terminated, _truncated, info = env.step(RIGHT)  # onto (3, 0)
    assert terminated
    assert reward == pytest.approx(env.config.step_reward + 1.0)
    assert not info["mining_attempted"]


def test_an_invalid_move_pays_the_invalid_move_reward_and_does_not_mine():
    env = always_succeeds()
    put_agent(env, NODE, (0, 0))
    walls_env = always_succeeds(walls={(1, 2)})
    put_agent(walls_env, NODE, (0, 0))

    _obs, reward, _terminated, _truncated, info = walls_env.step(UP)  # into the wall
    assert reward == pytest.approx(walls_env.config.invalid_move_reward)
    assert info["invalid_move"]
    assert not info["mining_attempted"]
    assert walls_env._agent_pos == NODE
    assert walls_env._mining_counts == (0, 0)
    assert env._mining_counts == (0, 0)


def test_slip_into_mine_counts_as_mining(monkeypatch):
    """The *actual* action decides, so a slip onto MINE really does mine."""
    env = always_succeeds(slip_probability=0.5)
    put_agent(env, NODE, (0, 0))
    monkeypatch.setattr(env, "_maybe_slip", lambda action: MINE)

    _obs, reward, _terminated, _truncated, info = env.step(UP)  # intended: move
    assert info["intended_action"] == UP
    assert info["actual_action"] == MINE
    assert info["mining_attempted"] and info["mining_success"]
    assert reward == pytest.approx(
        env.config.step_reward + env.config.positive_reward_scale(1)
    )
    assert env._mining_counts == (1, 0)


def test_slip_out_of_mine_does_not_count_as_mining(monkeypatch):
    """Conversely, an intended MINE that slips into a move must not mine."""
    env = always_succeeds(slip_probability=0.5)
    put_agent(env, NODE, (0, 0))
    monkeypatch.setattr(env, "_maybe_slip", lambda action: UP)

    _obs, reward, _terminated, _truncated, info = env.step(MINE)
    assert info["intended_action"] == MINE
    assert info["actual_action"] == UP
    assert not info["mining_attempted"]
    assert reward == pytest.approx(env.config.step_reward)
    assert env._agent_pos == (1, 2)
    assert env._mining_counts == (0, 0)
