"""Regression tests for the original GridWorld, which MineWorld extends.

These cover the parts of the existing code the mining work touched:
``dynamic_programming`` (which now honours the ``terminated`` flag and asks
the environment for its skip states), the plotting helpers, and the Gymnasium
registration.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import gridworld
from gridworld.dynamic_programming import policy_iteration, value_iteration
from gridworld.env import GridWorldEnv
from gridworld.examples import make_easy_gridworld, make_slippery_gridworld
from gridworld.q_learning import q_learning


def test_gridworld_transition_distributions_sum_to_one():
    for env in (make_easy_gridworld(), make_slippery_gridworld()):
        for state in range(env.n_states):
            for action in range(env.action_space.n):
                total = sum(p for p, _s, _r, _t in env.transition_probabilities(state, action))
                assert total == pytest.approx(1.0)


def test_gridworld_coordinate_round_trip():
    env = make_easy_gridworld()
    for state in range(env.n_states):
        assert env.coord_to_state(env.state_to_coord(state)) == state


def test_skip_state_indices_are_the_terminal_and_wall_cells():
    env = make_easy_gridworld()
    expected = {env.coord_to_state(c) for c in env.config.terminal_states}
    assert env.skip_state_indices() == expected


def test_value_iteration_still_solves_the_easy_gridworld():
    env = make_easy_gridworld()
    gamma = 0.95
    V, policy, _history = value_iteration(env, gamma=gamma)

    # The cell next to the goal should be worth close to the goal reward.
    below_goal = env.coord_to_state((4, 3))
    assert V[below_goal] == pytest.approx(env.config.step_reward + 1.0, abs=1e-6)
    # ... and its greedy action should be "up", straight into the goal.
    assert policy[below_goal] == 0
    assert V[env.coord_to_state((0, 0))] > 0.0


def test_policy_iteration_agrees_with_value_iteration_on_a_slippery_grid():
    gamma = 0.95
    V_vi, _pi, _ = value_iteration(make_slippery_gridworld(), gamma=gamma)
    V_pi, _pi2, _ = policy_iteration(make_slippery_gridworld(), gamma=gamma)
    assert np.max(np.abs(V_vi - V_pi)) < 1e-6


def test_q_learning_learns_a_positive_return_on_the_easy_gridworld():
    env = make_easy_gridworld()
    _Q, _policy, returns, _lengths = q_learning(env, episodes=400, seed=0)
    assert float(np.mean(returns[-50:])) > 0.5


def test_q_learning_runs_on_a_mining_world():
    from gridworld.examples import make_mining_gridworld

    env = make_mining_gridworld(seed=0)
    Q, policy, returns, lengths = q_learning(env, episodes=400, seed=0)
    assert Q.shape == (env.n_states, 5)
    assert policy.shape == (env.n_states,)
    assert len(returns) == len(lengths) == 400
    assert float(np.mean(returns[-50:])) > 0.0


def test_step_rejects_an_action_outside_the_action_space():
    env = make_easy_gridworld()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(9)


def test_mineworld_step_rejects_an_action_outside_the_action_space():
    from conftest import make_env

    env = make_env()
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(5)


# ----------------------------------------------------------------------
# Gymnasium registration
# ----------------------------------------------------------------------
def test_every_example_id_is_registered():
    for env_id in gridworld.ENVIRONMENT_IDS:
        assert env_id in gym.registry


def test_registering_twice_is_harmless():
    gridworld.register_envs()
    gridworld.register_envs()
    assert "MineWorld-v0" in gym.registry


@pytest.mark.parametrize("env_id", ["GridWorld-v0", "MineWorld-v0", "MineWorldRisky-v0"])
def test_gym_make_produces_a_working_environment(env_id):
    env = gym.make(env_id)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    env.close()


def test_gym_make_accepts_a_seed_and_render_mode():
    env = gym.make("MineWorld-v0", seed=7, render_mode="rgb_array")
    env.reset(seed=7)
    image = env.render()
    assert isinstance(image, np.ndarray) and image.ndim == 3
    env.close()


def test_mineworld_action_space_has_five_actions():
    env = gym.make("MineWorld-v0")
    assert env.action_space.n == 5
    assert isinstance(env.unwrapped, gridworld.MineWorldEnv)
    env.close()


def test_gridworld_env_is_untouched_by_the_mining_extension():
    """MineWorldEnv is a sibling of GridWorldEnv, not a replacement."""
    assert not issubclass(gridworld.MineWorldEnv, GridWorldEnv)
    assert make_easy_gridworld().action_space.n == 4
