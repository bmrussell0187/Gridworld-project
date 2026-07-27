"""Rendering, plotting and animation for both GridWorld and MineWorld."""

from __future__ import annotations

import numpy as np
import pytest

import matplotlib.pyplot as plt

from gridworld.animation import animate_episode
from gridworld.dynamic_programming import value_iteration
from gridworld.examples import make_easy_gridworld, make_mining_gridworld
from gridworld.plotting import (
    add_mining_count_labels,
    default_mining_counts,
    mining_nodes_of,
    plot_gridworld,
    plot_mining_schedule,
    plot_policy,
    plot_q_values,
    plot_value_function,
    state_index_of,
)

from conftest import MINE, UP, env_state_snapshot, make_env, put_agent

NODE_A = (1, 1)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_rgb_array_render_has_the_expected_shape_and_dtype():
    env = make_mining_gridworld(seed=0, render_mode="rgb_array")
    env.reset(seed=0)
    image = env.render()
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8
    assert image.shape[0] > 0 and image.shape[1] > 0


def test_render_works_before_and_after_mining_and_changes_the_picture():
    env = make_env()
    env.render_mode = "rgb_array"
    put_agent(env, NODE_A, (0, 0))
    before = env.render()

    env.step(MINE)
    after = env.render()

    assert before.shape == after.shape
    # The count label next to the node has changed, so the images must differ.
    assert not np.array_equal(before, after)


def test_render_does_not_mutate_the_environment():
    env = make_env()
    env.render_mode = "rgb_array"
    put_agent(env, NODE_A, (1, 0))
    snapshot = env_state_snapshot(env)
    env.render()
    env.render()
    assert env_state_snapshot(env) == snapshot


def test_render_returns_none_without_a_render_mode():
    env = make_env()
    assert env.render() is None


def test_plotting_helpers_handle_both_environment_types():
    grid = make_easy_gridworld()
    mine = make_mining_gridworld()

    assert mining_nodes_of(grid) == ()
    assert default_mining_counts(grid) == ()
    assert state_index_of(grid, (2, 3)) == grid.coord_to_state((2, 3))

    assert mining_nodes_of(mine) == mine.mining_nodes
    assert default_mining_counts(mine) == (0, 0)
    assert state_index_of(mine, (2, 3), (1, 2)) == mine.components_to_state((2, 3), (1, 2))
    # Omitting the counts means "nothing mined yet".
    assert state_index_of(mine, (2, 3)) == mine.components_to_state((2, 3), (0, 0))


def test_plot_gridworld_marks_mining_nodes_and_their_counts():
    env = make_mining_gridworld()
    ax = plot_gridworld(env, mining_counts=(2, 1))
    labels = [t.get_text() for t in ax.texts]
    assert "2" in labels and "1" in labels

    plain = plot_gridworld(make_easy_gridworld())
    assert plain is not None  # a plain gridworld draws no mining decoration


def test_plot_functions_accept_a_mining_count_slice():
    env = make_mining_gridworld()
    V, policy, _history = value_iteration(env, gamma=0.95)
    Q = np.zeros((env.n_states, env.action_space.n))

    for counts in [(0, 0), (1, 0), (3, 3)]:
        assert plot_policy(env, policy, mining_counts=counts) is not None
        assert plot_value_function(env, V, mining_counts=counts) is not None
        assert plot_q_values(env, Q, mining_counts=counts) is not None


def test_plot_functions_still_work_for_a_plain_gridworld():
    env = make_easy_gridworld()
    V, policy, _history = value_iteration(env, gamma=0.95)
    Q = np.zeros((env.n_states, env.action_space.n))
    assert plot_policy(env, policy) is not None
    assert plot_value_function(env, V) is not None
    assert plot_q_values(env, Q) is not None


def test_plot_policy_draws_a_marker_for_the_mine_action():
    env = make_mining_gridworld()
    policy = np.full(env.n_states, MINE)
    ax = plot_policy(env, policy, mining_counts=(0, 0))
    markers = [line.get_marker() for line in ax.lines]
    assert "X" in markers  # the MINE action is a cross, not an arrow


def test_plot_mining_schedule_runs():
    env = make_mining_gridworld()
    ax = plot_mining_schedule(env.config)
    assert ax is not None


def test_count_labels_can_be_created_empty_for_animation():
    env = make_mining_gridworld()
    _fig, ax = plt.subplots()
    labels = add_mining_count_labels(env, ax, show_counts=False)
    assert set(labels) == set(env.mining_nodes)
    assert all(label.get_text() == "" for label in labels.values())


# ----------------------------------------------------------------------
# Animation
# ----------------------------------------------------------------------
def test_trajectory_has_one_mining_count_snapshot_per_frame():
    env = make_env()
    env.reset(seed=0)
    put_agent(env, NODE_A, (0, 0))
    for action in (MINE, UP, MINE, UP):
        _obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break

    trajectory = env.trajectory
    assert len(trajectory["mining_counts"]) == len(trajectory["positions"])
    assert trajectory["mining_counts"][0] == (0, 0)  # the reset frame
    assert all(isinstance(counts, tuple) for counts in trajectory["mining_counts"])


def test_animation_writes_a_gif_for_a_mining_episode(tmp_path):
    env = make_env(initial_positive_probability=1.0, minimum_positive_probability=1.0)
    env.reset(seed=0)
    for action in (UP, 1, MINE, MINE):
        env.step(action)

    path = animate_episode(env, save_path=str(tmp_path / "mining.gif"), fps=2)
    assert (tmp_path / "mining.gif").exists()
    assert (tmp_path / "mining.gif").stat().st_size > 0
    assert path.endswith("mining.gif")


def test_animation_can_overlay_a_policy(tmp_path):
    env = make_mining_gridworld(seed=0)
    _V, policy, _history = value_iteration(env, gamma=0.95)
    env.reset(seed=0)
    for action in (0, 0, 0):
        env.step(action)

    animate_episode(
        env, save_path=str(tmp_path / "policy.gif"), fps=2, show_policy=policy
    )
    assert (tmp_path / "policy.gif").exists()


def test_animation_still_works_for_a_plain_gridworld(tmp_path):
    env = make_easy_gridworld()
    env.reset(seed=0)
    for action in (0, 1, 0):
        env.step(action)
    animate_episode(env, save_path=str(tmp_path / "plain.gif"), fps=2)
    assert (tmp_path / "plain.gif").exists()


def test_animation_rejects_a_mining_trajectory_with_missing_snapshots(tmp_path):
    env = make_env()
    env.reset(seed=0)
    env.step(UP)
    env.trajectory["mining_counts"].pop()

    with pytest.raises(ValueError, match="one 'mining_counts' snapshot per frame"):
        animate_episode(env, save_path=str(tmp_path / "broken.gif"))


def test_animation_rejects_an_empty_trajectory(tmp_path):
    env = make_env()
    env.trajectory = env._empty_trajectory()
    with pytest.raises(ValueError, match="empty"):
        animate_episode(env, save_path=str(tmp_path / "empty.gif"))
