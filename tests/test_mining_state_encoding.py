"""The augmented state encoding must be an exact bijection.

MineWorld's state is ``(position, mining_counts)`` flattened to one integer.
If that flattening were not a bijection, tabular value iteration and
Q-learning would silently share or lose table rows, so it is worth testing
exhaustively over the whole (small) state space.
"""

from __future__ import annotations

import pytest

from conftest import all_component_pairs, make_env


def test_state_space_size_matches_the_formula():
    env = make_env()
    expected = env.width * env.height * (env.config.max_mining_count + 1) ** env.n_mining_nodes
    assert env.n_states == expected
    assert env.observation_space.n == expected


def test_every_component_pair_maps_to_a_unique_state():
    env = make_env()
    states = [env.components_to_state(pos, counts) for pos, counts in all_component_pairs(env)]
    assert len(states) == env.n_states
    assert len(set(states)) == env.n_states           # unique
    assert set(states) == set(range(env.n_states))    # and exactly covers the range


def test_decode_is_the_exact_inverse_of_encode():
    env = make_env()
    for position, counts in all_component_pairs(env):
        state = env.components_to_state(position, counts)
        assert env.state_to_components(state) == (position, counts)


def test_encode_is_the_exact_inverse_of_decode():
    env = make_env()
    for state in range(env.n_states):
        position, counts = env.state_to_components(state)
        assert env.components_to_state(position, counts) == state


def test_convenience_accessors_agree_with_the_full_decoding():
    env = make_env()
    for state in range(env.n_states):
        position, counts = env.state_to_components(state)
        assert env.state_to_coord(state) == position
        assert env.state_to_mining_counts(state) == counts


def test_encoding_survives_a_third_mining_node():
    env = make_env(mining_nodes={(1, 1), (3, 3), (0, 2)}, max_mining_count=2)
    assert env.n_states == 4 * 4 * 3**3
    for state in range(env.n_states):
        position, counts = env.state_to_components(state)
        assert len(counts) == 3
        assert env.components_to_state(position, counts) == state


def test_encoding_degenerates_to_plain_gridworld_without_mining_nodes():
    env = make_env(mining_nodes=set())
    assert env.n_states == env.width * env.height
    for state in range(env.n_states):
        (x, y), counts = env.state_to_components(state)
        assert counts == ()
        assert y * env.width + x == state


@pytest.mark.parametrize("position", [(-1, 0), (0, -1), (4, 0), (0, 4), (99, 99)])
def test_out_of_bounds_positions_are_rejected(position):
    env = make_env()
    with pytest.raises(ValueError, match="outside"):
        env.components_to_state(position, (0, 0))


@pytest.mark.parametrize("counts", [(), (0,), (0, 0, 0)])
def test_wrong_length_count_vectors_are_rejected(counts):
    env = make_env()
    with pytest.raises(ValueError, match="one entry per mining node"):
        env.components_to_state((0, 0), counts)


@pytest.mark.parametrize("counts", [(-1, 0), (0, 3), (5, 5)])
def test_out_of_range_counts_are_rejected(counts):
    env = make_env()
    with pytest.raises(ValueError, match=r"mining counts must be in"):
        env.components_to_state((0, 0), counts)


@pytest.mark.parametrize("offset", [-1, 0])
def test_out_of_range_state_indices_are_rejected(offset):
    env = make_env()
    bad_state = env.n_states + offset if offset == 0 else offset
    with pytest.raises(ValueError, match="state must be in"):
        env.state_to_components(bad_state)


def test_skip_state_indices_covers_every_count_vector_of_a_terminal_cell():
    env = make_env()
    skip = env.skip_state_indices()
    goal = (3, 0)
    for _pos, counts in all_component_pairs(env):
        assert env.components_to_state(goal, counts) in skip
    # ... and nothing else: this env has no walls.
    assert len(skip) == env.config.n_mining_levels**env.n_mining_nodes
