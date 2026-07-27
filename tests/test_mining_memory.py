"""Mining counts are per-node, persistent within an episode, and capped."""

from __future__ import annotations

import pytest

from conftest import DOWN, LEFT, MINE, RIGHT, UP, always_succeeds, put_agent

NODE_A = (1, 1)  # index 0 in the ordered node list
NODE_B = (3, 3)  # index 1


def test_ordering_of_the_count_vector_is_sorted_and_stable():
    env = always_succeeds()
    assert env.mining_nodes == (NODE_A, NODE_B)
    assert env.mining_node_index(NODE_A) == 0
    assert env.mining_node_index(NODE_B) == 1
    assert env.mining_node_index((0, 0)) is None


def test_mining_increments_only_that_nodes_count():
    env = always_succeeds()
    put_agent(env, NODE_A, (0, 0))
    env.step(MINE)
    assert env._mining_counts == (1, 0)
    env.step(MINE)
    assert env._mining_counts == (2, 0)


def test_counts_persist_when_the_agent_leaves_and_returns():
    env = always_succeeds()
    put_agent(env, NODE_A, (0, 0))
    env.step(MINE)
    assert env._mining_counts == (1, 0)

    for action in (UP, RIGHT, DOWN, LEFT):  # a loop back to the node
        env.step(action)
    assert env._agent_pos == NODE_A
    assert env._mining_counts == (1, 0)

    _obs, _reward, _terminated, _truncated, info = env.step(MINE)
    assert info["mining_attempt_number"] == 2  # picks up where it left off
    assert env._mining_counts == (2, 0)


def test_the_second_node_has_an_independent_count():
    env = always_succeeds()
    put_agent(env, NODE_A, (0, 0))
    env.step(MINE)
    put_agent(env, NODE_B, env._mining_counts)
    _obs, _reward, _terminated, _truncated, info = env.step(MINE)

    assert info["mining_attempt_number"] == 1  # node B is untouched so far
    assert env._mining_counts == (1, 1)


def test_reset_clears_every_count():
    env = always_succeeds()
    put_agent(env, NODE_A, (0, 0))
    env.step(MINE)
    put_agent(env, NODE_B, env._mining_counts)
    env.step(MINE)
    assert env._mining_counts == (1, 1)

    obs, info = env.reset(seed=1)
    assert env._mining_counts == (0, 0)
    assert info["mining_counts"] == (0, 0)
    assert env._agent_pos == env.config.start
    assert env._step_count == 0
    assert obs == env.components_to_state(env.config.start, (0, 0))


def test_counts_stop_growing_at_the_cap_but_attempts_keep_working():
    cap = 2
    env = always_succeeds(max_mining_count=cap)
    put_agent(env, NODE_A, (0, 0))

    attempts = []
    for _ in range(5):
        _obs, _reward, _terminated, _truncated, info = env.step(MINE)
        attempts.append((info["mining_attempt_number"], env._mining_counts[0]))

    assert attempts == [(1, 1), (2, 2), (3, 2), (3, 2), (3, 2)]
    # Once capped, every further attempt is evaluated at the saturated level
    # m = cap + 1, which keeps the transition kernel a function of the state.


def test_reward_grows_with_the_attempt_number_up_to_the_cap():
    env = always_succeeds()
    put_agent(env, NODE_A, (0, 0))
    rewards = [env.step(MINE)[1] for _ in range(4)]

    scale = env.config.positive_reward_scale
    step = env.config.step_reward
    assert rewards[0] == pytest.approx(step + scale(1))
    assert rewards[1] == pytest.approx(step + scale(2))
    assert rewards[2] == pytest.approx(step + scale(3))
    assert rewards[3] == pytest.approx(step + scale(3))  # saturated
    assert rewards[0] < rewards[1] < rewards[2]


def test_info_and_trajectory_snapshots_are_immutable_tuples():
    env = always_succeeds()
    put_agent(env, NODE_A, (0, 0))
    _obs, _reward, _terminated, _truncated, info = env.step(MINE)

    assert isinstance(info["mining_counts"], tuple)
    assert info["current_node_mining_count"] == 1
    # A caller holding on to an earlier snapshot must not see later mutations.
    snapshot = info["mining_counts"]
    env.step(MINE)
    assert snapshot == (1, 0)
    assert env._mining_counts == (2, 0)


def test_trajectory_records_one_count_snapshot_per_frame():
    env = always_succeeds()
    put_agent(env, NODE_A, (0, 0))
    for action in (MINE, UP, MINE):
        env.step(action)

    positions = env.trajectory["positions"]
    counts = env.trajectory["mining_counts"]
    assert len(counts) == len(positions)
    assert counts[0] == (0, 0)                    # frame 0 is the reset state
    assert counts == [(0, 0), (1, 0), (1, 0), (1, 0)]
    assert all(isinstance(c, tuple) for c in counts)
