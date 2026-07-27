"""``step()`` and ``transition_probabilities()`` must describe the same MDP.

The environment is implemented once (``_transition_branches``) and consumed
twice, so these tests are the guarantee that the model used by dynamic
programming is the model the agent actually experiences.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from gridworld.dynamic_programming import policy_iteration, value_iteration
from gridworld.examples import (
    make_continuous_mining_gridworld,
    make_mining_gridworld,
    make_risky_mining_gridworld,
)
from gridworld.mining_config import ContinuousRewardModelError
from gridworld.mining_env import MineWorldEnv

from conftest import MINE, UP, all_component_pairs, env_state_snapshot, make_env, put_agent

NODE_A = (1, 1)


def test_every_transition_distribution_sums_to_one():
    for env in (make_env(), make_env(slip_probability=0.3)):
        for state in range(env.n_states):
            for action in range(env.action_space.n):
                total = sum(p for p, _s, _r, _t in env.transition_probabilities(state, action))
                assert total == pytest.approx(1.0), (state, action)


def test_transition_probabilities_are_all_non_negative():
    env = make_env(slip_probability=0.25)
    for state in range(env.n_states):
        for action in range(env.action_space.n):
            assert all(p > 0.0 for p, _s, _r, _t in env.transition_probabilities(state, action))


def test_mining_branch_carries_the_incremented_count():
    env = make_env()
    state = env.components_to_state(NODE_A, (0, 0))
    outcomes = env.transition_probabilities(state, MINE)

    assert len(outcomes) == 2  # success and collapse
    for probability, next_state, _reward, terminated in outcomes:
        position, counts = env.state_to_components(next_state)
        assert position == NODE_A       # MINE never moves the agent
        assert counts == (1, 0)         # the attempt is counted either way
        assert probability > 0.0
        del terminated

    success = [o for o in outcomes if not o[3]][0]
    failure = [o for o in outcomes if o[3]][0]
    assert success[0] == pytest.approx(env.config.positive_reward_probability(1))
    assert failure[0] == pytest.approx(1.0 - env.config.positive_reward_probability(1))
    assert failure[2] == pytest.approx(
        env.config.step_reward + env.config.mining_failure_reward_at(1)
    )


def test_a_terminating_mining_failure_is_represented_in_the_model():
    env = make_env()
    state = env.components_to_state(NODE_A, (0, 0))
    terminating = [o for o in env.transition_probabilities(state, MINE) if o[3]]
    assert len(terminating) == 1
    # ... and the cell it happens in is *not* an ordinary terminal state.
    assert NODE_A not in env.config.terminal_states


def test_movement_never_changes_the_counts_in_the_model():
    env = make_env(slip_probability=0.2)
    for position, counts in all_component_pairs(env):
        if position in env.config.terminal_states or position in env.walls:
            continue
        state = env.components_to_state(position, counts)
        for action in (0, 1, 2, 3):
            for _p, next_state, _r, _t in env.transition_probabilities(state, action):
                _next_position, next_counts = env.state_to_components(next_state)
                if env.mining_node_index(position) is None:
                    assert next_counts == counts
                else:
                    # On a node, only a slip *into* MINE may bump the count.
                    assert next_counts in (counts, env._incremented_counts(counts, env.mining_node_index(position)))


def test_ordinary_terminal_states_are_absorbing():
    env = make_env()
    goal = (3, 0)
    for _position, counts in all_component_pairs(env):
        state = env.components_to_state(goal, counts)
        for action in range(env.action_space.n):
            assert env.transition_probabilities(state, action) == [(1.0, state, 0.0, True)]


def test_transition_probabilities_do_not_mutate_the_environment():
    env = make_env(slip_probability=0.2)
    put_agent(env, NODE_A, (1, 0))
    env.step(UP)
    before = env_state_snapshot(env)

    for state in range(env.n_states):
        for action in range(env.action_space.n):
            env.transition_probabilities(state, action)

    assert env_state_snapshot(env) == before


def test_rendering_and_reward_queries_do_not_mutate_the_environment():
    env = make_env()
    env.render_mode = "rgb_array"
    put_agent(env, NODE_A, (1, 1))
    before = env_state_snapshot(env)
    env.render()
    env._movement_reward((0, 1), invalid_move=False)
    env._transition_branches(NODE_A, (1, 1), MINE)
    assert env_state_snapshot(env) == before


@pytest.mark.parametrize(
    "factory", [make_mining_gridworld, make_risky_mining_gridworld]
)
def test_sampled_frequencies_agree_with_the_explicit_model(factory):
    """Roll the dice many times from one state and compare with P(s'|s,a)."""
    env = factory(seed=0)
    env.reset(seed=4321)
    node = env.mining_nodes[0]
    counts = (1,) + (0,) * (env.n_mining_nodes - 1)
    state = env.components_to_state(node, counts)

    model = {}
    for probability, next_state, reward, terminated in env.transition_probabilities(state, MINE):
        model[(next_state, round(reward, 6), terminated)] = probability

    trials = 20_000
    empirical: collections.Counter = collections.Counter()
    for _ in range(trials):
        put_agent(env, node, counts)
        obs, reward, terminated, _truncated, _info = env.step(MINE)
        empirical[(obs, round(reward, 6), terminated)] += 1

    assert set(empirical) <= set(model)
    for key, probability in model.items():
        assert empirical[key] / trials == pytest.approx(probability, abs=0.02)


def test_no_sampled_outcome_is_missing_from_the_model():
    """Sweep many (state, action) pairs and compare sampling with the model.

    This is the broad version of the two focused tests above: for each pair
    it checks that every sampled ``(next_state, reward, terminated)`` triple
    appears in ``transition_probabilities`` and at roughly the right rate.
    """
    env = make_env(slip_probability=0.2)
    env.reset(seed=0)
    skip = env.skip_state_indices()
    rng = np.random.default_rng(0)
    trials = 1500

    for _ in range(25):
        state = int(rng.integers(env.n_states))
        if state in skip:
            continue
        action = int(rng.integers(env.action_space.n))
        model = {
            (next_state, round(reward, 6), terminated): probability
            for probability, next_state, reward, terminated
            in env.transition_probabilities(state, action)
        }
        position, counts = env.state_to_components(state)

        empirical: collections.Counter = collections.Counter()
        for _ in range(trials):
            put_agent(env, position, counts)
            obs, reward, terminated, _truncated, _info = env.step(action)
            empirical[(obs, round(reward, 6), terminated)] += 1

        assert set(empirical) <= set(model), (state, action, set(empirical) - set(model))
        for key, probability in model.items():
            assert empirical[key] / trials == pytest.approx(probability, abs=0.05)


def test_sampled_movement_frequencies_agree_under_slip():
    env = make_env(slip_probability=0.4)
    env.reset(seed=11)
    start = (2, 2)
    counts = (0, 0)
    state = env.components_to_state(start, counts)
    model = {
        (next_state, round(reward, 6), terminated): probability
        for probability, next_state, reward, terminated
        in env.transition_probabilities(state, UP)
    }

    trials = 20_000
    empirical: collections.Counter = collections.Counter()
    for _ in range(trials):
        put_agent(env, start, counts)
        obs, reward, terminated, _truncated, _info = env.step(UP)
        empirical[(obs, round(reward, 6), terminated)] += 1

    for key, probability in model.items():
        assert empirical[key] / trials == pytest.approx(probability, abs=0.02)


# ----------------------------------------------------------------------
# Continuous reward distributions: sampling only
# ----------------------------------------------------------------------
def test_continuous_rewards_refuse_to_produce_an_exact_model():
    env = make_continuous_mining_gridworld(seed=0)
    with pytest.raises(ContinuousRewardModelError):
        env.transition_probabilities(0, MINE)
    with pytest.raises(ContinuousRewardModelError):
        value_iteration(env, gamma=0.9)


def test_continuous_rewards_can_still_be_sampled():
    env = make_continuous_mining_gridworld(seed=0)
    env.reset(seed=0)
    node = env.mining_nodes[0]
    rewards = []
    for _ in range(50):
        put_agent(env, node, (0,) * env.n_mining_nodes)
        _obs, reward, _terminated, _truncated, info = env.step(MINE)
        if info["mining_success"]:
            rewards.append(reward - env.config.step_reward)
    assert rewards and all(r > 0.0 for r in rewards)
    assert len(set(rewards)) > 1  # genuinely continuous, not a point mass


# ----------------------------------------------------------------------
# Dynamic programming on the exact model
# ----------------------------------------------------------------------
def test_value_iteration_solves_the_bellman_optimality_equation():
    from gridworld.dynamic_programming import _action_value

    env = make_mining_gridworld(seed=0)
    gamma = 0.95
    V, _policy, _history = value_iteration(env, gamma=gamma)
    skip = env.skip_state_indices()

    residual = max(
        abs(max(_action_value(env, V, s, a, gamma) for a in range(env.action_space.n)) - V[s])
        for s in range(env.n_states)
        if s not in skip
    )
    assert residual < 1e-6


def test_policy_iteration_agrees_with_value_iteration():
    gamma = 0.95
    V_vi, _pi_vi, _ = value_iteration(make_mining_gridworld(seed=0), gamma=gamma)
    V_pi, _pi_pi, _ = policy_iteration(make_mining_gridworld(seed=0), gamma=gamma)
    assert np.max(np.abs(V_vi - V_pi)) < 1e-6


def test_the_optimal_policy_mines_a_fresh_node_but_not_an_exhausted_one():
    """The example is tuned so the gamble is worth exactly one attempt."""
    env = make_mining_gridworld(seed=0)
    _V, policy, _history = value_iteration(env, gamma=0.95)

    for index, node in enumerate(env.mining_nodes):
        fresh = [0] * env.n_mining_nodes
        assert policy[env.components_to_state(node, tuple(fresh))] == MINE

        exhausted = [0] * env.n_mining_nodes
        exhausted[index] = env.config.max_mining_count
        assert policy[env.components_to_state(node, tuple(exhausted))] != MINE


def test_dp_does_not_bootstrap_through_a_terminating_mining_failure():
    """A collapse earns its penalty and nothing more.

    With a certain collapse and a huge penalty, the value of standing on a
    node must be no better than the penalty itself -- which is only true if
    ``_action_value`` stops discounting the future on a terminated branch.
    """
    env = make_env(
        initial_positive_probability=0.0,
        minimum_positive_probability=0.0,
        mining_failure_reward=-100.0,
    )
    _V, policy, _history = value_iteration(env, gamma=0.9)
    assert policy[env.components_to_state(NODE_A, (0, 0))] != MINE

    outcomes = env.transition_probabilities(env.components_to_state(NODE_A, (0, 0)), MINE)
    assert len(outcomes) == 1
    probability, _next_state, reward, terminated = outcomes[0]
    assert probability == pytest.approx(1.0)
    assert terminated
    assert reward == pytest.approx(env.config.step_reward - 100.0)


def test_a_mineworld_without_nodes_behaves_like_a_plain_gridworld():
    env = MineWorldEnv(make_env(mining_nodes=set()).config)
    assert env.n_states == env.width * env.height
    _V, policy, _history = value_iteration(env, gamma=0.95)
    assert policy.shape == (env.n_states,)
    # MINE is available but pointless, so it is never optimal.
    skip = env.skip_state_indices()
    assert all(policy[s] != MINE for s in range(env.n_states) if s not in skip)
