"""Dynamic programming solutions to a finite MDP: value and policy iteration.

Dynamic programming (DP) methods are *model-based*: they assume full
knowledge of the transition function P(s'|s,a) and reward function R, and
compute an optimal policy by repeatedly applying the Bellman equations. This
contrasts with *model-free* methods like Q-learning (``q_learning.py``),
which only ever see samples (s, a, r, s') from interacting with the
environment and never see P or R directly.

Both algorithms below rely on ``env.transition_probabilities(state, action)``
(see ``env.py``), which exposes P(s'|s,a) explicitly -- this is the "model"
that Q-learning does *not* have access to.

Bellman optimality equation (what value iteration solves):

    V*(s) = max_a  sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * V*(s') ]

Bellman expectation equation (what policy evaluation solves, for a fixed
policy pi):

    V_pi(s) = sum_{s'} P(s'|pi(s),s) [ R(s,pi(s),s') + gamma * V_pi(s') ]

Terminal states are absorbing and, by convention, V(terminal) = 0 (no
future reward can be earned once the episode has ended), so they are
excluded from the update/improvement sweeps below.
"""

from __future__ import annotations

import numpy as np


def _terminal_and_wall_indices(env) -> set[int]:
    """States that should never be updated: terminal cells and wall cells.

    Wall cells are not reachable/occupiable states, but we still allocate
    array slots for every (x, y) index for simplicity; they are simply
    skipped by the sweeps below.

    Environments with an augmented state space (e.g.
    :class:`~gridworld.mining_env.MineWorldEnv`, whose state is a cell *plus*
    a mining-count vector) expose their own ``skip_state_indices()``, since
    one cell then corresponds to many state indices.
    """
    if hasattr(env, "skip_state_indices"):
        return set(env.skip_state_indices())
    skip = {env.coord_to_state(c) for c in env.config.terminal_states}
    skip |= {env.coord_to_state(c) for c in env.walls}
    return skip


def _action_value(env, V: np.ndarray, state: int, action: int, gamma: float) -> float:
    """Q(s, a) = sum_{s'} P(s'|s,a) [ R + gamma * V(s') ], from the model.

    A transition flagged ``terminated`` earns no future reward, so it
    contributes ``reward`` alone. For an ordinary GridWorld this changes
    nothing (terminal cells keep V = 0 anyway), but MineWorld can terminate
    on a *non-terminal* cell when the mine collapses, and bootstrapping
    through that successor state would badly overestimate the gamble.
    """
    q = 0.0
    for prob, next_state, reward, terminated in env.transition_probabilities(state, action):
        future = 0.0 if terminated else gamma * V[next_state]
        q += prob * (reward + future)
    return q


def value_iteration(
    env,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Solve for V* and an optimal policy pi* via the Bellman optimality equation.

    Repeatedly applies V(s) <- max_a Q(s, a) to every non-terminal state
    until the largest change in V across a full sweep drops below ``theta``
    (or ``max_iterations`` sweeps have elapsed).

    Returns
    -------
    V:
        Array of shape (n_states,), the converged state-value function.
    policy:
        Array of shape (n_states,), the greedy (optimal) action in each
        state, ``argmax_a Q(s, a)`` computed from the converged V.
    history:
        List of the maximum value change (delta) observed at each sweep --
        useful for plotting convergence.
    """
    n_states = env.n_states
    n_actions = env.action_space.n
    skip_states = _terminal_and_wall_indices(env)

    V = np.zeros(n_states)
    history: list[float] = []

    for _ in range(max_iterations):
        delta = 0.0
        for s in range(n_states):
            if s in skip_states:
                continue
            q_values = [_action_value(env, V, s, a, gamma) for a in range(n_actions)]
            best_value = max(q_values)
            delta = max(delta, abs(best_value - V[s]))
            V[s] = best_value
        history.append(delta)
        if delta < theta:
            break

    policy = _greedy_policy(env, V, gamma, skip_states, n_states, n_actions)
    return V, policy, history


def policy_iteration(
    env,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iterations: int = 1_000,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Solve for an optimal policy via alternating policy evaluation/improvement.

    Each outer iteration:
      1. **Policy evaluation**: compute V_pi for the current policy exactly
         (by iterating the Bellman expectation equation to convergence).
      2. **Policy improvement**: update the policy to be greedy with
         respect to V_pi.
    Repeats until the policy no longer changes (guaranteed to converge to
    an optimal policy in a finite MDP).

    Returns
    -------
    V:
        Array of shape (n_states,), the value function of the final policy
        (= V* once the policy has converged).
    policy:
        Array of shape (n_states,), the converged (optimal) policy.
    history:
        List of the policy-evaluation convergence delta (max value change
        in the final evaluation sweep) recorded at each outer iteration --
        useful for plotting convergence.
    """
    n_states = env.n_states
    n_actions = env.action_space.n
    skip_states = _terminal_and_wall_indices(env)

    V = np.zeros(n_states)
    # Arbitrary initial policy: always pick action 0 ("up").
    policy = np.zeros(n_states, dtype=int)
    history: list[float] = []

    for _ in range(max_iterations):
        # --- Policy evaluation: solve V_pi exactly for the current policy ---
        eval_delta = np.inf
        while eval_delta >= theta:
            eval_delta = 0.0
            for s in range(n_states):
                if s in skip_states:
                    continue
                new_v = _action_value(env, V, s, int(policy[s]), gamma)
                eval_delta = max(eval_delta, abs(new_v - V[s]))
                V[s] = new_v
        history.append(eval_delta)

        # --- Policy improvement: make the policy greedy w.r.t. V ---
        policy_stable = True
        for s in range(n_states):
            if s in skip_states:
                continue
            old_action = policy[s]
            q_values = [_action_value(env, V, s, a, gamma) for a in range(n_actions)]
            best_action = int(np.argmax(q_values))
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

        if policy_stable:
            break

    return V, policy, history


def _greedy_policy(
    env, V: np.ndarray, gamma: float, skip_states: set[int], n_states: int, n_actions: int
) -> np.ndarray:
    """Extract the greedy policy pi(s) = argmax_a Q(s, a) from a value function."""
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        if s in skip_states:
            continue
        q_values = [_action_value(env, V, s, a, gamma) for a in range(n_actions)]
        policy[s] = int(np.argmax(q_values))
    return policy
