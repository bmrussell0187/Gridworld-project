"""Tabular Q-learning: a model-free alternative to dynamic programming.

Unlike ``dynamic_programming.py``, this algorithm never calls
``env.transition_probabilities`` -- it only ever sees samples (s, a, r, s')
obtained by actually calling ``env.step``. This is the key distinction
between *model-based* methods (which require/exploit knowledge of P and R)
and *model-free* methods (which estimate the action-value function Q(s, a)
directly from experience).

Q-learning update rule (off-policy TD control):

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

The agent behaves epsilon-greedily with respect to its current Q estimates
(explore with probability epsilon, otherwise exploit the best known action),
while the update target always uses the *greedy* max_a' Q(s', a') -- this
"off-policy" mismatch between the behaviour policy and the target policy is
what makes it Q-learning rather than SARSA.
"""

from __future__ import annotations

import numpy as np



def q_learning(
    env,
    episodes: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: int | None = None,
    q_init: float = 0.0
) -> tuple[np.ndarray, np.ndarray, list[float], list[int]]:
    """Learn Q(s, a) from experience using epsilon-greedy tabular Q-learning.

    Parameters
    ----------
    env: GridWorldEnv
        The environment to train on.
    episodes:
        Number of episodes to train for.
    alpha:
        Learning rate.
    gamma:
        Discount factor.
    epsilon, epsilon_min, epsilon_decay:
        Epsilon-greedy exploration schedule: epsilon starts at ``epsilon``
        and is multiplied by ``epsilon_decay`` after every episode, floored
        at ``epsilon_min``.
    seed:
        Optional seed for both the environment and the exploration RNG, for
        reproducibility.

    Returns
    -------
    Q:
        Array of shape (n_states, n_actions), the learned action-value
        function.
    policy:
        Array of shape (n_states,), the greedy policy argmax_a Q(s, a).
    episode_returns:
        Total (undiscounted) reward obtained in each training episode.
    episode_lengths:
        Number of steps taken in each training episode.
    """
    n_states = env.n_states
    n_actions = env.action_space.n
    Q = np.full((n_states, n_actions),float(q_init))
    rng = np.random.default_rng(seed)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    # Seed the environment once, up front, so the whole training run is
    # reproducible; later resets reuse (and advance) the same env-internal
    # RNG rather than restarting it, so slip randomness varies episode to
    # episode as it should.
    obs, _info = env.reset(seed=seed)

    for _episode in range(episodes):
        if _episode > 0:
            obs, _info = env.reset()

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if rng.random() < epsilon:
                action = int(rng.integers(n_actions))
            else:
                action = int(np.argmax(Q[obs]))

            next_obs, reward, terminated, truncated, _info = env.step(action)

            # Terminal states are absorbing with V(terminal) = 0, so we do
            # not bootstrap through them; a truncated (time-limit) episode
            # is not a true terminal state, so bootstrapping continues.
            if terminated:
                td_target = reward
            else:
                td_target = reward + gamma * np.max(Q[next_obs])

            Q[obs, action] += alpha * (td_target - Q[obs, action])

            obs = next_obs
            total_reward += reward
            steps += 1

        episode_returns.append(total_reward)
        episode_lengths.append(steps)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    policy = np.argmax(Q, axis=1)
    return Q, policy, episode_returns, episode_lengths
