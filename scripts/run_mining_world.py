"""Solve and visualise the MineWorld MDP: GridWorld plus a MINE action.

MineWorld adds a fifth action and a *memory*: each mining node remembers how
many times it has been mined this episode, and the gamble gets both more
lucrative and more dangerous with every attempt. Because those counts change
the future reward distribution, they are part of the state -- so the state
space is (cell, count-vector), not just the cell.

This script:
  1. prints the risk/reward schedule the agent is up against;
  2. solves the MDP exactly with value iteration (the payout distribution of
     the default example is deterministic, so P(s'|s,a) has finite support);
  3. plots the optimal policy for several mining-count "slices";
  4. runs one greedy episode and animates it, with live per-node counts;
  5. trains tabular Q-learning on the same MDP for comparison.

Run from anywhere once the package is installed (``pip install -e .``):
    python scripts/run_mining_world.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from gridworld.animation import animate_episode
from gridworld.dynamic_programming import value_iteration
from gridworld.examples import make_mining_gridworld
from gridworld.mining_env import ACTION_NAMES, MINE
from gridworld.plotting import (
    plot_learning_curve,
    plot_mining_schedule,
    plot_policy,
    plot_value_function,
)
from gridworld.q_learning import q_learning

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
GAMMA = 0.95


def greedy_episode(env, policy: np.ndarray, seed: int = 0) -> float:
    """Run one episode following ``policy`` greedily; return the total reward."""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = int(policy[obs])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info["mining_attempted"]:
            outcome = "struck it lucky" if info["mining_success"] else "MINE COLLAPSED"
            print(
                f"  t={info['step']:>3} {ACTION_NAMES[info['actual_action']]:<5} "
                f"at {info['position']} attempt #{info['mining_attempt_number']} "
                f"-> {outcome}, reward {reward:+.2f}"
            )
    print(f"  episode finished after {info['step']} steps, total reward {total_reward:+.2f}")
    return total_reward


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = make_mining_gridworld(seed=0)

    print(f"MineWorld: {env.width}x{env.height} grid, mining nodes at {env.mining_nodes}")
    print(f"state space: {env.n_states} states "
          f"= {env.width} * {env.height} * {env.config.n_mining_levels}**{env.n_mining_nodes}")
    print()
    print(env.config.describe_mining_schedule())
    print()

    ax = plot_mining_schedule(env.config)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(os.path.join(OUTPUT_DIR, "mining_schedule.png"), dpi=150)
    plt.close(ax.get_figure())

    # --- 1. Exact solution via value iteration -------------------------
    V, policy, history = value_iteration(env, gamma=GAMMA)
    print(f"value iteration converged in {len(history)} sweeps "
          f"(final delta {history[-1]:.2e})")
    start_state = env.components_to_state(env.config.start, (0,) * env.n_mining_nodes)
    print(f"V*(start, nothing mined) = {V[start_state]:.3f}")

    # The optimal action at each node, as a function of how often it has
    # already been mined: this is the risk/reward decision, made explicit.
    print("\noptimal action at each mining node, by attempt count:")
    for index, node in enumerate(env.mining_nodes):
        actions = []
        for count in range(env.config.n_mining_levels):
            counts = [0] * env.n_mining_nodes
            counts[index] = count
            state = env.components_to_state(node, tuple(counts))
            actions.append(f"count={count}: {ACTION_NAMES[int(policy[state])]}")
        print(f"  node {node}: " + ", ".join(actions))

    # --- 2. Policy / value plots, one per mining-count slice -----------
    slices = [(0,) * env.n_mining_nodes, (1,) * env.n_mining_nodes,
              (env.config.max_mining_count,) * env.n_mining_nodes]
    fig, axes = plt.subplots(1, len(slices), figsize=(6 * len(slices), 6))
    for ax, counts in zip(np.atleast_1d(axes), slices):
        plot_policy(env, policy, ax=ax, mining_counts=counts,
                    title=f"pi* | mining counts = {counts}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "mining_policy.png"), dpi=150)
    plt.close(fig)

    ax = plot_value_function(env, V, mining_counts=(0,) * env.n_mining_nodes,
                             title="V* | nothing mined yet")
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(os.path.join(OUTPUT_DIR, "mining_value_function.png"), dpi=150)
    plt.close(ax.get_figure())

    # --- 3. One greedy episode, animated -------------------------------
    print("\ngreedy episode under pi*:")
    greedy_episode(env, policy, seed=3)
    animate_episode(
        env,
        save_path=os.path.join(OUTPUT_DIR, "mining_agent.gif"),
        fps=2,
        show_policy=policy,
    )
    print(f"  animation saved to {os.path.join(OUTPUT_DIR, 'mining_agent.gif')}")

    # --- 4. Model-free comparison --------------------------------------
    # Q-learning sees the same MDP but never sees P(s'|s,a): it has to
    # discover the mining gamble from samples, including the rare -10
    # collapse. That is much harder here than in a plain GridWorld, for two
    # reasons worth discussing in a write-up:
    #   * the state space is (cell, count-vector), so it is (max_count+1)**K
    #     times larger than the grid, and most of it is visited rarely;
    #   * the mining reward has high variance, so each TD update is noisy.
    print("\ntraining tabular Q-learning on the same MDP (model-free)...")
    q_env = make_mining_gridworld(seed=0)
    Q, q_policy, returns, _lengths = q_learning(
        q_env, episodes=5000, alpha=0.05, gamma=GAMMA, epsilon_decay=0.999, seed=0
    )
    print(f"  mean return over the last 100 episodes: {np.mean(returns[-100:]):+.3f} "
          f"(V*(start) = {V[start_state]:.3f}, discounted)")

    solvable = [s for s in range(env.n_states) if s not in env.skip_state_indices()]
    agreement = np.mean([q_policy[s] == policy[s] for s in solvable])
    print(f"  greedy Q-policy matches pi* on {agreement:.0%} of the {len(solvable)} "
          "non-absorbing states")

    fresh_nodes = [
        env.components_to_state(node, (0,) * env.n_mining_nodes) for node in env.mining_nodes
    ]
    print("  at a *fresh* mining node, Q-learning would take "
          f"{[ACTION_NAMES[int(q_policy[s])] for s in fresh_nodes]} "
          f"(pi* takes {[ACTION_NAMES[MINE]] * len(fresh_nodes)})")
    for node, state in zip(env.mining_nodes, fresh_nodes):
        # Q*(s, MINE) straight from the model: sum_s' P(s'|s,a)[R + gamma V*(s')],
        # with no bootstrapping through a terminating collapse.
        q_star_mine = sum(
            probability * (reward + (0.0 if terminated else GAMMA * V[next_state]))
            for probability, next_state, reward, terminated
            in env.transition_probabilities(state, MINE)
        )
        print(f"    node {node}: Q(mine) = {Q[state, MINE]:+.3f} vs "
              f"Q*(mine) = {q_star_mine:+.3f}; best other action learned "
              f"Q = {max(Q[state, a] for a in range(4)):+.3f}")
    print("  (a gap here is expected: tabular Q-learning has to visit each "
          "(cell, count-vector) state many times, and the rare mine collapse "
          "makes every update noisy -- exactly the trade-off that motivates "
          "model-based DP when a model is available.)")

    ax = plot_learning_curve(
        returns,
        save_path=os.path.join(OUTPUT_DIR, "mining_learning_curve.png"),
        window=50,
        title="Q-learning on MineWorld",
    )
    plt.close(ax.get_figure())
    print(f"\nall figures written to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
