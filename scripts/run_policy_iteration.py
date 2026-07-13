"""Solve the easy GridWorld MDP using policy iteration, and compare to value iteration.

Policy iteration alternates between two steps until convergence:
  1. Policy evaluation -- compute V_pi exactly for the current policy.
  2. Policy improvement -- make the policy greedy with respect to V_pi.

Both value iteration and policy iteration are guaranteed to converge to the
*same* optimal value function V* and an optimal policy pi* for a finite
MDP, but they get there differently -- this script solves the same MDP
both ways and checks that the results agree.

Run from the repository root:
    python scripts/run_policy_iteration.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from gridworld.examples import make_easy_gridworld
from gridworld.dynamic_programming import policy_iteration, value_iteration
from gridworld.plotting import plot_value_function, plot_policy

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = make_easy_gridworld(seed=0)

    V_pi, policy_pi, history = policy_iteration(env, gamma=0.95, theta=1e-8)
    print(f"Policy iteration converged in {len(history)} outer iterations")

    ax = plot_value_function(env, V_pi, title="Policy iteration: V_pi(s)")
    ax.get_figure().savefig(
        os.path.join(OUTPUT_DIR, "policy_iteration_value_function.png"), dpi=150
    )
    plt.close(ax.get_figure())

    ax = plot_policy(env, policy_pi, title="Policy iteration: optimal policy")
    ax.get_figure().savefig(
        os.path.join(OUTPUT_DIR, "policy_iteration_policy.png"), dpi=150
    )
    plt.close(ax.get_figure())

    # Cross-check against value iteration on the same MDP.
    V_vi, policy_vi, _ = value_iteration(env, gamma=0.95, theta=1e-8)
    value_diff = np.max(np.abs(V_pi - V_vi))
    policy_agreement = np.mean(policy_pi == policy_vi)
    print(f"Max |V_policy_iteration - V_value_iteration| = {value_diff:.2e}")
    print(f"Fraction of states with matching greedy action = {policy_agreement:.2%}")
    print(
        "(States can have equally-good actions in a tie, so <100% policy "
        "agreement does not necessarily mean either method is wrong -- "
        "check that V matches closely.)"
    )


if __name__ == "__main__":
    main()
