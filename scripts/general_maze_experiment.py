"""Vary ONE parameter of the maze gridworld experiment.

How to use: edit the SETTINGS block below (uncomment the experiment you want),
then just run this file. It trains an agent for every value x every seed, and
saves the results to outputs/ as

    maze_sweep_<EXPERIMENT>.json         the settings + one entry per seeded run
    maze_sweep_<EXPERIMENT>_arrays.npz   the big per-episode arrays

Every run is saved separately, because seeds of the same setting often do not agree 
(some find the +10 goal, some settle for the +1 goal).

All the plotting and post-experiment analysis happens in scripts/results_analysis.py.
"""

from __future__ import annotations

import datetime
import json
import os

import numpy as np

from gridworld.maze_gridworld import make_maze_gridworld
# To use the maze with shaping rewards instead, swap the import above for:
# from gridworld.maze_gridworld_shaping import make_maze_gridworld
# Note this version does not support changing step_reward or invalid_move_reward 
# in the SETTINGS block below. This can be fixed by adding those arguments to the 
# function signature in maze_gridworld_shaping.py, and passing them to GridWorldConfig, 
# just like in maze_gridworld.py.

from gridworld.q_learning import q_learning


# ===========================================================================
# SETTINGS - this is the only part you need to edit
# ===========================================================================

# Which parameter are we varying? 
# EXPERIMENT: short name, used for the output filenames.
# LABEL:      how it should appear on the x-axis of a plot.
# PARAM:      the exact keyword argument name it is passed as.
# TARGET:     "agent" if it is an argument of q_learning,
#             "env"   if it is an argument of make_maze_gridworld.
# VALUES:     the list of values to try.


# Baseline experiment
#EXPERIMENT = "baseline"
#LABEL = "baseline"
#PARAM = "q_init"
#TARGET = "agent"
#VALUES = [0]

# Experiment 1: how optimistic is the initial Q table?
#EXPERIMENT = "q_init"
#LABEL = "Q initialisation"
#PARAM = "q_init"
#TARGET = "agent"
#VALUES = [round(0.70 + 0.02 * i, 2) for i in range(16)]   # 0.70 ... 1.00

# Experiment 2: how quickly does exploration decay?
# EXPERIMENT = "epsilon_decay"
# LABEL = "Epsilon decay (per episode)"
# PARAM = "epsilon_decay"
# TARGET = "agent"
# VALUES = [0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995]

# Experiment 3: how much exploration do we keep forever?
EXPERIMENT = "epsilon_min"
LABEL = "Minimum epsilon"
PARAM = "epsilon_min"
TARGET = "agent"
VALUES = [0.0, 0.01, 0.05, 0.1, 0.2,0.5,0.9]

# Experiment 4: how big are the learning steps?
# EXPERIMENT = "alpha"
# LABEL = "Learning rate"
# PARAM = "alpha"
# TARGET = "agent"
# VALUES = [0.05, 0.1, 0.2, 0.4, 0.8]

# Experiment 5: how much does the environment punish each step taken?
# EXPERIMENT = "step_reward"
# LABEL = "Step penalty"
# PARAM = "step_reward"
# TARGET = "env"
# VALUES = [0.0, -0.005, -0.01, -0.02, -0.05, -0.1]

# Experiment 6: how much does the environment punish walking into a wall?
# EXPERIMENT = "invalid_move_reward"
# LABEL = "Invalid-move penalty"
# PARAM = "invalid_move_reward"
# TARGET = "env"
# VALUES = [0.0, -0.02, -0.05, -0.1, -0.2]


# Whether a run escapes the +1 goal is roughly a coin flip in the interesting range, 
# so a handful of seeds cannot measure it: 5 seeds can only ever report
#  0%, 20%, 40%, ... Should use maybe 10+.
SEEDS = list(range(10))
N_EPISODES = 30_000  # training episodes per run

AGENT_SETTINGS = {   # arguments of q_learning
    "alpha": 0.2,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.9999,
    "q_init": 0.0,
}

ENV_SETTINGS = {    # arguments of make_maze_gridworld
    "step_reward": -0.01,
    "invalid_move_reward": -0.05,
    "max_steps": 200,
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

# ===========================================================================


def run_one_agent(env_settings, agent_settings, seed):
    """Train a single agent, then walk its greedy policy once. Returns a dict."""
    env = make_maze_gridworld(seed=seed, **env_settings)

    Q, policy, episode_returns, episode_lengths = q_learning(
        env, episodes=N_EPISODES, seed=seed, **agent_settings
    )

    # Follow the learned policy with no exploration: how good is it really?
    # The maze has slip_probability = 0, so the environment is deterministic
    # and this single walk is the exact value of the learned policy.
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    greedy_return = 0.0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(int(policy[obs]))
        greedy_return += reward

    # The env records where the agent went, which tells us WHICH goal it found.
    greedy_path = [list(cell) for cell in env.trajectory["positions"]]

    # Average return over the last stretch of training, i.e. once it settled.
    window = min(500, N_EPISODES)
    final_train_return = float(np.mean(episode_returns[-window:]))

    return {
        # one number (or one short path) per run -> goes in the JSON
        "run": {
            "seed": seed,
            "greedy_return": float(greedy_return),
            "greedy_length": len(greedy_path) - 1,
            "greedy_terminated": bool(terminated),   # False means it ran out of steps
            "greedy_final_cell": greedy_path[-1],
            "greedy_path": greedy_path,
            "final_train_return": final_train_return,
        },
        # big arrays -> go in the .npz
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "q_values": Q,
        "policy": policy,
    }


# --- Run every value x every seed -----------------------------------------

runs = []               # one dict per (value, seed), in that order
episode_returns = []    # these four stay in step with runs, one entry per run
episode_lengths = []
q_values = []
policy = []

print(f"{LABEL}: {len(VALUES)} values x {len(SEEDS)} seeds x {N_EPISODES} episodes")

for value in VALUES:
    # Put the swept value where it belongs, leaving the other settings alone.
    agent_settings = dict(AGENT_SETTINGS)
    env_settings = dict(ENV_SETTINGS)
    if TARGET == "agent":
        agent_settings[PARAM] = value
    elif TARGET == "env":
        env_settings[PARAM] = value
    else:
        raise ValueError(f"TARGET must be 'agent' or 'env', not {TARGET!r}")

    for seed in SEEDS:
        print("Seed", SEEDS.index(seed)+1, "/", len(SEEDS), "in run", VALUES.index(value)+1, "/", len(VALUES))
        result = run_one_agent(env_settings, agent_settings, seed)
        result["run"][PARAM] = value        # remember which value this run used
        runs.append(result["run"])
        episode_returns.append(result["episode_returns"])
        episode_lengths.append(result["episode_lengths"])
        q_values.append(result["q_values"])
        policy.append(result["policy"])

    # Progress report. Printing the spread, not just the mean, because the
    # seeds of one setting often split into two different outcomes.
    just_run = [run["greedy_return"] for run in runs[-len(SEEDS):]]
    print(f"  {PARAM} = {value}: greedy return "
          f"min {min(just_run):+.2f}, median {np.median(just_run):+.2f}, "
          f"max {max(just_run):+.2f}", flush=True)


# ==== Saving ======================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
arrays_filename = f"maze_sweep_{EXPERIMENT}_arrays.npz"
json_path = os.path.join(OUTPUT_DIR, f"maze_sweep_{EXPERIMENT}.json")


def by_value_and_seed(flat_list):
    """Reshape one-entry-per-run into (values, seeds, ...).

    The runs were appended value by value, seed by seed, so the first
    len(SEEDS) entries are value 0, the next len(SEEDS) are value 1, and so on.
    """
    array = np.array(flat_list)
    return array.reshape((len(VALUES), len(SEEDS)) + array.shape[1:])


# The big arrays go in a .npz -- far too much data for a text file. Each has
# shape (values, seeds, ...), so averaging over the seeds is np.mean(a, axis=1).
np.savez_compressed(
    os.path.join(OUTPUT_DIR, arrays_filename),
    episode_returns=by_value_and_seed(episode_returns).astype(float),
    episode_lengths=by_value_and_seed(episode_lengths).astype(int),
    q_values=by_value_and_seed(q_values).astype(float),
    policy=by_value_and_seed(policy).astype(int),
)

# Everything else goes in a readable JSON file: the settings, so a plot can
# label itself, and every individual run, so the analysis can summarise them
# however it likes (mean, median, or "what fraction of seeds found the goal").
probe_env = make_maze_gridworld(seed=0, **ENV_SETTINGS)
results = {
    "experiment": EXPERIMENT,
    "created": datetime.datetime.now().isoformat(timespec="seconds"),
    "label": LABEL,
    "param": PARAM,
    "target": TARGET,
    "values": VALUES,
    "seeds": SEEDS,
    "episodes": N_EPISODES,
    "agent_settings": AGENT_SETTINGS,
    "env_settings": ENV_SETTINGS,
    # Which cells end an episode, and what they pay -- lets the analysis work
    # out which goal a run actually found.
    "terminal_states": [
        {"cell": list(cell), "reward": reward}
        for cell, reward in probe_env.config.terminal_states.items()
    ],
    # One entry per run, ordered value by value then seed by seed. No averaging.
    "runs": runs,
    "arrays_file": arrays_filename,
}
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"saved {len(runs)} runs to {json_path}")
print(f"saved {arrays_filename} -- now run scripts/results_analysis.py")
