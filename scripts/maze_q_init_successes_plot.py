#Frankensteined from results_analysis.py

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT = "q_init"            # must match EXPERIMENT in general_maze_experiment.py
VALUE=0.96


SMOOTHING_WINDOW = 100  

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


json_path = os.path.join(OUTPUT_DIR, f"maze_sweep_{EXPERIMENT}.json")
if not os.path.exists(json_path):
    raise FileNotFoundError(f"{json_path} not found -- run scripts/general_maze_experiment.py first")



with open(json_path) as f:
    results = json.load(f)

runs = results["runs"]                                # one entry per (value, seed)
values = np.array(results["values"], dtype=float)     # the swept parameter values
seeds = results["seeds"]
label = results["label"]                              # e.g. "Q initialisation"
n_values, n_seeds = len(values), len(seeds)



matches = np.flatnonzero(np.isclose(values, VALUE))

row = int(matches[0])
 
value_runs = runs[row * n_seeds:(row + 1) * n_seeds]

with np.load(os.path.join(OUTPUT_DIR, results["arrays_file"])) as arrays:
    curves = arrays["episode_returns"][row] 


kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
smoothed = np.array([np.convolve(curve, kernel, mode="valid") for curve in curves])
episodes = np.arange(SMOOTHING_WINDOW, SMOOTHING_WINDOW + smoothed.shape[1])


title = f"{label} -- {n_seeds} seeds x {results['episodes']:,} episodes"
print(f"loaded {len(runs)} runs from {json_path}")



best_terminal = max(results["terminal_states"], key=lambda t: t["reward"])
best_cell = tuple(best_terminal["cell"])

# Did each run's greedy policy actually end up there? (True/False per run.)
reached_best = np.array(
    [tuple(run["greedy_final_cell"]) == best_cell for run in runs]
).reshape(n_values, n_seeds)



fig, ax = plt.subplots(figsize=(7, 4.5))
for run, curve in zip(value_runs, smoothed):
    name = f"seed {run['seed']}"
    ax.plot(episodes, curve, linewidth=1.2, alpha=0.9, label=name)
 
ax.set_xlabel("Episode")
ax.set_ylabel(f"Training return ({SMOOTHING_WINDOW}-episode mean)")
ax.set_title("Return curves for Q_0=0.96")
fig.tight_layout()


 

path1 = os.path.join(OUTPUT_DIR, f"maze_returns_{EXPERIMENT}.png")
fig.savefig(path1, dpi=150)
print(f"\nsaved {path1}")
if fig is not None:
    path2 = os.path.join(OUTPUT_DIR, f"maze_sweep_{EXPERIMENT}_learning_curves.png")
    fig.savefig(path2, dpi=150)
    print(f"saved {path2}")

 
 
path1 = os.path.join(OUTPUT_DIR, f"maze_returns_by_seed_{EXPERIMENT}.png")
fig.savefig(path1, dpi=150)
print(f"\nsaved {path1}")
