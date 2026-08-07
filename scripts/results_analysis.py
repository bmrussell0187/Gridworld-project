"""Summarise and plot the runs saved by general_maze_experiment.py.

How to use: set EXPERIMENT below to the experiment you ran, then run this file.

The JSON holds every seeded run separately, so all the summarising happens here.
That matters: in the interesting range the seeds of one setting do not agree --
some agents find the +10 goal and some settle for the +1 goal, with nothing in
between -- so their mean is a number no single run ever achieved. The honest
summary of a split like that is "what fraction of seeds found the good goal",
which is what METRIC = "success_rate" plots.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

# ===========================================================================
# SETTINGS -- this is the only part you need to edit
# ===========================================================================

EXPERIMENT = "epsilon_min"            # must match EXPERIMENT in general_maze_experiment.py

METRIC = "success_rate"          # what to plot against the swept parameter:
                                 #   "success_rate"       fraction of seeds that
                                 #                        reached the best goal
                                 #   "greedy_return"      reward of the learned policy
                                 #   "greedy_length"      steps the learned policy took
                                 #   "final_train_return" mean return late in training

SHOW_LEARNING_CURVES = True      # also plot return-per-episode during training?
SMOOTHING_WINDOW = 100           # moving average, in episodes, for those curves
N_CURVES = 16                     # how many of the values to draw curves for

SAVE_PLOTS = True    # True: save PNGs to outputs/. False: show them.

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

# ===========================================================================


# --- Load ------------------------------------------------------------------

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

title = f"{label} -- {n_seeds} seeds x {results['episodes']:,} episodes"
print(f"loaded {len(runs)} runs from {json_path}")


def field_by_value_and_seed(name):
    """Pull one field out of every run, as an array of shape (values, seeds).

    The runs are stored value by value, then seed by seed, so reshaping puts
    each setting on its own row and each seed in its own column. Averaging over
    the seeds is then np.mean(..., axis=1).
    """
    return np.array([run[name] for run in runs]).reshape(n_values, n_seeds)


greedy_return = field_by_value_and_seed("greedy_return")
greedy_length = field_by_value_and_seed("greedy_length")
final_train_return = field_by_value_and_seed("final_train_return")

# Which terminal cell pays the most? That is the goal we hope the agent finds.
best_terminal = max(results["terminal_states"], key=lambda t: t["reward"])
best_cell = tuple(best_terminal["cell"])

# Did each run's greedy policy actually end up there? (True/False per run.)
reached_best = np.array(
    [tuple(run["greedy_final_cell"]) == best_cell for run in runs]
).reshape(n_values, n_seeds)



# --- Summarise each setting across its seeds -------------------------------

print(f"\n{results['param']:>10}  {'median':>7} {'mean':>7} {'min':>7} {'max':>7}   "
      f"reached {best_cell}")
for i, value in enumerate(values):
    row = greedy_return[i]
    print(f"{value:>10g}  {np.median(row):>+7.2f} {row.mean():>+7.2f} "
          f"{row.min():>+7.2f} {row.max():>+7.2f}   {reached_best[i].sum():>2d}/{n_seeds}")


# --- Plot 1: the chosen metric against the swept parameter ----------------

fig1, ax1 = plt.subplots(figsize=(7, 4.5))

if METRIC == "success_rate":
    successes = reached_best.sum(axis=1)
    rate = successes / n_seeds

    # A fraction out of n_seeds is uncertain, and most so near 50%. This is the
    # Wilson 95% interval -- unlike mean +/- s.d. it stays inside 0..1 and does
    # not collapse to zero width at 0/20 or 20/20.
    z = 1.96
    centre = (successes + z**2 / 2) / (n_seeds + z**2)
    half = (z / (n_seeds + z**2)) * np.sqrt(successes * (n_seeds - successes) / n_seeds + z**2 / 4)

    #ax1.fill_between(values, centre - half, centre + half, color="tab:blue", alpha=0.25,
    #                 label="95% interval")
    ax1.plot(values, rate, "o-", color="tab:blue", linewidth=2, markersize=4,
             label=f"fraction of {n_seeds} seeds")
    ax1.set_ylabel(f"Seeds whose policy reached {best_cell}")
    ax1.set_ylim(-0.05, 1.05)
else:
    metric = {
        "greedy_return": greedy_return,
        "greedy_length": greedy_length,
        "final_train_return": final_train_return,
    }[METRIC]
    ylabel = {
        "greedy_return": "Total reward (greedy policy)",
        "greedy_length": "Steps taken (greedy policy)",
        "final_train_return": "Mean training return, late in training",
    }[METRIC]

    # Every seed is drawn, so a setting whose seeds disagree looks disagreeing.
    # Mean and median are both shown: where they part company, the seeds have
    # split into groups and neither line describes a typical run.
    ax1.plot(values, metric, ".", color="lightgray", markersize=5, label=None)
    ax1.plot(values, np.median(metric, axis=1), color="tab:blue", linewidth=2, label="median seed")
    ax1.plot(values, np.mean(metric, axis=1), "--", color="tab:orange", linewidth=2, label="mean of seeds")
    ax1.set_ylabel(ylabel)

ax1.set_xlabel(label)
ax1.set_title(title)
ax1.legend()
fig1.tight_layout()


# --- Plot 2: learning curves for a few of the values ----------------------

fig2 = None

if SHOW_LEARNING_CURVES:
    with np.load(os.path.join(OUTPUT_DIR, results["arrays_file"])) as arrays:
        curves = arrays["episode_returns"]      # (values, seeds, episodes)

    # Average over the seeds, then smooth: raw episode returns are far too noisy.
    seed_mean = np.mean(curves, axis=1)
    kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
    smoothed = np.array([np.convolve(row, kernel, mode="valid") for row in seed_mean])
    episodes = np.arange(SMOOTHING_WINDOW, SMOOTHING_WINDOW + smoothed.shape[1])

    # Pick N_CURVES values spread evenly across the sweep, so the plot stays readable.
    chosen = np.unique(np.linspace(0, n_values - 1, N_CURVES).astype(int))
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(chosen)))

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for colour, i in zip(colours, chosen):
        ax2.plot(episodes, smoothed[i], color=colour, linewidth=1.5,
                 label=f"{results['param']} = {values[i]:g}")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel(f"Training return ({SMOOTHING_WINDOW}-episode mean)")
    ax2.set_title(title)
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig2.tight_layout()


# --- Show or save ----------------------------------------------------------

if SAVE_PLOTS:
    path1 = os.path.join(OUTPUT_DIR, f"maze_sweep_{EXPERIMENT}_{METRIC}.png")
    fig1.savefig(path1, dpi=150)
    print(f"\nsaved {path1}")
    if fig2 is not None:
        path2 = os.path.join(OUTPUT_DIR, f"maze_sweep_{EXPERIMENT}_learning_curves.png")
        fig2.savefig(path2, dpi=150)
        print(f"saved {path2}")
else:
    plt.show()
