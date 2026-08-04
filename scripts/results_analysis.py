from __future__ import annotations

import os

import matplotlib.pyplot as plt

import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

save_path = os.path.join(OUTPUT_DIR, "maze_multiseed_null_init.npz")

data=np.load(save_path)

print(np.where(data["episode_returns_null_maze"]>5))

#plt.pyplot.scatter(x=)

data["episode_returns_null_maze"].shape