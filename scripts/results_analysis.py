from __future__ import annotations

import os

import matplotlib.pyplot as plt

import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

save_path_1 = os.path.join(OUTPUT_DIR, "maze_multiseed_null_init.npz")

data_1=np.load(save_path_1)

print(greedy_1)