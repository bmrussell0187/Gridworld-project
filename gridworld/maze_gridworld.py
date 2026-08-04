from __future__ import annotations

from .config import GridWorldConfig
from .env import GridWorldEnv



def make_maze_gridworld(seed: int | None = 0, render_mode: str | None = None) -> GridWorldEnv:
    walls: set[tuple[int, int]] = set()
    walls |= {(0, y) for y in (3, 4, 5)}
    walls |= {(1, y) for y in (1, 3, 7, 8)}
    walls |= {(2, y) for y in (1, 3, 5, 6, 7, 8)}
    walls |= {(3, y) for y in (1, 5)}
    walls |= {(4, y) for y in (1, 2, 3, 4, 5, 7, 9)}
    walls |= {(5, y) for y in (1, 4, 5, 7, 9)}
    walls |= {(6, y) for y in (1, 2, 4, 7)}
    walls |= {(7, y) for y in (1, 4, 6)}
    walls |= {(8, y) for y in (3, 4, 6, 8)}
    walls |= {(9, y) for y in (0, 1, 2, 3, 6)}

#add shaping rewards back in as comment

    config = GridWorldConfig(
        width=10,
        height=10,
        start=(0, 9),
        terminal_states={
            (5, 2): 10.0,
            (7, 7): 1.0,
            (7, 8): -1.0,
            (9, 4): -4.0,
        },
        rewards={},
        walls=walls,
        step_reward=-0.01,
        invalid_move_reward=-0.05,
        slip_probability=0.0,
        max_steps=200,
        seed=seed,
    )
    


    return GridWorldEnv(config, render_mode=render_mode)

