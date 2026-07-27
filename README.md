# GridWorld: a configurable finite-MDP environment for RL dissertations

A small, transparent [Gymnasium](https://gymnasium.farama.org/) environment
for studying finite Markov Decision Processes (MDPs), dynamic programming,
and reinforcement learning. Designed for a master's dissertation: the code
prioritises clarity and mathematical traceability over scale.

```
gridworld_project/
    README.md
    pyproject.toml               # package metadata + dependencies
    requirements.txt             # convenience wrapper: `-e .[sb3]`
    gridworld/
        config.py                # GridWorldConfig dataclass (defines an MDP instance)
        env.py                   # GridWorldEnv (Gymnasium environment)
        mining_config.py          # MineWorldConfig: mining nodes + reward/probability schedules
        mining_env.py             # MineWorldEnv: GridWorld + a MINE action with per-node memory
        plotting.py               # static plots: layout, policy, value function, Q-values, learning curve
        animation.py              # episode animation (GIF/MP4)
        dynamic_programming.py    # value iteration, policy iteration (model-based)
        q_learning.py              # tabular Q-learning (model-free)
        examples.py                # ready-made GridWorld and MineWorld MDPs
        registration.py            # Gymnasium ids, e.g. gym.make("MineWorld-v0")
    scripts/
        run_random_agent.py
        run_value_iteration.py
        run_policy_iteration.py
        run_q_learning.py
        run_sb3_dqn.py             # optional: Stable-Baselines3 DQN
        run_mining_world.py        # the MineWorld extension end to end
        make_animation.py
    tests/                        # pytest suite for both environments
    outputs/                      # all generated PNG/GIF/MP4 files land here
```

---

## 1. What is a GridWorld MDP?

A GridWorld is one of the simplest non-trivial examples of a **finite
Markov Decision Process**: an agent occupies a cell in a rectangular grid
and, at each discrete time step, chooses one of four moves (up, right,
down, left). Some cells are *terminal* (the episode ends on entry, e.g. a
goal or a trap); some cells may be blocked (*walls*); moves may
occasionally be corrupted by noise (*slippery* transitions). Despite this
simplicity, GridWorlds exhibit everything needed to discuss the core
theory of sequential decision-making under uncertainty:

- a well-defined, enumerable **state space**,
- a small **action space**,
- an explicit, known **transition function** (so it can be solved exactly
  with dynamic programming, *and* used to generate samples for model-free
  learning),
- a tunable **reward function**, including reward shaping,
- **terminal (absorbing) states**.

Formally, an MDP is the tuple `(S, A, P, R, gamma)`:

| Symbol | Meaning | In this code |
|---|---|---|
| `S` | Set of states | Grid cells `(x, y)`, flattened to `Discrete(width * height)` |
| `A` | Set of actions | `Discrete(4)` = {0: up, 1: right, 2: down, 3: left} |
| `P(s'\|s,a)` | Transition probabilities | `env.transition_probabilities(state, action)` |
| `R(s,a,s')` | Reward function | `step_reward` + wall/terminal/shaping rewards, see `env.py` |
| `gamma` | Discount factor | passed to the solvers (`value_iteration`, `policy_iteration`, `q_learning`), not stored in the env |

The grid uses `(x, y)` coordinates with `(0, 0)` at the bottom-left and `y`
increasing **upwards** (matches the plots: "up" moves to a higher row on
screen).

### States, actions, transitions, rewards, discounting

- **State**: a flattened integer index over grid cells, `state = y * width
  + x` (see `GridWorldEnv.coord_to_state` / `state_to_coord`). Wall cells
  still have valid indices but are never occupied or updated by the
  solvers.
- **Action**: one of `{0: up, 1: right, 2: down, 3: left}`.
- **Transition**: deterministic unless `slip_probability > 0`. When
  slippery, with probability `slip_probability` the environment silently
  substitutes a *uniformly random different* action for the one requested
  -- this is exactly the "slippery ice" idea from Sutton & Barto / OpenAI's
  FrozenLake. Attempting to move into a wall or off the grid leaves the
  agent in place (an "invalid move").
- **Reward**: `step_reward` on every transition (typically a small negative
  cost, to encourage short paths), plus `invalid_move_reward` if the move
  was invalid, plus a terminal reward if the new cell is in
  `terminal_states`, plus a shaping reward if the new cell is in `rewards`.
  See `GridWorldEnv._reward` for the exact composition.
- **Terminal states**: cells in `terminal_states`. Entering one sets
  `terminated=True`; by convention by the solvers, `V(terminal) = 0` since
  no further reward can be accrued (a standard treatment of absorbing
  states in episodic MDPs).
- **Discounting** (`gamma`): *not* stored on the environment (which has no
  notion of "how much the future matters") -- it is a parameter to the
  *solvers* (`value_iteration`, `policy_iteration`, `q_learning`), exactly
  as in the mathematical formulation where gamma only ever appears inside
  the Bellman equations, not the MDP's dynamics themselves.

---

## 2. Installation

`gridworld` is a proper Python package. Install it in **editable mode** so
that edits to the source tree take effect immediately, without reinstalling
and without any `sys.path` juggling in the scripts.

From inside `gridworld_project/`:

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -e ".[sb3]"         # or: pip install -r requirements.txt
```

Requires Python >= 3.10.

Install variants:

```bash
pip install -e .                # core only (no Stable-Baselines3/torch)
pip install -e ".[sb3]"         # + Stable-Baselines3, for scripts/run_sb3_dqn.py
pip install -e ".[sb3,dev]"     # + pytest
```

After this, `import gridworld` works from *any* working directory:

```python
from gridworld.examples import make_easy_gridworld
from gridworld.dynamic_programming import value_iteration
```

Dependencies are declared in `pyproject.toml`: `gymnasium`, `numpy`,
`matplotlib`, `pillow`, plus `stable-baselines3` under the optional `sb3`
extra. All plotting uses matplotlib's headless
`Agg` backend, so nothing here requires a display / X server -- it works
fine over SSH or in CI.

GIF export uses matplotlib's `PillowWriter` (no external dependency). MP4
export uses `FFMpegWriter`, which requires `ffmpeg` to be installed and on
`PATH` (e.g. `brew install ffmpeg` on macOS, `apt install ffmpeg` on
Debian/Ubuntu). If `ffmpeg` is not found, `animate_episode` prints a clear
message and raises rather than failing silently -- just use a `.gif` path
instead.

---

## 3. Running the scripts

With the package installed (section 2), the scripts can be run from any
working directory; they resolve `outputs/` relative to their own location,
so everything still lands in `gridworld_project/outputs/`:

```bash
python scripts/run_random_agent.py       # baseline: uniformly random policy
python scripts/run_value_iteration.py    # exact solution via value iteration
python scripts/run_policy_iteration.py   # exact solution via policy iteration (cross-checked against VI)
python scripts/run_q_learning.py         # model-free tabular learning on the slippery gridworld
python scripts/run_sb3_dqn.py            # optional: deep RL via Stable-Baselines3
python scripts/make_animation.py         # minimal animate_episode usage example
python scripts/run_mining_world.py       # the MineWorld extension (see section 8)
```

Expected files after running the first four (plus `make_animation.py`):

```
outputs/easy_gridworld.png              # the layout of the MDP itself
outputs/value_function.png              # V*(s) from value iteration
outputs/optimal_policy.png              # pi*(s) from value iteration
outputs/value_iteration_agent.gif       # one episode under pi*
outputs/policy_iteration_value_function.png
outputs/policy_iteration_policy.png
outputs/q_learning_curve.png            # episode return vs. episode number
outputs/q_learning_policy.png
outputs/q_learning_q_values.png
outputs/q_learning_agent.gif
outputs/random_agent.gif
outputs/example_trajectory.gif
```

If `ffmpeg` is available, any `animate_episode(..., save_path="....mp4")`
call will produce an MP4 instead of (or alongside) a GIF -- e.g. change the
save path in a script to `outputs/q_learning_agent.mp4`.

`run_sb3_dqn.py` additionally produces `outputs/sb3_dqn_policy.png` and
`outputs/sb3_dqn_agent.gif`.

---

## 4. Creating a custom GridWorld

Every experiment starts from a `GridWorldConfig` (see `gridworld/config.py`):

```python
from gridworld.config import GridWorldConfig
from gridworld.env import GridWorldEnv

config = GridWorldConfig(
    width=5,
    height=5,
    start=(0, 0),
    terminal_states={(4, 4): 1.0, (4, 0): -1.0},  # goal (+1), trap (-1)
    rewards={},                                    # optional shaping rewards
    walls=set(),                                   # optional blocked cells
    step_reward=-0.01,
    invalid_move_reward=-0.05,
    slip_probability=0.0,                          # 0 = deterministic, >0 = "slippery"
    max_steps=100,
    seed=0,
)
env = GridWorldEnv(config)
```

`gridworld/examples.py` provides four ready-made configurations you can
use directly or as templates:

```python
from gridworld.examples import (
    make_easy_gridworld,             # Example 1: 5x5, deterministic
    make_walled_gridworld,            # Example 2: 6x6, obstacles between start/goal
    make_slippery_gridworld,          # Example 3: Example 1 with slip_probability=0.2
    make_reward_shaping_gridworld,    # Example 4: intermediate shaping rewards
)

env = make_easy_gridworld()
```

To modify layout, just change `terminal_states` (goal/trap positions and
their reward), `walls` (blocked cells), `rewards` (shaping), or
`slip_probability` (stochasticity), then re-run any of the scripts against
your new config/environment.

---

## 5. Interpreting the plots

- **`plot_gridworld`**: the raw MDP layout. Grey cells are walls; green/red
  cells are terminal states (goal/trap) with their reward printed;
  light-blue/orange cells are shaping-reward cells; "S" marks the start
  state.
- **`plot_policy`**: one purple arrow per non-terminal, non-wall cell,
  showing the action the policy selects there -- i.e. a visualisation of
  `pi(s)`.
- **`plot_value_function`**: a heatmap of `V(s)`, the expected discounted
  return from each state, with numeric values printed per cell. Terminal
  cells are 0 by convention (no further reward can be accrued once an
  episode ends).
- **`plot_q_values`**: each cell shows four small numbers (one per action,
  labelled with an arrow glyph `^ > v <`), the learned `Q(s, a)`; the
  action with the highest value is highlighted in bold red.
- **`plot_learning_curve`**: episode return vs. episode number, with a
  moving-average overlay, for judging whether a model-free method (e.g.
  Q-learning) is actually learning.
- **Animations** (`animate_episode`): the agent's position each timestep,
  a faded trail of its path so far, and a title that updates with the
  timestep, action taken, reward received, and cumulative reward.

---

## 6. Value iteration vs. policy iteration vs. Q-learning

All three solve the same underlying MDP, but differ in what they assume and how they compute an answer.

| | Value iteration | Policy iteration | Q-learning |
|---|---|---|---|
| Category | model-based (dynamic programming) | model-based (dynamic programming) | model-free (temporal-difference RL) |
| Requires knowledge of P(s'\|s,a), R? | Yes | Yes | No -- learns from sampled (s,a,r,s') transitions only |
| What it iterates on | the value function V(s) directly, via the Bellman *optimality* equation | a full evaluate-then-improve cycle: exactly solve V_pi (Bellman *expectation* equation), then act greedily w.r.t. it | the action-value function Q(s,a), updated online after every environment step |
| Convergence guarantee | to V* (and hence an optimal policy) for a finite MDP | to an optimal policy in a finite number of outer iterations | to Q* with probability 1 under standard step-size/exploration conditions |
| Needs to interact with the environment? | No -- pure computation over the model | No -- pure computation over the model | Yes -- must actually run episodes (`env.step`) |

Concretely in this codebase:

- `value_iteration(env, gamma, theta, max_iterations)` in
  `dynamic_programming.py` repeatedly applies
  `V(s) <- max_a sum_{s'} P(s'|s,a) [R + gamma * V(s')]` to every
  non-terminal state until the largest per-sweep change drops below
  `theta`, then extracts the greedy policy.
- `policy_iteration(env, gamma, theta, max_iterations)` alternates *policy
  evaluation* (solve `V_pi` exactly for the current policy) and *policy
  improvement* (make the policy greedy w.r.t. `V_pi`) until the policy
  stops changing. It should converge to the same `V*`/`pi*` as value
  iteration -- `scripts/run_policy_iteration.py` checks this explicitly.
- `q_learning(env, episodes, alpha, gamma, epsilon, epsilon_min,
  epsilon_decay, seed)` in `q_learning.py` never touches
  `transition_probabilities`; it only calls `env.step` and updates
  `Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]`
  using an epsilon-greedy behaviour policy. This is what makes it suitable
  for environments where the transition model is unknown or too complex to
  enumerate -- the price is that it needs many episodes of experience
  instead of a handful of full sweeps over the state space.

`scripts/run_sb3_dqn.py` additionally shows the model-free idea taken to
function approximation (a neural network estimates `Q(s,a)` instead of a
table), via Stable-Baselines3's DQN implementation, for students who want
to compare tabular RL against a standard deep RL baseline on the same MDP.

---

## 7. Code quality notes

- Python 3.10+, type-annotated throughout.
- `GridWorldEnv` passes Stable-Baselines3's `check_env`.
- All plotting uses the headless `Agg` backend; no display required.
- Random seeds are threaded through `reset(seed=...)`, `GridWorldConfig.seed`,
  and the `seed` arguments of `q_learning` / DQN for reproducibility.
- The environment records a full trajectory (`env.trajectory`) during each
  episode, which `animate_episode` consumes by default.
- `pytest` covers both environments (`python -m pytest`).

---

## 8. MineWorld: the mining extension

`MineWorldEnv` (`gridworld/mining_env.py`, configured by
`gridworld/mining_config.py`) adds a fifth action, `MINE`, and a set of
*mining nodes*. Standing on a node and executing `MINE` is a gamble whose
odds worsen — and whose payoff grows — every time that node is mined.

```python
from gridworld.examples import make_mining_gridworld
from gridworld.dynamic_programming import value_iteration

env = make_mining_gridworld()
print(env.config.describe_mining_schedule())
V, policy, history = value_iteration(env, gamma=0.95)
```

### The state must include the mining counts

The payoff of the *next* attempt depends on how often each node has already
been mined, so the counts are part of the state — a single scalar "streak"
would not be Markov once several nodes can be mined independently:

```
state = (agent_position, mining_count_at_node_1, ..., mining_count_at_node_K)
```

Each count is capped at `max_mining_count = C`, so the flattened state space
has `width * height * (C + 1) ** K` entries, and
`components_to_state` / `state_to_components` are exact inverses over all of
it. Counts record **attempts** (including the final, fatal one), reset to
zero in `reset()`, and are only ever touched when the *executed* action is
`MINE` on a node — never by entering, leaving, bumping into, or slipping
past one. Once a node's count reaches `C`, it stays there and every further
attempt is evaluated at the saturated level `m = C + 1`.

> **Scalability.** That state space grows *exponentially* in the number of
> mining nodes `K`. It is fine for the teaching examples (the default is
> 5 × 5 × 4² = 400 states), but `MineWorldConfig` warns above 100k states
> and refuses to build above 2M. Beyond a handful of nodes, tabular methods
> stop being the right tool.

### The mining gamble

On the `m`-th attempt at a node the environment pays out with probability
`p(m)` and otherwise collapses, ending the episode:

| outcome | probability | reward | terminates? |
|---|---|---|---|
| strike | `p(m)` | `step_reward + R_positive(m)` | no |
| collapse | `1 - p(m)` | `step_reward + mining_failure_reward(m)` | **yes** |

`p(m)` is non-increasing (`"constant"`, `"linear"` or `"geometric"`
schedules) while `R_positive(m)` grows with `m`. Both are configured with
plain named parameters rather than callables, so a config stays printable
and serialisable. A collapse is signalled purely by the transition's
`terminated` flag — the node itself is never a terminal cell, since a
*successful* mine has to leave the agent free to act from that same square.

### Exact dynamic programming vs. continuous rewards

`transition_probabilities()` returns a finite list of
`(probability, next_state, reward, terminated)` tuples, which cannot
represent a continuous reward law. So the payout distributions split in two:

| distribution | `step()` | `transition_probabilities()` |
|---|---|---|
| `deterministic`, `categorical` | ✅ | ✅ enumerated exactly |
| `normal`, `lognormal`, `gamma` | ✅ | ❌ `ContinuousRewardModelError` |

The random reward is **never** silently replaced by its expectation, because
that would make the DP model describe a different MDP from the one `step()`
samples. Both views are generated by one side-effect-free enumerator
(`_transition_branches`), so they cannot drift apart; querying the model
never mutates the environment.

### Ready-made examples

| factory | Gymnasium id | payout | slip | exact DP? |
|---|---|---|---|---|
| `make_mining_gridworld` | `MineWorld-v0` | deterministic | 0.0 | yes |
| `make_risky_mining_gridworld` | `MineWorldRisky-v0` | categorical | 0.1 | yes |
| `make_continuous_mining_gridworld` | `MineWorldContinuous-v0` | log-normal | 0.0 | no |

The default schedule is tuned so mining is a real decision rather than a
free lunch — the first attempt at a fresh node is clearly worth taking, the
second is exactly break-even, and the third destroys value:

```
  m   p(m)    R+(m)    E[reward of the attempt]
  1   0.95     1.0     +0.45
  2   0.80     2.5      0.00
  3   0.65     4.0     -0.90
  4+  0.50     5.5     -2.25
```

Value iteration duly mines each node exactly once and then walks to the goal.

### Plots and animation

Mining nodes are shaded and marked with a diamond plus the current attempt
count, placed in the cell corner so it stays readable when the agent is
standing on the node. Because a MineWorld policy/value/Q array has one entry
per *(cell, count-vector)* pair, `plot_policy`, `plot_value_function` and
`plot_q_values` take a `mining_counts=` argument selecting which slice to
draw (default: all zeros); `MINE` is drawn as a cross rather than an arrow.
`plot_mining_schedule(config)` charts the risk/reward trade-off against `m`.

Trajectories store one immutable count snapshot per frame, so
`animate_episode` shows the counts *as they were at that time*, with the
reset state as frame 0 and a red marker plus a "MINE COLLAPSED" annotation
on a fatal final frame.
