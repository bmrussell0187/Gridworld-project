import time
from gridworld.examples import make_deep_mining_gridworld
from gridworld.dynamic_programming import value_iteration

env = make_deep_mining_gridworld()
t0 = time.time()
V, policy, history = value_iteration(env, gamma=0.99, theta=1e-6)
print(f"{time.time()-t0:.1f}s, {len(history)} sweeps, {len(env._transition_cache):,} cached")

MINE = 4
for node in env.mining_nodes:
    i = env.mining_node_index(node)
    depth = next(c for c in range(20)
                 if policy[env.components_to_state(node, tuple(
                     c if j == i else 0 for j in range(len(env.mining_nodes))))] != MINE)
    print(f"node {node}: mines {depth} times before walking away")