from gridworld.examples import make_deep_mining_gridworld
env = make_deep_mining_gridworld()
print(env.config.describe_mining_schedule())