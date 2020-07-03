import gfootball.env as football_env

env = football_env.create_environment(
    env_name='11_vs_11_stochastic',
    render=True
)

state = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

state_dim = env.observation_space.shape
print(state_dim)

n_actions = env.action_space.n
print(n_actions)


