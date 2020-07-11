from lib import get_arguments, model
import gfootball.env as football_env
import torch
import numpy as np


def get_tensor(obs):
    return torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32)


if __name__ == '__main__':
    args = get_arguments.get_args()
    model_path = args.save_dir + args.env_name + '/model.pt'
    test_env = football_env.create_environment(env_name=args.env_name, stacked=True, render=True)
    network = model.OutputNet(test_env.action_space.n)
    network.load_state_dict(torch.load(model_path))
    # network.eval()

    # start to test
    obs = test_env.reset()
    for _ in range(100):
        obs_tensor = get_tensor(np.expand_dims(obs, 0))
        with torch.no_grad():
            _, pi = network(obs_tensor)
        action = torch.argmax(pi, dim=1).item()
        obs, reward, done, _ = test_env.step(action)
        if done:
            obs = test_env.reset()
    test_env.close()
