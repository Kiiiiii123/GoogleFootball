from lib import get_arguments, model
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from ppo_agent import PPOAgent
import gfootball.env as football_env


def create_single_football_env(args)
    env = football_env.create_environment(env_name=args.env_name, stacked=True, )
    return env

if __name__ == '__main__':
    args = get_arguments.get_args()
    # create vectorized environments for multi-processing
    envs = SubprocVecEnv([lambda _i=i: create_single_football_env(args) for i in range(args.num_workers)])
    network = model.OutputNet(envs.action_space.n)
    agent = PPOAgent(envs, args, network)
    agent.learn()
    envs.close()
