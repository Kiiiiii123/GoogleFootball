import argparse


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gamma', type=float, default=0.993, help='The discount factor of RL')
    parse.add_argument('--cuda', action='store_true', help='use cuda to do the training')
    parse.add_argument('--env-name', type=str, default='academy_emtpy_goal_close', help='the environment name')
    parse.add_argument('--eps', type=float, default=1e-5, help='param for Adam optimizer')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder to save models')
    parse.add_argument('--log-dir', type=str, default='logs/', help='the folder to save log')
    parse.add_argument('--num-workers', type=int, help='the number of workers to collect samples')
    parse.add_argument('--nsteps', type=int, default=128, help='the steps to collect samples')
    args = parse.parse_args()

    return args
