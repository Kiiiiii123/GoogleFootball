import argparse


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('gamma', type=float, default=0.993, help='The discount factor of RL')
    args = parse.parse_args()

    return args
