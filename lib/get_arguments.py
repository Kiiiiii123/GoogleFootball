import argparse


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gamma', type=float, default=0.993, help='the discount factor of RL')
    parse.add_argument('--cuda', action='store_true', help='use cuda to do the training')
    parse.add_argument('--env-name', type=str, default='academy_emtpy_goal_close', help='the environment name')
    parse.add_argument('--eps', type=float, default=1e-5, help='param for Adam optimizer')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder to save models')
    parse.add_argument('--log-dir', type=str, default='logs/', help='the folder to save log')
    parse.add_argument('--num-workers', type=int, help='the number of workers to collect samples')
    parse.add_argument('--nsteps', type=int, default=128, help='the steps to collect samples')
    parse.add_argument('--total-frames', type=int, default=int(2e6), help='the total frames for training')
    parse.add_argument('--lr-decay', action='store_true', help='if use learning rate decay during training')
    parse.add_argument('--tau', type=float, default=0.95, help='the generalized advantage estimator coefficient')
    parse.add_argument('--batch-size', type=int, default=8, help='the batch size of updating')
    parse.add_argument('--epoch', type=int, default=4, help='the epoch during training')
    parse.add_argument('--clip', type=float, default=0.27, help='the ratio clip param')
    parse.add_argument('--vloss-coef', type=float, default=0.5, help='the coefficient of value loss')
    parse.add_argument('--eloss-coef', type=float,default=0.01, help='the coefficient of entropy loss')
    parse.add_argument('--max-grad-norm', type=float, default=0.5, help='the clip grad norm param')
    parse.add_argument('--display-intreval', type=int, default=10, help='the interval that display log information')
    args = parse.parse_args()

    return args
