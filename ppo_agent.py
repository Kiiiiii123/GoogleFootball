import numpy
import torch
from torch import optim
import copy
import os


class PPOAgent:
    def __init__(self, envs, args, net):
        self.envs = envs
        self.args = args

        self.net = net
        self.old_net = copy.deepcopy(net)
        if self.args.cuda:
            self.net.cuda()
            self.old_net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, eps=self.args.eps)

        # check saving folders
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.exists(self.args.log_dir):
            os.mkdir(self.args.log_dir)
        self.log_path = self.args.log_dir + self.args.env_name + '.log'

        # get the observation


