import numpy as np
import torch
from torch import optim
import copy
import os

from lib import utils


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
        self.batch_obs_shape = (self.args.num_workers * self.args.nsteps, ) + self.envs.observation_space.shape
        self.obs = np.zeros((self.args.num_workers, ) + self.envs.observation_space.shape, dtype=self.envs.observation_space.dtype.name)
        self.obs[:] = self.envs.reset()

        self.dones = [False for _ in range(self.args.num_workers)]
        self.logger = utils.config_logger(self.log_path)

    def learn(self):
        # get the reward to calculate other information
        episode_rewards = torch.zeros([self.args.num_workers, 1])
        final_rewards = torch.zeros([self.args.num_workers, 1])

        iter_num = self.args.total_frames // (self.args.num_workers * self.args.nsteps)
        for iteration in range(iter_num):
            mb_obs, mb_rewards, mb_actions, mb_dones, mb_values = [], [], [], [], []
            if self.args.lr_decay:
                self.adjust_learning_rate(iteration, iter_num)
            for step in range(self.args.nsteps):
                with torch.no_grad():
                    obs_tensor = self.get_tensor(self.obs)
                    values, pis = self.net(obs_tensor)
                actions = utils.select_actions(pis)

                # start to store information
                mb_obs.append(self.obs)
                mb_actions.append(actions)
                mb_dones.append(self.dones)
                mb_values.append(values.detach().cpu().numpy().squeeze())

                # execute the actions in the environment
                obs, rewards, dones, _ = self.envs.step(actions)
                self.dones = dones
                mb_rewards.append(rewards)

                # clear the observation
                for n, done in enumerate(dones):
                    if done:
                        obs[n] = obs[n] * 0
                self.obs = obs

                # process the reward
                rewards = torch.tensor(np.expand_dims(np.stack(rewards), 1), dtype=torch.float32)
                episode_rewards += rewards
                masks = torch.tensor([[0.0] if done else [1.0] for done in dones], dtype=torch.float32)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks

            # process the rollouts
            mb_obs = np.asarray(mb_obs, dtype=np.float32)
            mb_actions = np.asarray(mb_actions, dtype=np.float32)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            mb_values = np.asarray(mb_values, dtype=np.float32)

            # calculate the last state value
            with torch.no_grad():
                obs_tensor = self.get_tensor(self.obs)
                last_values, _ = self.net(obs_tensor)
                last_values.detach().cpu().numpy().suqeeze()

            # start to compute advantages



    def get_tensor(self, obs):
        obs_tensor = torch.tensor(np.transpose(obs, (0, 3, 1, 2)), dtype=torch.float32)
        if self.args.cuda:
            obs_tensor.cuda()
        return obs_tensor

    def adjust_learning_rate(self, iteration, iter_num):
        lr_frac = 1 - (iteration / iter_num)
        adjust_lr = self.args.lr * lr_frac
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adjust_lr

