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
                last_obs_tensor = self.get_tensor(self.obs)
                last_values, _ = self.net(last_obs_tensor)
                last_values = last_values.detach().cpu().numpy().suqeeze()

            # start to compute advantages
            mb_returns = np.zeros_like(mb_rewards)
            mb_advs = np.zeros_like(mb_rewards)
            last_gae = 0.0
            for i in reversed(range(self.args.nsteps)):
                if i == self.args.nsteps - 1:
                    next_terminal = 1.0 - self.dones
                    next_values = last_values
                else:
                    next_terminal = 1.0 - mb_dones[i + 1]
                    next_values = mb_values[i + 1]
                delta = mb_rewards[i] + self.args.gamma * next_terminal * next_values - mb_values[i]
                mb_advs[i] = last_gae = delta + self.args.gamma * self.args.tau * next_terminal * last_gae
            mb_returns = mb_advs + mb_values

            # process the rollouts again
            mb_obs = mb_obs.swapaxes(0, 1).reshape(self.batch_obs_shape)
            mb_actions = mb_actions.swapaxes(0, 1).flatten()
            mb_advs = mb_advs.swapaxes(0, 1).flatten()
            mb_returns = mb_returns.swapaxes(0, 1).flatten()

            # update the network
            self.old_net.load_state_dict(self.net.state_dict())
            policy_loss, value_loss, entropy_loss = self.update_network(mb_obs, mb_actions, mb_returns, mb_advs)




    def update_network(self, obs, actions, returns, advantages):
        indexes = np.arange(obs.shape[0])
        batch_num = obs.shape[0] // self.args.batch_size
        for _ in range(self.args.epoch):
            np.random.shuffle(indexes)
            for start in range(0, obs.shape[0], batch_num):
                end = start + batch_num
                index_slice = indexes[start:end]

                # get the mini-batches
                mb_obs = obs[index_slice]
                mb_actions = actions[index_slice]
                mb_returns = returns[index_slice]
                mb_advs = advantages[index_slice]

                # convert the mini-batches to tensor
                mb_obs = self.get_tensor(mb_obs)
                mb_actions = torch.tensor(mb_actions, dtype=torch.float32)
                mb_returns = torch.tensor(mb_returns, dtype=torch.float32).squeeze(1)
                mb_advs = torch.tensor(mb_advs, dtype=torch.float32).squeeze(1)
                # normalize the advantage
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

                if self.args.cuda:
                    mb_actions = mb_actions.cuda()
                    mb_returns = mb_returns.cuda()
                    mb_advs = mb_advs.cuda()

                # value loss
                mb_values, pis = self.old_net(mb_obs)
                value_loss = (mb_returns - mb_values).pow(2).mean()

                # policy loss and entropy loss
                with torch.no_grad():
                    _, old_pis = self.old_net(mb_obs)
                    old_log_prob, _ = utils.evaluate_actions(old_pis, mb_actions)
                    old_log_prob = old_log_prob.detach()
                log_prob, entropy_loss = utils.evaluate_actions(pis, mb_actions)
                prob_ratio = torch.exp(log_prob - old_log_prob)






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

