import logging
import torch
from torch.distributions.categorical import Categorical
import numpy as np


def config_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    basic_format = '%(message)s'
    formatter = logging.Formatter(basic_format)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    fhlr = logging.FileHandler(log_dir)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger


def select_actions(pi):
    actions = Categorical(pi).sample()
    return actions.detach().cpu().numpy().squeeze()


def evaluate_actions(pi, actions):
    distr = Categorical(pi)
    log_prob = distr.log_prob(actions).unsqueeze(-1)
    entropy = distr.entropy().mean()
    return log_prob, entropy
