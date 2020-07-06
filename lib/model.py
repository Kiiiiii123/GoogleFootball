import torch
from torch import nn
from torch.nn import functional as F


class BackboneNet(nn.Module):
    def __init__(self):
        super(BackboneNet, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4 , stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.fc1 = nn.Linear(32 * 5 * 8, 512)





class OutputNet(nn.Module):
