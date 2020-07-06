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

        # init the weight
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))

        # init the bias
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        nn.init.constant_(self.fc1.bias.data, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 5 * 8)
        x = F.relu(self.fc1(x))

        return x


class OutputNet(nn.Module):
    def __init__(self, act_size):
        super(OutputNet, self).__init__()
        self.backbone = BackboneNet()
        self.actor = nn.Linear(512, act_size)
        self.critic = nn. Linear(512, 1)

        # init the actor
        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        nn.init.constant_(self.actor.bias.data, 0)

        # init the critic
        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.constant_(self.critic.bias.data, 0)

    def forward(self, inputs):
        x = self.backbone(inputs)
        pi = F.softmax(self.actor(x), dim=1)
        value = self.critic(x)
        return pi, value

