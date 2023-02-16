from torch import nn
import torch

# for an observation, output probability of taking an action (same as DeepQLunar but with a softmax activation)
class PolicyGradNNLunar(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()

        self.in_layer = nn.Sequential(nn.Linear(obs_size, 64, dtype=torch.double), nn.LeakyReLU())
        self.inner = nn.Sequential(
            nn.Linear(64, 64, dtype=torch.double), nn.LeakyReLU(),
        )
        self.out_layer = nn.Sequential(nn.Linear(64, action_size, dtype=torch.double), nn.Softmax(dim=1),)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.inner(x)
        x = self.out_layer(x)
        return x