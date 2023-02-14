import torch.nn as nn
import torch

# note that this DeepQ Net does not take sequences of states and actions unlike the DeepMind paper
# this is because this problem is not partially observed (POMDP), so only current state is needed (as in MDP)
class DeepQLunar(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()

        self.in_layer = nn.Sequential(nn.Linear(obs_size, 64, dtype=torch.double), nn.LeakyReLU())
        self.inner = nn.Sequential(
            nn.Linear(64, 64, dtype=torch.double), nn.LeakyReLU(),
        )
        self.out_layer = nn.Sequential(nn.Linear(64, action_size, dtype=torch.double), ) # nn.Softmax(dim=1),)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.inner(x)
        x = self.out_layer(x)
        return x