import torch.distributions as distrbs
import torch
from .. import actor

def sample_act_single(act_prob):
    return distrbs.Categorical(act_prob).sample()

def sample_act(act_probs): # batched version
    samples = tuple(map(sample_act_single, act_probs))
    return torch.stack(samples)

class PolicyGradActor(actor.Actor):
    def __init__(self, policy=None):
        self.policy = policy
    def __call__(self, obs):
        return sample_act(self.policy(obs))
    def save_model(self, file_path):
        torch.save(self.policy, file_path)
    def load_model(self, file_path):
        self.policy = torch.load(file_path)