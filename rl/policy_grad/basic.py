import torch.distributions as distrbs
import torch
import torch.nn.functional as nn_func
from .. import actor

def sample_act_single(act_logits):
    # print("prob",nn_func.softmax(act_logits, dim=0))
    return distrbs.Categorical(nn_func.softmax(act_logits, dim=0)).sample()

def sample_act(act_logits): # batched version
    # print(act_logits)
    samples = tuple(map(sample_act_single, act_logits))
    # print(samples)
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