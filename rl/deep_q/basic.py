from .. import actor
import torch

def select_q(q_vals):
    max_vals, _indices = torch.max(q_vals, dim=1)
    return max_vals

def act_from_q(q): # take largest q value as action, behaves as if q is batched
    return torch.argmax(q, dim=1)

class DeepQActor(actor.Actor):
    def __init__(self, q_func):
        self.q_func = q_func
    def __call__(self, obs):
        return act_from_q(self.q_func(obs))
    def save_model(self, file_path):
        torch.save(self.q_func, file_path)
    def load_model(self, file_path):
        self.q_func = torch.load(file_path)