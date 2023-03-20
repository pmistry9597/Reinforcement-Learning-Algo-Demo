import torch

class Actor:
    def __call__(self, obs):
        pass

    def save_model(self, file_path):
        pass

    def load_model(self, file_path):
        pass

class ExternalActor(Actor):
    def __init__(self, retrieve_act):
        self.retriever = retrieve_act
    def __call__(self, obs):
        return self.retriever(obs)

import random
class RandomActor(Actor):
    def __init__(self, action_space):
        self.action_space = action_space
    def __call__(self, obs):
        return torch.tensor(self.action_space.sample())