from functools import partial
import torch

def reward_to_go(trajecs_rewards, decay):
    advantages_map = map(partial(trajec_advantage, reward_decay=decay, advantage_fn=decayed_advantage), trajecs_rewards)
    decayed_advs = tuple(map(torch.tensor, advantages_map))
    return decayed_advs

def reward_to_go_mean_baseline(trajecs_rewards, decay):
    rew_to_go = reward_to_go(trajecs_rewards, decay)
    mean_trajecs = torch.mean(torch.cat(decayed_advs, dim=0))
    mean_baseline_adv = tuple(map(lambda trajec: trajec - mean_trajecs, decayed_advs))
    return mean_baseline_adv

def trajec_advantage(trajec_rewards, reward_decay, advantage_fn):
    trajec_rewards = torch.stack(trajec_rewards)
    return tuple(map(lambda i: advantage_fn(trajec_rewards[i:], reward_decay), range(len(trajec_rewards))))

def decayed_advantage(rew_traj, decay):
    return torch.sum( decay ** torch.arange(len(rew_traj)) * rew_traj )