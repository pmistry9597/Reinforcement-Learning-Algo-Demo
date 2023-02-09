############# Screwing around with random agent in lunar lander environment
import math
import os
import imageio
from datetime import datetime
import statistics as stats
from functools import reduce
import torch
import numpy as np

from deep_q_trial import normalize

def measure_ep(agent, env, obs_norm, max_steps=math.inf, record_frames=False):
    obs0, info0 = env.reset()
    rewards = []
    frames = []

    obs_mean, obs_scale = obs_norm
    # rew_mean, rcew_scale = rew_norm

    # measurements pls? rewards over time, cumulative reward per episode, steps required
    obs = obs0 #obs will be observation variable used to store previously observed state to choose next action
    obs = normalize(obs, obs_mean, obs_scale)
    termin = False
    steps = 0
    while not termin and steps < max_steps:
        prev_obs = obs
        act = agent(torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.double))
        obs, reward, termin, trunc, info  = env.step(act.item())
        obs = normalize(obs, obs_mean, obs_scale)
        rewards.append(reward)
        if record_frames:
            frames.append(env.render())
        steps += 1

    return rewards, frames, termin

# pass in empty iterable for record_eps if you don't want to record
def measure_for_eps(episodes, agent, env, obs_norm, max_steps=math.inf, record_eps=set(), record_dir="recorded", prefix=""):
    score_seqs = []
    if len(record_eps) > 0 and not os.path.isdir(record_dir):
        os.makedirs(record_dir, exist_ok=True)

    for e in range(episodes):
        record_frames = e in record_eps
        rewards, frames, termin = measure_ep(agent, env, obs_norm, max_steps=max_steps, record_frames=record_frames)
        if record_frames:
            imageio.mimwrite(os.path.join(record_dir, "{}_ep_{}_{}.gif").format(prefix, e, datetime.now()), frames, fps=30) #save gif of images
        score_seqs.append(rewards)

    return score_seqs

def calculate_bellman(i_r, decay): # expects index and reward tuple, reward decay
    i, r = i_r
    return r * decay ** i

def get_stats_on_seq(seq):
    mean = stats.mean(seq)
    std = stats.stdev(seq)

    return mean, std

def stats_score_seq(score_seq, rew_decay):
    steps = list(map(lambda l: len(l), score_seq))
    rew_seq_means = list(map(stats.mean, score_seq))
    bellman_seq = list(map(lambda seq: reduce(lambda f, i_r: f + calculate_bellman(i_r, rew_decay), enumerate(seq), 0), score_seq))
    end_scores = list(map(lambda seq: seq[-1], score_seq))

    return rew_seq_means, end_scores, bellman_seq, steps