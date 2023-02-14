import gymnasium as gym
from datetime import datetime
from functools import partial
import torch

############# Screwing around with training an agent in lunar lander environment, not meant to be a final product
# the file of cursed long functions and some redundant code - I wanted to get something working!

import rl.deep_q.lunar.model as lunar_dqn
from rl.deep_q import basic as deep_q_base
from rl.deep_q.trainer import DeepQTrainer
from rl.train_generic import train_for_eps, ep_r, measurement_r
import measure

def ep_policy(epis_no, eps_decay, eps_min):
    return max((eps_decay ** epis_no), eps_min)

import pickle
import os

def train_once(trial_no, start_time, hyperparam_list):
    (episodes, target_update_steps, 
        buffer_len_to_start, max_steps, 
        steps_for_update, reward_decay,
        eps_min, lr, eps_decay,
        buffer_sample_len
        ) = hyperparam_list
        
    env = gym.make('LunarLander-v2', render_mode="rgb_array") #moved here to delete environment every so often

    record_dir = "recorded/trial_{}_{}".format(trial_no, start_time)
    if not os.path.isdir(record_dir):
        os.makedirs(record_dir, exist_ok=True)

    obs_size, = env.observation_space.shape
    act_size = 4

    measure_eps = set(filter(lambda epis: epis % 20 == 20 - 1, range(episodes)))
    with open(os.path.join(record_dir, "hyperparam.txt"), 'wb+') as hyperparam_f:
        pickle.dump(hyperparam_list, hyperparam_f)

    q_func = lunar_dqn.DeepQLunar(obs_size, act_size)
    optim = torch.optim.Adam(q_func.parameters(), lr=lr) # no longer using rmsprop unlike dqn paper
    q_targ = lunar_dqn.DeepQLunar(obs_size, act_size)
    q_targ.load_state_dict(q_func.state_dict())
    
    trainer = DeepQTrainer((q_func, optim), q_targ, 64, reward_decay, target_update_steps, steps_for_update=steps_for_update, buffer_len_to_start=buffer_len_to_start, buffer_sample_len=buffer_sample_len)
    trainer.set_epsilon_policy(partial(ep_policy, eps_decay=eps_decay, eps_min=eps_min))
    actor = deep_q_base.DeepQActor(q_func)

    obs_mean = (env.observation_space.high + env.observation_space.low) / 2
    obs_scale = 1.0 / (env.observation_space.high - obs_mean)

    measure_ep_thing = 3
    measure.measure_for_eps(measure_ep_thing, actor, env, (obs_mean, obs_scale), record_eps=set(range(measure_ep_thing)), record_dir=record_dir, prefix="initial")

    measure_seq = train_for_eps(episodes=episodes, 
        trainer_actor=(trainer, actor), 
        env=env, max_steps=max_steps, 
        measure_eps=measure_eps,
        measure_ep_reporter=ep_r, 
        )

    final_measure_ep = 20
    score_seq = measure.measure_for_eps(final_measure_ep, actor, env, (obs_mean, obs_scale), record_dir=record_dir+"/final", prefix="final", record_eps=set(range(5)))
    _, end_scores, bellman_seq, steps = measure.stats_score_seq(score_seq, trainer.reward_decay)
    measurement_r(end_scores, bellman_seq, steps, "final measurement w {} episodes, ".format(final_measure_ep))

    with open(os.path.join(record_dir, "measure_seq.txt"), 'wb+') as measure_seq_f:
        pickle.dump(measure_seq, measure_seq_f)

def encode_hypers(
    episodes = 45, #1200,
    target_update_steps = 20,
    buffer_len_to_start = 1000,
    max_steps = 600,
    steps_for_update = 1,
    reward_decay = 0.99,
    eps_min = 0.01,
    lr=0.0005,
    eps_decay = .995,
    buffer_sample_len=100000):

    hyperparam_list = (episodes, target_update_steps, 
        buffer_len_to_start, max_steps, 
        steps_for_update, reward_decay,
        eps_min, lr, eps_decay,
        buffer_sample_len
        )
    return hyperparam_list

def train_for_hypers(hypers):
    for t_no, h in enumerate(hypers):
        curr_time = datetime.now()
        train_once(t_no, curr_time, h)

# note: something is seizing during interrupted training on the cloud server, when rewards are deemed adequate
# it is during the final recording process that it happens