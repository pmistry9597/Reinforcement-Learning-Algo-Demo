import gymnasium as gym
import measure
import numpy as np
import math
import statistics as stats
from datetime import datetime
from functools import partial

############# Screwing around with training an agent in lunar lander environment, not meant to be a final product
# the file of cursed long functions and some redundant code - I wanted to get something working!

def measure_at_train_episode(trainer_actor, env, obs_norm, max_steps, measurements, epis_no, measure_ep_reporter):
    obs_mean, obs_scale = obs_norm
    trainer, actor = trainer_actor

    print("steps:", len(trainer.steps))
    rew_seq = measure.measure_for_eps(13, actor, env, (obs_mean, obs_scale), max_steps=max_steps)
    _, end_scores, bellman_seq, steps = measure.stats_score_seq(rew_seq, trainer.reward_decay)

    measurements.append((epis_no, end_scores, bellman_seq, steps))
    measure_ep_reporter(epis_no, end_scores, bellman_seq, steps)

    # stopping when recorded average high nuff for me
    end_mean = stats.mean(end_scores)

    return end_mean

# reporters are meant to accept any input if needed
def train_for_eps(episodes, trainer_actor, env, max_steps=math.inf, measure_eps=set(), step_reporter=lambda step, epis, reward: None, measure_ep_reporter=lambda epis, seq, bellman_seq, steps: None):
    trainer, actor = trainer_actor
    measurements = []

    # normalizing params for obs, action, reward
    obs_mean = (env.observation_space.high + env.observation_space.low) / 2
    obs_scale = 1.0 / (env.observation_space.high - obs_mean)

    # measurements pls? rewards over time, cumulative reward per episode, steps required
    for e in range(episodes):
        termin = False
        step = 0
        obs, info = env.reset()
        obs = deep_q_trial.normalize(obs, obs_mean, obs_scale)

        while not termin and max_steps > step:
            prev_obs = obs
            act_tensor = trainer.act(torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.double), e)
            obs, reward, termin, trunc, info = env.step(act_tensor.item())

            obs = deep_q_trial.normalize(obs, obs_mean, obs_scale)
            sample = (torch.tensor(prev_obs, dtype=torch.double), torch.tensor(obs, dtype=torch.double), termin, act_tensor.squeeze(), torch.tensor(reward, dtype=torch.double))
            trainer.new_step(sample)

            step_reporter(step, e, reward)
            step += 1

        if e in measure_eps:
            # trainer_actor, env, (obs_mean, obs_scale), max_steps, measurements
            end_mean = measure_at_train_episode(trainer_actor, env, (obs_mean, obs_scale), max_steps, measurements, e, measure_ep_reporter)
            if end_mean >= 95:
                break

    return measurements

def ep_r(epis, end_scores, bellman_seq, steps):
    measurement_r(end_scores, bellman_seq, steps, "measured episode {}, ".format(epis))

def measurement_r(end_scores, bellman_seq, steps, msg):
    end_score_mean, std = measure.get_stats_on_seq(end_scores)
    step_mean, step_std = measure.get_stats_on_seq(steps)
    bell_mean, bell_std = measure.get_stats_on_seq(bellman_seq)
    print("{}end score mean: {}, std: {};  bellman mean: {}, std: {}; steps mean: {}, std {}".format(msg, end_score_mean, std, bell_mean, bell_std, step_mean, step_std))

import deep_q_trial
import torch

def ep_policy(epis_no, eps_decay, eps_min):
    return max((eps_decay ** epis_no), eps_min)

import pickle
import os

def train_once(trial_no, start_time, hyperparam_list):
    (episodes, target_update_steps, 
        buffer_len_to_start, max_steps, 
        steps_for_update, reward_decay,
        eps_min, lr, eps_decay
        ) = hyperparam_list
        
    env = gym.make('LunarLander-v2', render_mode="rgb_array") #moved here to delete environment every so often

    record_dir="recorded/trial_{}_{}".format(trial_no, start_time)
    if not os.path.isdir(record_dir):
        os.makedirs(record_dir, exist_ok=True)

    obs_size, = env.observation_space.shape
    act_size = 4

    measure_eps = set(filter(lambda epis: epis % 20 == 20 - 1, range(episodes)))
    with open(os.path.join(record_dir, "hyperparam.txt"), 'wb+') as hyperparam_f:
        pickle.dump(hyperparam_list, hyperparam_f)

    q_func = deep_q_trial.DeepQLunar(obs_size, act_size)
    optim = torch.optim.Adam(q_func.parameters(), lr=lr) # no longer using rmsprop unlike dqn paper
    q_targ = deep_q_trial.DeepQLunar(obs_size, act_size)
    q_targ.load_state_dict(q_func.state_dict())
    
    trainer = deep_q_trial.DeepQLunarTrainer((q_func, optim), q_targ, 64, reward_decay, target_update_steps, steps_for_update=steps_for_update, buffer_len_to_start=buffer_len_to_start)
    trainer.set_epsilon_policy(partial(ep_policy, eps_decay=eps_decay, eps_min=eps_min))
    actor = deep_q_trial.DeepQActor(q_func)

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
    episodes = 1200,
    target_update_steps = 20,
    buffer_len_to_start = 1000,
    max_steps = 600,
    steps_for_update = 1,
    reward_decay = 0.99,
    eps_min = 0.01,
    lr=0.0005,
    eps_decay = .995):

    hyperparam_list = (episodes, target_update_steps, 
        buffer_len_to_start, max_steps, 
        steps_for_update, reward_decay,
        eps_min, lr, eps_decay
        )
    return hyperparam_list

def train_for_hypers(hypers):
    for t_no, h in enumerate(hypers):
        curr_time = datetime.now()
        train_once(t_no, curr_time, h)

# note: something is seizing during interrupted training on the cloud server, when rewards are deemed adequate
# it is during the final recording process that it happens