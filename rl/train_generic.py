# ------ train given an environment and a trainer object to work with -----
import math
import torch
import numpy as np
import statistics as stats
import measure
import os
import pickle
from datetime import datetime

from rl.helpers import normalize

# train with saving, measurements, etc
def complete_train(trainer_actor, env, basic_hypers, measure_ep_param, should_save_cond, class_code, save_code, cut_off_mean=math.inf):
    trainer, actor = trainer_actor
    episodes, max_steps, obs_norm, rew_norm = basic_hypers
    ep_r, measure_eps = measure_ep_param # fn to call when measurement occurs, and set of which episodes to run it on
    if cut_off_mean is None:
        cut_off_mean = math.inf

    default_save_root = "recorded"
    path = os.path.join(default_save_root, class_code, save_code)
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "trainer_state"), 'wb+') as trainer_f:
        pickle.dump(trainer, trainer_f)

    initial_measure_ep = 3
    measure.measure_for_eps(initial_measure_ep, actor, env, obs_norm, record_eps=set(range(initial_measure_ep)), record_dir=path, prefix="initial")

    measure_seq = train_for_eps(episodes=episodes, 
        trainer_actor=trainer_actor, 
        env=env, max_steps=max_steps, 
        path=path, should_save_cond=should_save_cond,
        measure_eps=measure_eps,
        measure_ep_reporter=ep_r, 
        obs_norm=obs_norm,
        rew_norm=rew_norm,
        cut_off_mean=cut_off_mean
        )

    final_measure_ep = 20
    score_seq = measure.measure_for_eps(final_measure_ep, actor, env, obs_norm, record_dir=path+"/final", prefix="final", record_eps=set(range(5)))
    _, end_scores, bellman_seq, steps = measure.stats_score_seq(score_seq, trainer.reward_decay)
    measurement_r(end_scores, bellman_seq, steps, "final measurement w {} episodes, ".format(final_measure_ep))

    with open(os.path.join(path, "measure_seq"), 'wb+') as measure_seq_f:
        pickle.dump(measure_seq, measure_seq_f)
    with open(os.path.join(path, "final_seq"), 'wb+') as final_seq_f:
        pickle.dump((end_scores, bellman_seq, steps), final_seq_f)

# reporters are meant to accept any input if needed
def train_for_eps(episodes, trainer_actor, env, path, should_save_cond, obs_norm=(0.0,1.0), rew_norm=(0.0,1.0), max_steps=math.inf, cut_off_mean=math.inf, measure_eps=set(), step_reporter=lambda step, epis, reward: None, measure_ep_reporter=lambda epis, seq, bellman_seq, steps: None):
    trainer, actor = trainer_actor
    measurements = []
    default_model_dir = "model"
    path = os.path.join(path, default_model_dir) #work from here

    # normalizing params for obs, action, reward
    obs_mean, obs_scale = obs_norm
    rew_mean, rew_scale = rew_norm

    last_max = -math.inf # max score of a model
    
    trainer.punctuate_trajectory()
    for e in range(episodes):
        termin = False
        step = 0
        obs, info = env.reset()
        obs = normalize(obs, obs_mean, obs_scale)

        while not termin and max_steps > step:
            prev_obs = obs
            act_tensor = trainer.act(torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.double), e)
            obs, reward, termin, trunc, info = env.step(act_tensor.item())

            obs = normalize(obs, obs_mean, obs_scale)
            reward = normalize(reward, rew_mean, rew_scale)
            sample = (torch.tensor(prev_obs, dtype=torch.double), torch.tensor(obs, dtype=torch.double), termin, act_tensor.squeeze(), torch.tensor(reward, dtype=torch.double))
            trainer.new_step(sample)

            step_reporter(step, e, reward)
            step += 1
        trainer.punctuate_trajectory()

        if e in measure_eps:
            # trainer_actor, env, (obs_mean, obs_scale), max_steps, measurements
            end_mean = measure_at_train_episode(trainer_actor, env, (obs_mean, obs_scale), max_steps, measurements, e, measure_ep_reporter)
            if should_save_cond(e):
                save_model_score(path, str(e), actor, end_mean)
            if end_mean > last_max:
                last_max = end_mean
                save_model_score(path, "max_ep_{}".format(e), actor, end_mean)
            if end_mean >= cut_off_mean:
                break
    
    save_model_score(path, "final", actor, end_mean)
    return measurements

def save_model_score(path, name, actor, score):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, '{}_score'.format(name, score)), 'wb+') as score_f:
        pickle.dump(score, score_f)
    actor.save_model(os.path.join(path, name))

def measure_at_train_episode(trainer_actor, env, obs_norm, max_steps, measurements, epis_no, measure_ep_reporter):
    obs_mean, obs_scale = obs_norm
    trainer, actor = trainer_actor

    print("steps:", trainer.get_total_steps())
    rew_seq = measure.measure_for_eps(13, actor, env, (obs_mean, obs_scale), max_steps=max_steps)
    _, end_scores, bellman_seq, steps = measure.stats_score_seq(rew_seq, trainer.reward_decay)

    measurements.append((epis_no, end_scores, bellman_seq, steps))
    measure_ep_reporter(epis_no, end_scores, bellman_seq, steps)

    end_mean = stats.mean(end_scores)

    return end_mean

def default_ep_r(epis, end_scores, bellman_seq, steps):
    measurement_r(end_scores, bellman_seq, steps, "measured episode {}, ".format(epis))

def measurement_r(end_scores, bellman_seq, steps, msg):
    end_score_mean, std = measure.get_stats_on_seq(end_scores)
    step_mean, step_std = measure.get_stats_on_seq(steps)
    bell_mean, bell_std = measure.get_stats_on_seq(bellman_seq)
    print("{}end score mean: {}, std: {};  bellman mean: {}, std: {}; steps mean: {}, std {}".format(msg, end_score_mean, std, bell_mean, bell_std, step_mean, step_std))

def train_for_hypers(hypers, train_once):
    for t_no, h in enumerate(hypers):
        curr_time = datetime.now()
        train_once(t_no, curr_time, h)

# note: something is seizing during interrupted training on the cloud server, when rewards are deemed adequate
# it is during the final recording process that it happens