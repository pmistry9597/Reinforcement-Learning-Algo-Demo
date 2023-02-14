# ------ train given an environment and a trainer object to work with -----
import math
import torch
import numpy as np
import statistics as stats
import measure

from rl.helpers import normalize

# reporters are meant to accept any input if needed
def train_for_eps(episodes, trainer_actor, env, max_steps=math.inf, cut_off_mean=math.inf, measure_eps=set(), step_reporter=lambda step, epis, reward: None, measure_ep_reporter=lambda epis, seq, bellman_seq, steps: None):
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
        obs = normalize(obs, obs_mean, obs_scale)

        while not termin and max_steps > step:
            prev_obs = obs
            act_tensor = trainer.act(torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.double), e)
            obs, reward, termin, trunc, info = env.step(act_tensor.item())

            obs = normalize(obs, obs_mean, obs_scale)
            sample = (torch.tensor(prev_obs, dtype=torch.double), torch.tensor(obs, dtype=torch.double), termin, act_tensor.squeeze(), torch.tensor(reward, dtype=torch.double))
            trainer.new_step(sample)

            step_reporter(step, e, reward)
            step += 1

        if e in measure_eps:
            # trainer_actor, env, (obs_mean, obs_scale), max_steps, measurements
            end_mean = measure_at_train_episode(trainer_actor, env, (obs_mean, obs_scale), max_steps, measurements, e, measure_ep_reporter)
            if end_mean >= cut_off_mean:
                break

    return measurements

def measure_at_train_episode(trainer_actor, env, obs_norm, max_steps, measurements, epis_no, measure_ep_reporter):
    obs_mean, obs_scale = obs_norm
    trainer, actor = trainer_actor

    print("steps:", len(trainer.steps))
    rew_seq = measure.measure_for_eps(13, actor, env, (obs_mean, obs_scale), max_steps=max_steps)
    _, end_scores, bellman_seq, steps = measure.stats_score_seq(rew_seq, trainer.reward_decay)

    measurements.append((epis_no, end_scores, bellman_seq, steps))
    measure_ep_reporter(epis_no, end_scores, bellman_seq, steps)

    end_mean = stats.mean(end_scores)

    return end_mean

def ep_r(epis, end_scores, bellman_seq, steps):
    measurement_r(end_scores, bellman_seq, steps, "measured episode {}, ".format(epis))

def measurement_r(end_scores, bellman_seq, steps, msg):
    end_score_mean, std = measure.get_stats_on_seq(end_scores)
    step_mean, step_std = measure.get_stats_on_seq(steps)
    bell_mean, bell_std = measure.get_stats_on_seq(bellman_seq)
    print("{}end score mean: {}, std: {};  bellman mean: {}, std: {}; steps mean: {}, std {}".format(msg, end_score_mean, std, bell_mean, bell_std, step_mean, step_std))