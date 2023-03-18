import gymnasium as gym
from datetime import datetime
import torch

from rl.policy_grad.lunar.model import PolicyGradNNLunar
from rl.policy_grad import basic
from rl.policy_grad.trainer import PolicyGradTrainer, decayed_advantage
from rl.train_generic import complete_train, default_ep_r

def train_once(t_no, curr_time, hyperparams):
    (episodes, max_steps, 
        lr, reward_decay,
        trajecs_til_update, 
        cut_off_mean, 
        entropy_bonus,
        discard_non_termined,
        advantage_fn,
        ) = hyperparams

    env = gym.make('LunarLander-v2', render_mode="rgb_array")
    obs_size, = env.observation_space.shape
    act_size = 4

    policy = PolicyGradNNLunar(obs_size, act_size)
    optim = torch.optim.Adam(policy.parameters(), lr)

    obs_mean = (env.observation_space.high + env.observation_space.low) / 2
    obs_scale = 1.0 / (env.observation_space.high - obs_mean)
    obs_norm = (obs_mean, obs_scale)
    rew_norm = (0.0, 1.0/100.0)

    trainer = PolicyGradTrainer((policy, optim), reward_decay, trajecs_til_update, entropy_coef=entropy_bonus, discard_non_termined=discard_non_termined, advantage_fn=advantage_fn)
    actor = basic.PolicyGradActor(policy)

    measure_eps = set(filter(lambda epis: epis % 100 == 100 - 1, range(episodes)))
    save_eps = set(filter(lambda epis: epis % 100 == 100 - 1, range(episodes)))
    should_save_cond = lambda e: e in save_eps

    complete_train(
        (trainer, actor), 
        env, (episodes, max_steps, obs_norm, rew_norm),
        (default_ep_r, measure_eps), 
        should_save_cond, 
        "policy_grad", str(datetime.now()), cut_off_mean)

def encode_hypers(episodes, max_steps, 
    lr, reward_decay, 
    trajecs_til_update, cut_off_mean=None, entropy_bonus=0.0, discard_non_termined=False, advantage_fn=decayed_advantage):

    return (episodes, max_steps, 
        lr, reward_decay, 
        trajecs_til_update, 
        cut_off_mean,
        entropy_bonus,
        discard_non_termined,
        advantage_fn,
        )