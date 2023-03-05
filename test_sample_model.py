from measure import measure_for_eps
from rl.policy_grad.basic import PolicyGradActor
import torch
import gymnasium as gym

pol = torch.load('sample_model')
actor = PolicyGradActor(pol)
env = gym.make('LunarLander-v2', render_mode="rgb_array")

obs_mean = (env.observation_space.high + env.observation_space.low) / 2
obs_scale = 1.0 / (env.observation_space.high - obs_mean)
obs_norm = (obs_mean, obs_scale)

ep_count = 5
measure_for_eps(ep_count, actor, env, obs_norm, max_steps=600, record_eps=set(range(ep_count)))