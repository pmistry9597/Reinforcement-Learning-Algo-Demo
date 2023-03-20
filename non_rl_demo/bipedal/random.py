import measure
from rl import actor

import matplotlib.pyplot as plt
import statistics as stats
import gymnasium as gym

env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
agent = actor.RandomActor(env.action_space)

episodes = 20
record_steps = set(filter(lambda i: i % 10 == 9, range(episodes)))

scores = measure.measure_for_eps(episodes, agent, env, max_steps=1000, record_eps=record_steps, obs_norm=(0, 1))

mean_scores = list(map(lambda x: stats.mean(x), scores))
overall_score_mean = stats.mean(mean_scores)
overall_score_std = stats.stdev(mean_scores)
steps = list(map(lambda x: len(x), scores))
mean_steps = stats.mean(steps)
std_steps = stats.stdev(steps)

print("scores mean, std: {}, {}:".format(overall_score_mean, overall_score_std))
print("steps mean, std: {}, {}:".format(mean_steps, std_steps))
plt.plot(mean_scores, 'b-')
plt.title("score means - fuck you")
plt.show()
plt.plot(steps, 'r-')
plt.title("steps")
plt.show()