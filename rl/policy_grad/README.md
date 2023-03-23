# Policy Gradients

WIP! This method uses a model to output probabilities of taking an action a, given a state s. The training is done via a policy gradient, which is quite a simple formula that takes a trajectory of states, actions, and rewards as input. It needs a full (terminated) trajectory, which means this algorithm cannot be applied to never-ending (online) environments.

## Policy Gradient Description
The gradient formula takes in the states, actions, and rewards as a sequence over a trajectory. The states and actions are used to compute the policy's likelihood of taking the action that was actually taken during each time step (essentially plug in state s and find probability it outputs for actual action taken a).

The rewards are used to compute a view of how well the agent did. This can simply be the sum of total rewards, the discounted reward through the Bellman equation, or more complicated things such as the advantage estimation.

The two quantities are used to compute the policy gradient. In many cases, however, the loss function form is used and then the gradient is computed using the typical automatic differentiation scheme that comes with most deep-learning libraries.

## Lunar Lander Results
I was a bit thrown off by my initial policy gradient trials, as they showed no sign of convergence within several hundred episodes as with Deep-Q Networks. However, it turns out it is well known that policy gradient methods are far less sample efficient. This means that policy gradients only live up to their reputation as fast convergers only if many samples can be generated from an environment quickly. It is also known that policy gradients often converge to local minima. Often times, instead of achieving an end score of 100 with a cumulative reward of 200, it would converge to getting close to the landing pad but fail to land without a crash.

## Beyond Policy Gradients

Policy gradient methods are the most commonly used class of algorithms to train an effective policy, atleast as of writing this. There is a catch, however - the advantage function used to judge the long term rewards for the loss calculation is a method that can be considered to be based off of value-based policy methods. This leads to the common actor-critic scenario that we see today.