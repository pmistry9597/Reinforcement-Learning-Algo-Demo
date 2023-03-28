# Policy Gradients

This method uses a model to output probabilities of taking an action a, given a state s. The training is done via a policy gradient, which is quite a simple formula that takes a trajectory of states, actions, and rewards as input. It needs a full (terminated) trajectory, which means this algorithm cannot be applied to never-ending (online) environments.

## Policy Gradient Description
The gradient formula takes in the states, actions, and rewards as a sequence over a trajectory. The states and actions are used to compute the policy's likelihood of taking the action that was actually taken during each time step (essentially plug in state s and find probability it outputs for actual action taken a).

The rewards are used to compute a view of how well the agent did. This can simply be the sum of total rewards, the discounted reward through the Bellman equation, or more complicated things such as the advantage estimation with a value function.

The two quantities are used to compute the policy gradient. In many cases, however, the loss function form is used and then the gradient is computed using the typical automatic differentiation scheme that comes with most deep-learning libraries.

## Lunar Lander Results
I was a bit thrown off by my initial policy gradient trials, as they showed no sign of convergence within several hundred episodes as with Deep-Q Networks. However, it turns out that it is well known for policy gradient methods to be far less sample efficient. This means that policy gradients only live up to their reputation as fast convergers only if many samples can be generated from an environment quickly. It is also known that policy gradients often converge to local minima. Often times, instead of achieving an end score of 100 with a cumulative reward of 200, it would converge to getting close to the landing pad but fail to land without a crash.
This turned out to be a lot less predictable than Deep-Q Networks. Overall this took around 3000 iterations on average to converge to near optimal solutions. Each iteration, however, is a batch which, here, was between 16-64 episodes. While it is possible that there is an error in this code, many others online seem to have similar results (there are those who claim to get convergence faster than this, however).

## Beyond Policy Gradients

Policy gradient methods are the most commonly used class of algorithms to train an effective policy, atleast as of writing this. There is a catch, however - the advantage function used to judge the long term rewards for the loss calculation is a method that can be considered to be based off of value-based policy methods. This leads to the common actor-critic scenario that we see today.

Another variation is that the loss function from the original policy gradients algorithm (REINFORCE) uses log-based loss. Many modern algorithms, on the other hand, use different versions, such as in PPO where probability ratios are used with a clipping function.