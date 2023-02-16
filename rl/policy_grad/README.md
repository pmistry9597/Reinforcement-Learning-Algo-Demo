# Policy Gradients (w/ basic baseline)

WIP! This method uses a model to output probabilities of taking an action a, given a state s. The training is done via a policy gradient, which is quite a simple formula that takes a trajectory of states, actions, and rewards as input. It needs a full trajectory, which means this algorithm cannot be applied to never-ending (online) environments.

## Policy Gradient Description
The gradient formula takes in the states, actions, and rewards as a sequence over a trajectory. The states and actions are used to compute the policy's likelihood of taking the action that was actually taken during each time step (essentially plug in state s and find probability it outputs for actual action taken a).

The rewards are used to compute a view of how well the agent did. This can simply be the sum of total rewards, the discounted reward through the Bellman equation, or more complicated things such as the advantage estimation. In this algorithm's case, a simple baseline will be used and then subtracted from the Bellman reward. This is known as policy gradients with a baseline. The baseline here will most likely be a running average of rewards over previous time steps.

The two quantities are used to compute the policy gradient. In many cases, however, the loss function form is used and then the gradient is computed using the typical automatic differentiation scheme that comes with most deep-learning libraries.

## Beyond Policy Gradients

The policy gradient computation here will be applied to Proximal Policy Optimization (PPO) later on, where the baseline is replaced with an advantage estimation. Many algorithms in RL today make use of this advantage estimation scheme, where a different model computes something that directly or indirectly allows an advantage to be computed. The advantage scheme is usually similar to how DQN computes the value of an action, and in this case these algorithms are known as actor-critic methods.