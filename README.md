# Reinforcement Learning Algorithms Demonstration

This repository is meant to be a demo of some reinforcement learning (RL) algorithms I find to be the most prevalent at the time of writing this README. Of course, it is also meant to get me into more advanced RL to begin with. Some of the descriptions in this README are a little basic, as this not a research paper :P.

![DQN LunarLander-v2 Success](/images/dqn_lunarlanderv2.gif)

The GIF above is a demo of the DQN algorithm I wrote based on [Deepmind's 2015 paper](https://www.nature.com/articles/nature14236), and some other internet articles to guide me in hyperparameter tuning :P. The game environment shown is pre-built in Gymnasium, which was created by OpenAI and maintained currently by Farama.

## TF is reinforcment learning?

RL is a subset of machine learning that focuses toward creating algorithms that can solve problems that involve a sequence of decisions (over time as far as I can tell). The quality of an algorithm is quantized via rewards, with the usual goal being to maximize the long-term reward. It often is meant to be robust to changes in the environment, as most RL problems are framed as environments probabilistic transitions into new states, and the agent makes a decision based on the state (or sometimes previous states) it can see. This branch of ML was behind breakthroughs such as the first ML victory against a top Go player in [2016](https://www.nature.com/articles/nature.2016.19575).

The most common theoretical framework for RL problems is the Markov Decision Process.

## What is in this repo?

I intend on demoing the following algorithms in simulations that are pre-built in Gymnasium. The example shown above is the LunarLander-v2 environment, which is one of the simpler ones. The algorithms are written from scratch, with PyTorch being the basis for the neural networks.

- [x] Deep Q-Networks (DQN)
- [ ] Policy Gradients - with basic baseline
- [ ] Proximal Policy Optimization (PPO)
- [ ] Soft Actor Critic (SAC)

The first two algorithms are much simpler than the last two, and were early and foundational algorithms in the field of modern RL. The last two are sort of the workhorses of today's RL as far as I can tell.

Descriptions of algorithms currently completed can be found in their respective folders under the rl folder.