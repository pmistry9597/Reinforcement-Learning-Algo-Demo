# Deep Q-Networks (DQN)

![DQN LunarLander-v2 Success](/images/dqn_lunarlanderv2.gif)

The GIF above is a demo of the DQN algorithm I wrote based on [Deepmind's 2015 paper](https://www.nature.com/articles/nature14236), and some other internet articles to guide me in hyperparameter tuning :P. The game environment shown is pre-built in Gymnasium, which was created by OpenAI and maintained currently by Farama.

Deep Q-Networks involves a neural network predicting the value of a Q function. A Q function is an estimate of the long term reward, usually denoted as Q(s,a). The only inputs are the current state and an action value. It essentially predicts what happens if you were to take an action, a, in a state, s. An optimal Q function, Q*(s,a), predicts what will *actually* happen in an environment, assuming the best actions are always taken (this last part has to do with a *policy*).

This algorithm works in a discrete action space only, which is perfect for the LunarLander environment. This means, however, it won't be applied to other environments. It is based on [Deepmind's 2015 paper](https://www.nature.com/articles/nature14236).