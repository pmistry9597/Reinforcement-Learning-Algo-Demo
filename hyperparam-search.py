import rl.deep_q.lunar.train as q_train
import rl.policy_grad.lunar.train as policy_train_lunar
import rl.policy_grad.cartpole.train as policy_grad_cartpole
from rl.train_generic import train_for_hypers
from rl.policy_grad import advantage_fns
import numpy as np
import math
from functools import partial
import sys

# make sure there is a directory called "recorded" in this repo

def get_q_hyperlist():
    hypers = []

    hypers = [q_train.encode_hypers(episodes=1000), q_train.encode_hypers(steps_for_update=4), 
            q_train.encode_hypers(lr=0.001), q_train.encode_hypers(steps_for_update=4, lr=0.001), 
    ]

    return hypers

def get_policy_grad_lunar_hyperlist():
    hypers = [
        policy_train_lunar.encode_hypers(episodes=100000, max_steps=10000, lr=0.002, reward_decay=1.0, entropy_bonus=0.0, trajecs_til_update=16, discard_non_termined=False, advantage_fn=advantage_fns.reward_to_go_normalized)
    ]

    return hypers

def get_policy_grad_cartpole_hyperlist():
    hypers = [
        policy_grad_cartpole.encode_hypers(episodes=30000, max_steps=10000, lr=0.002, reward_decay=0.999, entropy_bonus=0.0, trajecs_til_update=16, discard_non_termined=False, advantage_fn=advantage_fns.reward_to_go)
    ]

    return hypers

def handle_hyper_in(hypers, train_once):
    hyper_i = int(sys.argv[1])
    if len(hypers) <= hyper_i:
        sys.exit(1)
    train_for_hypers(hypers, train_once)

if __name__ == '__main__':
    # handle_hyper_in(get_policy_grad_cartpole_hyperlist(), policy_grad_cartpole.train_once)
    handle_hyper_in(get_policy_grad_lunar_hyperlist(), policy_train_lunar.train_once)
    # handle_hyper_in(get_q_hyperlist(), q_train.train_once)
