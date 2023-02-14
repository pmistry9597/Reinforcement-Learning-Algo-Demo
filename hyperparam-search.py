from rl.deep_q.lunar.train import train_for_hypers, encode_hypers
import numpy as np
import math
from functools import partial
import sys

# make sure there is a directory called "recorded" in this repo

def get_curr_hyperlist():
    hypers = []

    hypers = [encode_hypers(episodes=50), encode_hypers(steps_for_update=4), 
            encode_hypers(lr=0.001), encode_hypers(steps_for_update=4, lr=0.001), 
    ]

    return hypers

if __name__ == '__main__':
    hyper_i = int(sys.argv[1])
    hypers = get_curr_hyperlist()
    if len(hypers) <= hyper_i:
        sys.exit(1)
    train_for_hypers([hypers[hyper_i]])
