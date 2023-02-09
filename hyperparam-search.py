from lunar_train import train_for_hypers, encode_hypers
import numpy as np
import math
from functools import partial

import sys

def get_curr_hyperlist():
    hypers = []

    hypers = [encode_hypers(), encode_hypers(steps_for_update=4), 
            encode_hypers(lr=0.001), encode_hypers(steps_for_update=4, lr=0.001), 
    ]

    return hypers

if __name__ == '__main__':
    hyper_i = int(sys.argv[1])
    hypers = get_curr_hyperlist()
    if len(hypers) <= hyper_i:
        sys.exit(1)
    train_for_hypers([hypers[hyper_i]])


# ---- section of normal distribution shit that i didn't fucking need ----

def in_bounds(sampl, seq, low, high):
    return sampl >= low and sampl <= high

def far_nuff(sampl, seq, dist):
    for s in seq:
        if math.fabs(s - sampl) < dist:
            return False
    return True

def eval_pass_fns(sampl, seq, pass_fns):
    for fn in pass_fns:
        if not fn(sampl, seq):
            return False
    return True

def add_new_smpls(smpl_count, distr, pass_fns, seq):
    curr_count = 0
    while curr_count < smpl_count:
        sampl = distr()
        if eval_pass_fns(sampl, seq, pass_fns):
            seq.append(sampl)
            curr_count += 1
    return seq

def norm_scal_distr(mean, std):
    return partial(np.random.default_rng().normal, loc=mean, scale=std)
