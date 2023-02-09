import pickle
import matplotlib.pyplot as plt
import statistics as stat
from functools import partial
import os

# handy little info displays/plots for items in recorded dir

def get_meaned_dat(item, no):
    return stat.mean(item[no])

def get_dat_path(path):
    with open(path, 'rb') as f:
        dat = pickle.load(f)
        return dat

def get_dat(folder, name):
    return get_dat_path('recorded/{}/{}'.format(folder, name))

def get_hypers(folder):
    return get_dat(folder, 'hyperparam.txt')

def get_seq(folder, no):
    dat = get_dat(folder, 'measure_seq.txt')
    spec = map(partial(get_meaned_dat, no=no), dat)
    spec = list(spec)
    return spec

def get_bells(folder):
    return get_seq(folder, 2)

def get_means(folder):
    return get_seq(folder, 1)

def plot_numbered(folder, no):
    spec = get_seq(folder, no)
    plt.plot(spec)
    plt.show()

def plot_bells(folder):
    plt.plot(get_bells(folder))
    plt.show()

def plot_means(folder):
    plt.plot(get_means(folder))
    plt.show()

def plot_all_bells_hypers(no):
    dir_list = list(filter(lambda x: len(x) > 1, map(lambda x: x[1], os.walk('recorded'))))[0]
    bells_it = map(get_bells, dir_list)
    hypers_it = map(lambda f: get_hypers(f)[no], dir_list)
    bells = []
    hypers = []
    while True:
        try:
            b = next(bells_it, None)
            h = next(hypers_it, None)
            if b is None or h is None:
                break
            b = b[-1]
        except FileNotFoundError:
            pass
        bells.append(b)
        hypers.append(h)
    # print(hypers, bells)
    plt.hist(hypers)
    plt.show()
    plt.plot(hypers, bells, 'go')
    plt.show()