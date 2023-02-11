# good for normalizing inputs into a neural net
def normalize(val, mean, scale):
    val = val - mean
    return val * scale