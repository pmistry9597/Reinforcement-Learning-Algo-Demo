# ---- normal distribution shit that i didn't fucking need ----

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