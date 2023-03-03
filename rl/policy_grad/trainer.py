from .. import trainer
from functools import partial, reduce
from rl.policy_grad.basic import sample_act
import torch
import torch.nn.functional as nn_func

class PolicyGradTrainer(trainer.Trainer):
    def act(self, obs, trajec_no):
        policy, _ = self.policy_optim
        logits_out = policy(obs)
        self.logits_outs[-1].append(logits_out)
        # print(pol_out)
        return sample_act(logits_out)

    def new_step(self, sampl):
        self.trajecs[-1].append(sampl)

    def punctuate_trajectory(self):
        # trigger training update here when right
        if len(self.trajecs) % self.TRAJECS_TIL_UPDATE == self.TRAJECS_TIL_UPDATE - 1:
            self.update_policy()
            self.logits_outs = []
            self.trajecs = []
        self.new_buffers()

    # --- specific to class section ---

    def __init__(self, policy_optim, reward_decay, trajecs_til_update):
        # self.actor = actor
        self.policy_optim = policy_optim
        self.reward_decay = reward_decay
        self.TRAJECS_TIL_UPDATE = trajecs_til_update

        self.trajecs = []
        self.logits_outs = []

    def new_buffers(self):
        self.logits_outs.append([])
        self.trajecs.append([])

    def update_policy(self):
        smpl_rew_i = 4
        trajecs_rewards = map(lambda trajec: tuple(map(lambda smpl: smpl[smpl_rew_i], trajec)), self.trajecs)
        smpl_act_i = 3
        acts_taken = tuple(map(lambda trajec: torch.tensor(tuple(map(lambda smpl: smpl[smpl_act_i], trajec))), self.trajecs))
        logits_outs = tuple(map(torch.stack, self.logits_outs))
        loss = trajecs_loss(trajecs_rewards, self.reward_decay, decayed_advantage, logits_outs, acts_taken)

        _, optim = self.policy_optim
        optim.zero_grad()
        loss.backward()
        optim.step()

        return loss

    # note: test this garbage class

# calculate loss for policy gradient method
# take in trajectories to compute over, advantage/reward fn of trajectory, policy probability outputs recorded, actions actually taken
def trajecs_loss(trajecs_rewards, reward_decay, advantage_fn, logits_outs, acts_taken):
    advantages_map = map(partial(trajec_advantage, reward_decay=reward_decay, advantage_fn=advantage_fn), trajecs_rewards)
    advantages_tens = tuple(map(torch.tensor, advantages_map))

    log_act_probs = map(get_log_act_prob, logits_outs)
    log_act_sels = map(get_log_act_sel, zip(log_act_probs, acts_taken))
    # log_act_sel = log_act_prob.gather(2, acts_taken.unsqueeze(2)) # 2 is dim to select individual probabilities

    losses = tuple(map(mul_both, zip(advantages_tens, log_act_sels)))
    total = sum(map(len, losses))
    accum_losses = tuple(map(torch.sum, losses))
    total_loss = reduce(lambda x, y: x + y, accum_losses, torch.tensor(0.0, dtype=torch.double))
    return total_loss / total

    # note: may need to update as may accept tensor instead of list of trajectory features

def mul_both(log_act):
    adv, acts = log_act
    return adv * acts.squeeze()

def get_log_act_sel(log_act_w_acts_taken):
    log_act_prob, acts_taken_sing = log_act_w_acts_taken
    return log_act_prob.gather(1, acts_taken_sing.unsqueeze(1))

def get_log_act_prob(logits_out):
    return nn_func.log_softmax(logits_out.squeeze(1), dim=1)

def trajec_advantage(trajec_rewards, reward_decay, advantage_fn):
    trajec_rewards = torch.stack(trajec_rewards)
    return tuple(map(lambda i: advantage_fn(trajec_rewards[i:], reward_decay), range(len(trajec_rewards))))

def decayed_advantage(rew_traj, decay):
    return torch.sum( decay ** torch.arange(len(rew_traj)) * rew_traj )