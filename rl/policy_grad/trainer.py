from .. import trainer
from functools import reduce
from rl.policy_grad.basic import sample_act
from rl.policy_grad import advantage_fns
import torch
import torch.nn.functional as nn_func

# calculate loss for policy gradient method
# take in trajectories to compute over, advantage/reward fn of trajectory, policy probability outputs recorded, actions actually taken
def trajecs_loss(trajecs_rewards, reward_decay, advantage_fn, logits_outs, acts_taken):
    advantages_tens = advantage_fn(trajecs_rewards, reward_decay)

    log_act_probs = map(get_log_act_prob, logits_outs)
    log_act_sels = tuple(map(get_log_act_sel, zip(log_act_probs, acts_taken)))

    losses = tuple(map(mul_both, zip(advantages_tens, log_act_sels)))
    accum_losses = tuple(map(torch.sum, losses))
    overall_loss = torch.mean(torch.stack(accum_losses))
    return overall_loss

class PolicyGradTrainer(trainer.Trainer):
    def act(self, obs, trajec_no):
        policy, _ = self.policy_optim
        logits_out = policy(obs)
        self.logits_outs[-1].append(logits_out)
        action = sample_act(logits_out)
        return action

    def new_step(self, sampl):
        self.trajecs[-1].append(sampl)

    def punctuate_trajectory(self):
        if self.DISCARD_NON_TERMIN and len(self.trajecs) > 0:
            _,_,trajec_termin,_,_ = self.trajecs[-1][-1]
            if not trajec_termin:
                del self.trajecs[-1]
                del self.logits_outs[-1]
        # trigger training update here when right
        if len(self.trajecs) >= self.TRAJECS_TIL_UPDATE:
            curr_chunk_steps = sum(map(len, self.trajecs))
            self.total_steps += curr_chunk_steps
            self.update_policy()
            self.logits_outs = []
            self.trajecs = []
        self.new_buffers()

    def get_total_steps(self):
        return self.total_steps

    # --- specific to class section ---

    def __init__(self, policy_optim, reward_decay, trajecs_til_update, entropy_coef, discard_non_termined, advantage_fn=advantage_fns.reward_to_go):
        # self.actor = actor
        self.policy_optim = policy_optim
        self.advantage_fn = advantage_fn
        self.reward_decay = reward_decay
        self.TRAJECS_TIL_UPDATE = trajecs_til_update
        self.ENTROPY_COEF = entropy_coef
        self.DISCARD_NON_TERMIN = discard_non_termined

        self.trajecs = []
        self.logits_outs = []
        self.total_steps = 0

    def new_buffers(self):
        self.logits_outs.append([])
        self.trajecs.append([])

    def compute_loss(self):
        smpl_rew_i = 4
        trajecs_rewards = tuple(map(lambda trajec: tuple(map(lambda smpl: smpl[smpl_rew_i], trajec)), self.trajecs))
        smpl_act_i = 3
        acts_taken = tuple(map(lambda trajec: torch.tensor(tuple(map(lambda smpl: smpl[smpl_act_i], trajec))), self.trajecs))
        logits_outs = tuple(map(torch.stack, self.logits_outs))
        logits_outs = tuple(map(lambda t: t.squeeze(1), logits_outs))
        entropy_total = torch.mean(torch.cat(tuple(map(lambda t: t.view([-1]), map(entropy, logits_outs))), dim=0))
        # print(trajecs_rewards[:2])
        # print(tuple(map(lambda t: t.shape, acts_taken[:2])))
        # print(tuple(map(lambda t: t.shape, logits_outs[:2])))
        # print(acts_taken[:2])
        # print(logits_outs[:2])
        return -trajecs_loss(trajecs_rewards, self.reward_decay, self.advantage_fn, logits_outs, acts_taken) + -self.ENTROPY_COEF * entropy_total

    def update_policy(self):
        loss = self.compute_loss()

        _, optim = self.policy_optim
        optim.zero_grad()
        loss.backward()
        optim.step()

        return loss

def entropy(logits):
    logits = logits.squeeze()
    return -torch.sum(nn_func.softmax(logits, dim=-1) * nn_func.log_softmax(logits, dim=-1), dim=-1)

def mul_both(log_act):
    adv, acts = log_act
    prod = adv * acts.squeeze()
    return prod

def get_log_act_sel(log_act_w_acts_taken):
    log_act_prob, acts_taken_sing = log_act_w_acts_taken
    return log_act_prob.gather(1, acts_taken_sing.unsqueeze(1))

def get_log_act_prob(logits_out):
    return nn_func.log_softmax(logits_out.squeeze(1), dim=1)