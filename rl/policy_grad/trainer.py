from .. import trainer
from functools import partial, reduce
from rl.policy_grad.basic import sample_act
import torch
import torch.nn.functional as nn_func

# calculate loss for policy gradient method
# take in trajectories to compute over, advantage/reward fn of trajectory, policy probability outputs recorded, actions actually taken
def trajecs_loss(trajecs_rewards, reward_decay, advantage_fn, logits_outs, acts_taken):
    # mean_trajec = trajecs_mean(trajecs_rewards)
    # advantage_fn = partial(mean_adv, mean_trajec=mean_trajec)
    # -- above is section of pain --
    # print("mean:", mean_trajec)

    advantages_map = map(partial(trajec_advantage, reward_decay=reward_decay, advantage_fn=decayed_advantage), trajecs_rewards)
    decayed_advs = tuple(map(torch.tensor, advantages_map))
    mean_trajecs = torch.mean(torch.cat(decayed_advs, dim=0))
    mean_baseline_adv = tuple(map(lambda trajec: trajec - mean_trajecs, decayed_advs))
    advantages_tens = mean_baseline_adv #decayed_advs # ignore mean baseline while we fix policy grad computation issue

    # --- all above needs to be changed to generic fn system to take in entire batch ---

    log_act_probs = map(get_log_act_prob, logits_outs)
    log_act_sels = tuple(map(get_log_act_sel, zip(log_act_probs, acts_taken)))

    losses = tuple(map(mul_both, zip(advantages_tens, log_act_sels)))
    # print(losses)
    # total = sum(map(len, losses))
    accum_losses = tuple(map(torch.sum, losses))
    overall_loss = torch.mean(torch.stack(accum_losses))
    # total_loss = reduce(lambda x, y: x + y, accum_losses, torch.tensor(0.0, dtype=torch.double))
    # print("loss:", total_loss / total)
    return overall_loss

    # note: may need to update as may accept tensor instead of list of trajectory features

def decayed_advantage(rew_traj, decay):
    # rew_traj += rew_traj + torch.ones_like(rew_traj) * -0.001
    return torch.sum( decay ** torch.arange(len(rew_traj)) * rew_traj )

def trajec_advantage(trajec_rewards, reward_decay, advantage_fn):
    trajec_rewards = torch.stack(trajec_rewards)
    return tuple(map(lambda i: advantage_fn(trajec_rewards[i:], reward_decay), range(len(trajec_rewards))))

def trajecs_mean(trajecs_rewards):
    trajecs_tens = map(torch.stack, trajecs_rewards)
    trajecs_single = torch.cat(tuple(trajecs_tens), dim=0)
    return torch.mean(trajecs_single)

def mean_adv(rew_trajec, reward_decay, mean_trajec):
    # print("adv before mean:", torch.sum( reward_decay ** torch.arange(len(rew_trajec)) * rew_trajec ))
    return torch.sum( reward_decay ** torch.arange(len(rew_trajec)) * rew_trajec ) - mean_trajec

class PolicyGradTrainer(trainer.Trainer):
    def act(self, obs, trajec_no):
        policy, _ = self.policy_optim
        logits_out = policy(obs)
        self.logits_outs[-1].append(logits_out)
        # print(pol_out)
        action = sample_act(logits_out)
        # print('action', action)
        return action

    def new_step(self, sampl):
        # print(sampl)
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

    def __init__(self, policy_optim, reward_decay, trajecs_til_update, entropy_coef, discard_non_termined, advantage_fn=decayed_advantage):
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
        # print(trajecs_rewards[0], logits_outs[0], acts_taken[0])
        entropy_total = torch.mean(torch.cat(tuple(map(lambda t: t.view([-1]), map(entropy, logits_outs))), dim=0))
        # print(entropy_total)
        # print("fak:", trajecs_rewards, logits_outs, acts_taken)
        return -trajecs_loss(trajecs_rewards, self.reward_decay, self.advantage_fn, logits_outs, acts_taken) + -self.ENTROPY_COEF * entropy_total

    def update_policy(self):
        loss = self.compute_loss()
        # print(loss)

        _, optim = self.policy_optim
        optim.zero_grad()
        loss.backward()
        optim.step()

        return loss

    # note: test this garbage class

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