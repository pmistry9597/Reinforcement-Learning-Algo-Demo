from .. import trainer
from functools import partial
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
    advantages_tup = tuple(map(partial(trajec_advantage, reward_decay=reward_decay, advantage_fn=advantage_fn), trajecs_rewards))
    advantages = torch.tensor(advantages_tup)
    
    logits_outs = torch.stack(logits_outs)
    acts_taken = torch.stack(acts_taken)
    # print("shap", policy_outs.shape, acts_taken.shape)
    # print(logits_outs.shape)
    log_act_prob = nn_func.log_softmax(logits_outs.squeeze(2), dim=2)
    log_act_sel = log_act_prob.gather(2, acts_taken.unsqueeze(2)) # 2 is dim to select individual probabilities

    # policy loss formula <- sum of(log of acts .* advantages)
    # print(log_act_prob.shape)
    return torch.mean(log_act_sel.squeeze() * advantages)

    # note: may need to update as may accept tensor instead of list of trajectory features

def trajec_advantage(trajec_rewards, reward_decay, advantage_fn):
    trajec_rewards = torch.stack(trajec_rewards)
    return tuple(map(lambda i: advantage_fn(trajec_rewards[i:], reward_decay), range(len(trajec_rewards))))

def decayed_advantage(rew_traj, decay):
    return torch.sum( decay ** torch.arange(len(rew_traj)) * rew_traj )