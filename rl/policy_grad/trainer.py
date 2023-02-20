from .. import trainer
from functools import partial
from rl.policy_grad.basic import sample_act
import torch

class PolicyGradTrainer(trainer.Trainer):
    def act(self, obs, trajec_no):
        # note: might have to unsqueeze or squeeze to get dims right for neural net and other shit
        # pol_out = self.actor(obs)
        policy, _ = self.policy_optim
        pol_out = policy(obs)
        self.policy_outs[-1].append(pol_out)
        return sample_act(pol_out.squeeze())

    def new_step(self, sampl):
        self.trajecs[-1].append(sampl)

    def punctuate_trajectory(self):
        # trigger training update here when right
        if len(self.trajecs) % self.TRAJECS_TIL_UPDATE == self.TRAJECS_TIL_UPDATE - 1:
            self.update_policy()
            self.policy_outs = []
            self.trajecs = []
        self.new_buffers()

    # --- specific to class section ---

    def __init__(self, policy_optim, reward_decay, trajecs_til_update):
        # self.actor = actor
        self.policy_optim = policy_optim
        self.reward_decay = reward_decay
        self.TRAJECS_TIL_UPDATE = trajecs_til_update

        self.trajecs = []
        self.policy_outs = []

    def new_buffers(self):
        self.policy_outs.append([])
        self.trajecs.append([])

    def update_policy(self):
        smpl_rew_i = 4
        trajecs_rewards = map(lambda trajec: tuple(map(lambda smpl: smpl[smpl_rew_i], trajec)), self.trajecs)
        smpl_act_i = 3
        acts_taken = tuple(map(lambda trajec: torch.tensor(tuple(map(lambda smpl: smpl[smpl_act_i], trajec))), self.trajecs))
        policy_outs = tuple(map(torch.stack, self.policy_outs))
        loss = trajecs_loss(trajecs_rewards, self.reward_decay, decayed_advantage, policy_outs, acts_taken)

        _, optim = self.policy_optim
        optim.zero_grad()
        loss.backward()
        optim.step()

        return loss

    # note: test this garbage class

# calculate loss for policy gradient method
# take in trajectories to compute over, advantage/reward fn of trajectory, policy probability outputs recorded, actions actually taken
def trajecs_loss(trajecs_rewards, reward_decay, advantage_fn, policy_outs, acts_taken):
    advantages_tup = tuple(map(partial(trajec_advantage, reward_decay=reward_decay, advantage_fn=advantage_fn), trajecs_rewards))
    advantages = torch.tensor(advantages_tup)
    
    policy_outs = torch.stack(policy_outs)
    acts_taken = torch.stack(acts_taken)
    # print("shap", policy_outs.shape, acts_taken.shape)
    prob_of_acts = policy_outs.squeeze(2).gather(2, acts_taken.unsqueeze(2)) # 2 is dim to select individual probabilities

    # policy loss formula <- sum of(log of acts .* advantages)
    log_act_prob = torch.log(prob_of_acts)
    return torch.mean(log_act_prob.squeeze() * advantages)

    # note: may need to update as may accept tensor instead of list of trajectory features

def trajec_advantage(trajec_rewards, reward_decay, advantage_fn):
    trajec_rewards = torch.stack(trajec_rewards)
    return tuple(map(lambda i: advantage_fn(trajec_rewards[i:], reward_decay), range(len(trajec_rewards))))

def decayed_advantage(rew_traj, decay):
    return torch.sum( decay ** torch.arange(len(rew_traj)) * rew_traj )