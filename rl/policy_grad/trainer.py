from .. import trainer
from functools import partial
import torch

class PolicyGradTrainer(trainer.Trainer):
    def act(self, obs, trajec_no):
        # self.trajec_no = trajec_no
        return self.actor(obs)

    def new_step(self, sampl):
        self.trajecs[-1].append(sampl)

    def termin_trajectory(self):
        # note: trigger training update here when right
        # and create new lists for recording actions, policy prob outputs, etc
        pass

    # --- specific to class section ---

    def __init__(self, policy_optim, actor, reward_decay, trajecs_til_update):
        self.trajecs = []
        self.actor = actor
        self.policy_optim = policy_optim
        self.reward_decay = reward_decay
        self.trajecs_til_update = trajecs_til_update

# calculate loss for policy gradient method
# take in trajectories to compute over, advantage/reward fn of trajectory, policy probability outputs recorded, actions actually taken
def trajec_loss(trajecs_rewards, reward_decay, advantage_fn, policy_outs, acts_taken):
    advantages_tup = tuple(map(partial(trajec_advantage, reward_decay=reward_decay, advantage_fn=advantage_fn), trajecs_rewards))
    advantages = torch.tensor(advantages_tup)
    
    policy_outs = torch.stack(policy_outs)
    prob_of_acts = policy_outs.gather(2, acts_taken.unsqueeze(2)) # 2 is dim to select individual probabilities

    # policy loss formula <- sum of(log of acts .* advantages)
    log_act_prob = torch.log(prob_of_acts)
    return torch.mean(log_act_prob.squeeze() * advantages)

    # note: may need to update as may accept tensor instead of list of trajectory features

def trajec_advantage(trajec_rewards, reward_decay, advantage_fn):
    trajec_rewards = torch.stack(trajec_rewards)
    return tuple(map(lambda i: advantage_fn(trajec_rewards[i:], reward_decay), range(len(trajec_rewards))))