import torch
import torch.nn as nn
from rl.deep_q.basic import select_q, act_from_q
from .. import trainer

# batch input should be in form ([state_matrix, next_state_matrix, termin_matrix, action_matrix, reward_matrix])
# where same index in each matrix is corresponding to same sample
def batch_loss(batch, reward_decay, q_func, q_func_targ, mseloss):
    states, next_states, termin, actions, rewards = batch

    future_q = select_q(q_func_targ(next_states))
    future_q = future_q.where(torch.logical_not(termin), torch.tensor(0.0, dtype=torch.double)) # terminal state condition - no future rewards

    target_q = rewards + reward_decay * future_q
    q = q_func(states)

    q_taken = q[torch.arange(q.size(0)), actions]
    loss = mseloss(target_q, q_taken)

    return loss

def batch_optim(batch, reward_decay, q_func_optim, q_func_targ, mseloss):
    q_func, optim = q_func_optim

    loss = batch_loss(batch, reward_decay, q_func, q_func_targ, mseloss)
    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss

# things to consider: pushing out older states in the step sequence stored, concat states + actions for dataset
class DeepQTrainer(trainer.Trainer):
    # acts with the expectation of producing a batch of actions
    def act(self, obs, trajec_no): # episilon greedy here
        epis_no = trajec_no
        b_size = obs.shape[0] # assume batched
        rand = torch.zeros([b_size, 1])
        if len(self.steps) >= self.BUFFER_LEN_TO_START:
            rand = torch.rand([b_size, 1])
        return self.act_epsilon(obs, rand, epis_no)

    def new_step(self, sampl):
        self.steps.append(sampl)
        if len(self.steps) >= self.BUFFER_LEN_TO_START and self.new_steps % self.STEPS_FOR_UPDATE == self.STEPS_FOR_UPDATE - 1:
            self.training_update()
        self.new_steps += 1

    # ------ more specific to this class methods ----

    def __init__(self, q_func_optim, q_func_targ, batch_size, reward_decay, targ_update_steps, steps_for_update, buffer_len_to_start, buffer_sample_len):
        assert type(q_func_optim[0]) is type(q_func_targ), "q_func_targ has to be same type of model as q_func"
        self.q_func_optim = q_func_optim
        self.batch_size = batch_size
        self.reward_decay = reward_decay
        self.q_func_targ = q_func_targ
        self.TARG_UPDATE_STEPS = targ_update_steps
        self.STEPS_FOR_UPDATE = steps_for_update
        self.BUFFER_LEN_TO_START = buffer_len_to_start
        self.BUFFER_SAMPLE_LEN = buffer_sample_len

        self.steps = []
        self.update_counter = 0
        self.new_steps = 0

        self.MSE = nn.MSELoss()

    def training_update(self):
        step_loader = torch.utils.data.DataLoader(
            self.steps[-min(self.BUFFER_SAMPLE_LEN, len(self.steps)):], # option to sample from later stage of buffer
            batch_size=self.batch_size, 
            shuffle=True,
            )
        samples = next(iter(step_loader))
        # batch loss and optim
        loss = batch_optim(samples, self.reward_decay, self.q_func_optim, self.q_func_targ, self.MSE)
        # decide whether to update target network
        if self.update_counter % self.TARG_UPDATE_STEPS == self.TARG_UPDATE_STEPS - 1:
            q_func, _ = self.q_func_optim
            self.q_func_targ.load_state_dict(q_func.state_dict())
            
        self.update_counter += 1
        return loss

    #assuming epsilon greedy exploration
    def set_epsilon_policy(self, ep_pol):
        self.ep_policy = ep_pol

    def act_epsilon(self, obs, rand, epis_no):
        q_func, _ = self.q_func_optim
        q = q_func(obs)

        q_act = act_from_q(q) # deterministic from model q values
        q_act = q_act.unsqueeze(1)

        b_size, act_size = q.shape
        rand_act = torch.randint(high=act_size, size=[b_size, 1]) # random action

        # decide which one to pick for each action
        epsil = self.ep_policy(epis_no)
        explore = rand < epsil

        return rand_act.where(explore, q_act)