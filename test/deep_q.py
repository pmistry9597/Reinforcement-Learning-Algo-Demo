from rl.deep_q.lunar.model import *
from rl.deep_q.basic import select_q, act_from_q
from rl.deep_q.trainer import DeepQTrainer, batch_loss
import unittest
import math

# --- this test file was meant to test things that I felt have a high chance of failure but would be quite difficult to debug during actual training
# some tests here *may* occasionally fail due to use of torch's random values/weights and assumption that they'll be pretty random
# although so far, I think I've mitigated them

class DeepQFnsTest(unittest.TestCase):
    def test_batch_loss(self):
        N = 120
        obs_size = 140
        act_size = 98
        states = torch.randn([N, obs_size], dtype=torch.double)
        next_states = torch.randn([N, obs_size], dtype=torch.double)
        actions = torch.randint(high=act_size, size=[N])
        rewards = torch.randn([N], dtype=torch.double)
        termin = torch.randn([N], dtype=torch.double) > 0

        q_func = DeepQLunar(obs_size, act_size)
        # optim = torch.optim.RMSprop(q_func.parameters())
        q_func_targ = DeepQLunar(obs_size, act_size)

        batch = (states, next_states, termin, actions, rewards)
        reward_decay = 0.8
        mseloss = nn.MSELoss()

        q_func.train()
        loss_real = batch_loss(batch, reward_decay, q_func, q_func_targ, mseloss)

        # expected result
        cum_loss = torch.tensor(0.0, dtype=torch.double)
        for n in range(N):
            future_q = 0
            if not termin[n]:
                future_q = select_q(q_func_targ(torch.unsqueeze(next_states[n], 0)))
            
            target = rewards[n] + reward_decay * future_q
            q = q_func(torch.unsqueeze(states[n], 0))
            q_taken = torch.squeeze(q)[actions[n]]
            cum_loss += torch.pow(target - q_taken, 2).item()
        mean_loss = torch.squeeze(cum_loss / N)

        # assert False
        self.assertTrue(torch.all(torch.isclose(loss_real, mean_loss, atol=0.0001)))

import copy

class DeepQTrainerTest(unittest.TestCase):
    def test_q_func_param_update__new_step(self):
        in_size = 80
        act_size = 23
        q_func = DeepQLunar(in_size, act_size)
        optim = torch.optim.RMSprop(q_func.parameters())
        q_func_targ = DeepQLunar(in_size, act_size)
        q_func_targ.load_state_dict(q_func.state_dict())
        trainer = DeepQTrainer((q_func, optim), q_func_targ, batch_size=16, reward_decay=0.6, targ_update_steps=5, steps_for_update=1, buffer_len_to_start=1, buffer_sample_len=math.inf)

        orig_pars = copy.deepcopy(list(q_func.parameters()))
        # form ([state_matrix, next_state_matrix, termin_matrix, action_matrix, reward_matrix])
        sars = (torch.randn([in_size], dtype=torch.double),torch.randn([in_size], dtype=torch.double),torch.zeros([1], dtype=torch.bool),torch.randint(high=act_size, size=[1]),torch.randn(1, dtype=torch.double))
        trainer.new_step(sars)

        pars = list(q_func.parameters())
        self.assertGreater(len(orig_pars) * len(pars), 0)
        for i, f in zip(orig_pars, pars):
            self.assertFalse(torch.all(torch.isclose(i, f, atol=0.0000001)))

    # potential POF - target network not all tensor parameters updating under low amount of its
    def test_target_q_update__new_step(self):
        in_size = 20
        act_size = 9
        targ_update_steps = 5
        q_func = DeepQLunar(in_size, act_size)
        optim = torch.optim.RMSprop(q_func.parameters())
        q_func_targ = DeepQLunar(in_size, act_size)
        trainer = DeepQTrainer((q_func, optim), q_func_targ, batch_size=16, reward_decay=0.6, targ_update_steps=targ_update_steps, steps_for_update=1, buffer_len_to_start=1, buffer_sample_len=math.inf)

        orig_targ_pars = copy.deepcopy(list(q_func_targ.parameters()))

        # form ([state_matrix, next_state_matrix, termin_matrix, action_matrix, reward_matrix])
        for _ in range(targ_update_steps):
            sars = (torch.randn([in_size], dtype=torch.double),torch.randn([in_size], dtype=torch.double),torch.zeros([1], dtype=torch.bool),torch.randint(high=act_size, size=[1]),torch.tensor(3.0, dtype=torch.double))
            trainer.new_step(sars)

        targ_pars = list(q_func_targ.parameters())
        self.assertGreater(len(orig_targ_pars) * len(targ_pars), 0)
        # print(type(orig_targ_pars[0]))
        for i, f in zip(orig_targ_pars, targ_pars):
            # print(torch.sum(torch.any(torch.isclose(i, f, atol=0.0000001))))
            self.assertFalse(torch.all(torch.isclose(i.detach(), f.detach(), atol=0.0000001)))

    # ----- more specific to this class

    def test_epsilon_explores__act_epsilon(self):
        ep = 0.4

        b_size = 10

        in_size = 20
        act_size = 9
        targ_update_steps = 5
        q_func = DeepQLunar(in_size, act_size)
        optim = torch.optim.RMSprop(q_func.parameters())
        q_func_targ = DeepQLunar(in_size, act_size)
        trainer = DeepQTrainer((q_func, optim), q_func_targ, batch_size=16, reward_decay=0.6, targ_update_steps=targ_update_steps, steps_for_update=1, buffer_len_to_start=1, buffer_sample_len=math.inf)
        trainer.set_epsilon_policy(lambda count: ep)

        obs = torch.randn([b_size, in_size], dtype=torch.double)
        q = q_func(obs)

        acts_raw = self.select_best_from_q(q)
        actual_acts = torch.tensor(acts_raw)

        rand_in = torch.zeros([b_size, 1]) # force exploration
        out = trainer.act_epsilon(obs, rand_in, 0)

        self.assertFalse(torch.all(torch.isclose(actual_acts, out, atol=0.000001))) # if any values are off (fakse in all), unlikely that q predicted action presented

    def test_epsilon_greedy__act_epsilon(self):
        ep = 0.4

        b_size = 10

        in_size = 20
        act_size = 9
        targ_update_steps = 5
        q_func = DeepQLunar(in_size, act_size)
        optim = torch.optim.RMSprop(q_func.parameters())
        q_func_targ = DeepQLunar(in_size, act_size)
        trainer = DeepQTrainer((q_func, optim), q_func_targ, batch_size=16, reward_decay=0.6, targ_update_steps=targ_update_steps, steps_for_update=1, buffer_len_to_start=1, buffer_sample_len=math.inf)
        trainer.set_epsilon_policy(lambda count: ep)

        obs = torch.randn([b_size, in_size], dtype=torch.double)
        q = q_func(obs)

        acts_raw = self.select_best_from_q(q)
        actual_acts = torch.tensor(acts_raw)

        rand_in = torch.ones([b_size, 1]) # force exploration
        out = trainer.act_epsilon(obs, rand_in, 0)

        self.assertTrue(torch.all(torch.isclose(actual_acts, out, atol=0.000001))) # all must be the same!
    
    def test_ep_policy_changes_exploration__act_epsilon(self):
        ep_pol = lambda count: 1 - count # 1 on first step, 0 on second
        expect_greedy = [False, True] # directly based on epsilon policy

        b_size = 10

        in_size = 20
        act_size = 3
        targ_update_steps = 5
        q_func = DeepQLunar(in_size, act_size)
        fake_net = DeepQLunar(in_size, act_size)
        optim = torch.optim.RMSprop(fake_net.parameters()) #use ineffective optimizer to prevent model from changin
        q_func_targ = DeepQLunar(in_size, act_size)
        trainer = DeepQTrainer((q_func, optim), q_func_targ, batch_size=16, reward_decay=0.6, targ_update_steps=targ_update_steps, steps_for_update=1, buffer_len_to_start=1, buffer_sample_len=math.inf)
        trainer.set_epsilon_policy(ep_pol)

        obs = (torch.randn([b_size, in_size], dtype=torch.double),torch.randn([b_size, in_size], dtype=torch.double))
        qs = list(map(lambda o: q_func(o), obs))

        acts_raw = list(map(self.select_best_from_q, qs))
        actual_acts = list(map(lambda act: torch.tensor(act), acts_raw))

        rand_in = torch.ones([b_size, 1]) * 0.6 # just not an extreme value, so episilon will have effect
        outs = []
        for e, o in enumerate(obs):
            outs.append(trainer.act_epsilon(o, rand_in, e))
            sampl = (torch.randn([in_size], dtype=torch.double), torch.randn([in_size], dtype=torch.double), torch.zeros([1], dtype=torch.bool), torch.randint(high=act_size, size=[1]), torch.ones([1]))
            trainer.new_step(sampl) #value of the sample actually doesn't matter

        self.assertGreater(len(actual_acts) * len(outs) * len(expect_greedy), 0)
        for act, out, exp in zip(actual_acts,outs,expect_greedy):
            self.assertEqual(exp, torch.all(torch.isclose(act, out, atol=0.000001)))

    def select_best_from_q(self, q):
        b_size = len(q)
        act_size = len(q[0])
        acts_raw = []
        for b in range(b_size): # batch level
            m = -math.inf
            i = -1
            for a in range(act_size):
                q_val = q[b, a]
                if q_val > m:
                    m = q_val
                    i = a
            acts_raw.append([i])
        return acts_raw