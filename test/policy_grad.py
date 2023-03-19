import rl.policy_grad.trainer as pol_train
from rl.policy_grad.lunar.model import PolicyGradNNLunar
from rl.policy_grad import advantage_fns
import unittest
import torch
from torch.nn.functional import softmax, log_softmax
from functools import partial, reduce
from copy import deepcopy

class PolicyGradTrainerFnsTest(unittest.TestCase):
    def test_trajecs_loss_rewards_to_go_adv(self):
        trajec_rewards = [(1,-5,-69, 45, 343,), (3,4), (45, 343, -69,)]
        trajec_rewards = tuple(map(lambda seq: tuple(map(torch.tensor, seq)), trajec_rewards))
        reward_decay = 0.9
        advantage_fn = advantage_fns.reward_to_go

        logits_outs = [tuple(map(lambda ten: torch.tensor(ten, dtype=torch.double), ((3,5,4),(1,2,0),(10,4,2),(1,2,9),(10,4,34)))), tuple(map(lambda ten: torch.tensor(ten, dtype=torch.double), ((4,3),(6,4),))), tuple(map(lambda ten: torch.tensor(ten, dtype=torch.double), ((4,3),(6,4),(6,4),)))]
        logits_outs = tuple(map(torch.stack, logits_outs))
        # print(logits_outs[0].shape)

        acts_taken = [torch.tensor((0,1,2,0,2)), torch.tensor((1,1)), torch.tensor((1,1,0))]
        actual_loss_val = pol_train.trajecs_loss(trajec_rewards, reward_decay, advantage_fn, logits_outs, acts_taken)

        decay_fct = reward_decay ** torch.arange(5)
        expected_loss_seq = []
        # self.assertGreater(0, len(trajec_rewards))
        for full_traj, act_traj, pol_traj in zip(trajec_rewards, acts_taken, logits_outs):
            # self.assertGreater(0, len(full_traj))
            expected_loss_seq.append([])
            for i in range(len(full_traj)):
                adv = reduce(lambda bef, e_r: bef + (e_r[1] * reward_decay ** e_r[0]), enumerate(full_traj[-(i+1):]), torch.tensor(0.0, dtype=torch.double))
                act = act_traj[-(i+1)]
                log_pol = log_softmax(pol_traj[-(i+1)], dim=0)[act]
                # print(adv, act, log_pol)
                loss_single = log_pol * adv
                expected_loss_seq[-1].append(loss_single)
        # print(expected_loss_seq)
        exp_loss_means = tuple(map(lambda k: torch.sum(torch.stack(k)), expected_loss_seq))
        expected_loss_val = torch.mean(torch.stack(exp_loss_means))
        self.assertTrue(torch.all(torch.isclose(actual_loss_val.detach(), expected_loss_val.detach(), atol=0.000001)))

class PolicyGradTrainerTest(unittest.TestCase):
    def test_correct_loss_returned_rewards_to_go_adv(self):
        trajec_rewards = [(343, 5, 69, 45, 2,), (3,4)]
        reward_decay = 0.9
        advantage_fn = advantage_fns.reward_to_go

        # require grad to pass into update policy fn
        logits_outs = [list(map(lambda ten: torch.tensor(ten, dtype=torch.double), ((3,5,4),(1,2,0),(10,4,2),(1,2,9),(10,4,34)))), tuple(map(lambda ten: torch.tensor(ten, dtype=torch.double), ((4,3),(6,4),)))]
        # print(logits_outs[0].shape)

        acts_taken = (torch.tensor((0,1,2,0,2)), torch.tensor((1,1)))
        # print("hollow")
        trajec_rewards_ex = tuple(map(lambda seq: tuple(map(torch.tensor, seq)), trajec_rewards))
        logits_outs_ex = tuple(map(torch.stack, logits_outs))
        expected_loss_val = -pol_train.trajecs_loss(trajec_rewards_ex, reward_decay, advantage_fn, logits_outs_ex, acts_taken)
        trajec_rewards = tuple(map(torch.tensor, trajec_rewards)) # tensor conversion for trainer object

        policy = PolicyGradNNLunar(2, 3)
        optim = torch.optim.SGD(policy.parameters(), lr=0.01)
        trajecs_til_update = 2
        trainer = pol_train.PolicyGradTrainer((policy, optim), reward_decay, trajecs_til_update, entropy_coef=0.0, discard_non_termined=False)
        trainer.logits_outs = logits_outs
        trainer.trajecs = [[], []]
        for j in range(len(trajec_rewards)):
            for rew, act in zip(trajec_rewards[j], acts_taken[j]):
                trainer.trajecs[j].append((-1,-1,-1,act,rew))
        actual_loss = trainer.compute_loss()

        # print("fucking penis:", expected_loss_val.detach(), actual_loss.detach())
        self.assertTrue(torch.all(torch.isclose(expected_loss_val.detach(), actual_loss.detach(), atol=0.000001)))

    def test_update_policy(self):
        policy = PolicyGradNNLunar(2, 3)
        optim = torch.optim.SGD(policy.parameters(), lr=0.01)
        reward_decay = 0.8
        trajecs_til_update = 4
        trainer = pol_train.PolicyGradTrainer((policy, optim), reward_decay, trajecs_til_update, entropy_coef=1.0, discard_non_termined=False)

        tens = partial(torch.tensor, dtype=torch.double)
        tensi = partial(torch.tensor, dtype=torch.long)
        trainer.trajecs = [[(tens([0.4,-0.1]),tens([0.9,0.4]),True,tensi(2),tens(2.9)),(tens([-0.4,-0.1]),tens([0.1,0.4]),True,tensi(1),tens(-2.9))]]
        # trainer.policy_outs = [[tens([0.1,0.4,0.5]),tens([0.01,0.01,0.98])]]
        obses = map(lambda traj: map(lambda smpl: smpl[0], traj), trainer.trajecs) # take current states for policy probability compute
        logits_outs = tuple(map(lambda obs_traj: tuple(map(lambda obs: policy(obs.unsqueeze(0)), obs_traj)), obses))
        trainer.logits_outs = logits_outs
        
        prev_params = deepcopy(list(policy.parameters()))
        loss = trainer.update_policy()
        # print("celibate ape")
        curr_params = list(policy.parameters())
        for p, c in zip(prev_params, curr_params):
            self.assertFalse(torch.all(torch.isclose(p.detach(), c.detach(), atol=0.000001)))

    def test_nn_update_after_sampling(self):
        policy = PolicyGradNNLunar(2, 3)
        optim = torch.optim.SGD(policy.parameters(), lr=0.01)
        reward_decay = 0.8
        trajecs_til_update = 2
        trainer = pol_train.PolicyGradTrainer((policy, optim), reward_decay, trajecs_til_update, entropy_coef=1.0, discard_non_termined=False)

        tens = partial(torch.tensor, dtype=torch.double)
        tensi = partial(torch.tensor, dtype=torch.long)
        init, final = tens([0.4,-0.1]), tens([-0.4,-0.1])
        
        prev_params = deepcopy(list(policy.parameters()))

        trainer.punctuate_trajectory()
        for t in range(trajecs_til_update):
            act = trainer.act(init.unsqueeze(0), t)
            smpl = (init, final, True, act, torch.tensor(0.8))
            trainer.new_step(smpl)
            trainer.punctuate_trajectory()

        curr_params = list(policy.parameters())
        for p, c in zip(prev_params, curr_params):
            self.assertFalse(torch.all(torch.isclose(p.detach(), c.detach(), atol=0.000001)))