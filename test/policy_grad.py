import rl.policy_grad.trainer as pol_train
from rl.policy_grad.lunar.model import PolicyGradNNLunar
import unittest
import torch
from torch.nn.functional import softmax, log_softmax
from functools import partial
from copy import deepcopy

class PolicyGradTrainerFnsTest(unittest.TestCase):
    def test_trajecs_loss(self):
        trajec_rewards = ([(1,-5,-69), (3,4,-5)])
        trajec_rewards = tuple(map(lambda seq: tuple(map(torch.tensor, seq)), trajec_rewards))
        reward_decay = 0.9
        advantage_fn = lambda rew_traj, decay: torch.sum( decay ** torch.arange(len(rew_traj)) * rew_traj )

        logits_outs = [tuple(map(lambda ten: torch.tensor(ten, dtype=torch.double), ((3,5),(1,2),(10,4)))), tuple(map(lambda ten: torch.tensor(ten, dtype=torch.double), ((4,3),(6,4),(7,3))))]
        logits_outs = tuple(map(torch.stack, logits_outs))
        # print(logits_outs[0].shape)

        acts_taken = ([torch.tensor((0,1,0,)), torch.tensor((1,1,0))])
        # print("hollow")
        actual_loss_val = pol_train.trajecs_loss(trajec_rewards, reward_decay, advantage_fn, logits_outs, acts_taken)
        # print("pee")

        decay_fct = reward_decay ** torch.arange(3)
        expected_loss_seq = []
        # self.assertGreater(0, len(trajec_rewards))
        for full_traj, act_traj, pol_traj in zip(trajec_rewards, acts_taken, logits_outs):
            # self.assertGreater(0, len(full_traj))
            for i in range(len(full_traj)):
                traj = torch.stack(full_traj[-(i+1):])
                adv = torch.sum( traj * decay_fct[:i+1] )
                act = act_traj[-(i+1)]
                # print(pol_traj[-(i+1)].shape)
                log_pol = log_softmax(pol_traj[-(i+1)], dim=0)[act]
                # print(log_pol.shape)
                loss_single = log_pol * adv
                expected_loss_seq.append(loss_single)
        # print(expected_loss_seq)
        expected_loss_val = torch.mean(torch.stack(expected_loss_seq))
        self.assertEqual(actual_loss_val, expected_loss_val)
        # note: may need to update test as function may accept tensor instead of list of trajectory features

class PolicyGradTrainerTest(unittest.TestCase):
    def test_update_policy(self):
        policy = PolicyGradNNLunar(2, 3)
        optim = torch.optim.SGD(policy.parameters(), lr=0.01)
        reward_decay = 0.8
        trajecs_til_update = 4
        trainer = pol_train.PolicyGradTrainer((policy, optim), reward_decay, trajecs_til_update)

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
        trainer = pol_train.PolicyGradTrainer((policy, optim), reward_decay, trajecs_til_update)

        tens = partial(torch.tensor, dtype=torch.double)
        tensi = partial(torch.tensor, dtype=torch.long)
        init, final = tens([0.4,-0.1]), tens([-0.4,-0.1])
        # trainer.policy_outs = [[tens([0.1,0.4,0.5]),tens([0.01,0.01,0.98])]]
        # obses = map(lambda traj: map(lambda smpl: smpl[0], traj), trainer.trajecs) # take current states for policy probability compute
        # pol_outs = tuple(map(lambda obs_traj: tuple(map(lambda obs: policy(obs.unsqueeze(0)), obs_traj)), obses))
        # trainer.policy_outs = pol_outs
        
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