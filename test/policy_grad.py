import rl.policy_grad.trainer as pol_train
import unittest
import torch
from torch.nn.functional import softmax

class PolicyGradTrainerFnsTest(unittest.TestCase):
    def test_trajec_loss(self):
        trajec_rewards = ([(1,-5,-69), (3,4,-5)])
        trajec_rewards = tuple(map(lambda seq: tuple(map(torch.tensor, seq)), trajec_rewards))
        reward_decay = 0.9
        advantage_fn = lambda rew_traj, decay: torch.sum( decay ** torch.arange(len(rew_traj)) * rew_traj )

        policy_outs = [tuple(map(lambda ten: softmax(torch.tensor(ten, dtype=torch.double), dim=0), ((3,5),(1,2),(10,4)))), tuple(map(lambda ten: softmax(torch.tensor(ten, dtype=torch.double), dim=0), ((4,3),(6,4),(7,3))))]
        policy_outs = tuple(map(torch.stack, policy_outs))

        acts_taken = torch.tensor([(0,1,0,), (1,1,0)])

        actual_loss_val = pol_train.trajec_loss(trajec_rewards, reward_decay, advantage_fn, policy_outs, acts_taken)

        decay_fct = reward_decay ** torch.arange(3)
        expected_loss_seq = []
        # self.assertGreater(0, len(trajec_rewards))
        for full_traj, act_traj, pol_traj in zip(trajec_rewards, acts_taken, policy_outs):
            # self.assertGreater(0, len(full_traj))
            for i in range(len(full_traj)):
                traj = torch.stack(full_traj[-(i+1):])
                adv = torch.sum( traj * decay_fct[:i+1] )
                act = act_traj[-(i+1)]
                log_pol = torch.log(pol_traj[-(i+1),act])
                loss_single = log_pol * adv
                expected_loss_seq.append(loss_single)
        expected_loss_val = torch.mean(torch.stack(expected_loss_seq))
        self.assertEqual(actual_loss_val, expected_loss_val)
        # note: may need to update test as function may accept tensor instead of list of trajectory features