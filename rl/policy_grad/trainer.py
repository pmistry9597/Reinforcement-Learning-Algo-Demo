from .. import trainer

class PolicyGradTrainer(trainer.Trainer):
    def act(self, obs, trajec_no):
        # self.trajec_no = trajec_no
        return self.actor(obs)

    def new_step(self, sampl):
        self.trajecs[-1].append(sampl)

    def termin_trajectory(self):
        # note: trigger training loop here when right
        pass

    # --- specific to class section ---

    def __init__(self, policy_optim, actor, reward_decay, trajecs_til_update):
        self.trajecs = []
        self.actor = actor
        self.policy_optim = policy_optim
        self.reward_decay = reward_decay
        self.trajecs_til_update = trajecs_til_update
