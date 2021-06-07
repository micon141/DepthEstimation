import torch


class StepDecayLR():
    def __init__(self, optimizer, lr_init, lr_end, epoch_steps):
        """
        :param optimizer: Optimizer
        :param lr_init: Initial learning rate
        :param lr_end: Final learning rate
        :param epoch_steps: Number of epochs when learning rate will be scaled
        """
        self.optimizer = optimizer
        self.epoch_steps = epoch_steps
        self.lr_init = lr_init
        self.lr_end = lr_end
        self.alpha = (lr_end / lr_init) ** (1 / len(epoch_steps))

    def step(self, epoch):
        step_num = 0
        for ep_step in self.epoch_steps:
            if epoch < ep_step:
                break
            else:
                step_num += 1
        lr = self.lr_init * (self.alpha ** step_num)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

