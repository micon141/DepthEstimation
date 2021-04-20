import os
import argparse
import yaml
from tqdm import tqdm
import torch

from utils import get_logger, get_tb_writer, get_device
from dataset import create_dataloaders
from model import ResNet50Encoder, ResNet101Encoder, ResNet152Encoder, ResNet34Encoder
from loss import MonoDepthLoss
from lr_schedulers import StepDecayLR

class Trainer(object):

    def __init__(self, config):
        self.config = config
        self.logger = get_logger()
        self.input_shape = config["Dataset"]["image_size"]
        self.channels = config["Dataset"]["channels"]
        self.train_loader = create_dataloaders(data_path=config["Dataset"]["train_data_path"],
                                               image_size=config["Dataset"]["image_size"],
                                               batch_size=config["Train"]["batch_size"],
                                               augment=config["Train"]["augmentation"],
                                               logger=self.logger)
        self.val_loader = create_dataloaders(data_path=config["Dataset"]["val_data_path"],
                                             image_size=config["Dataset"]["image_size"],
                                             batch_size=config["Train"]["batch_size"],
                                             augment=None,
                                             logger=self.logger)
        self.batch_size = config["Train"]["batch_size"]
        self.epochs = config["Train"]["epochs"]
        self.device = get_device(config["Train"]["device"])
        if config["Train"]["pretrained"]:
            self.model = self.load_model(config["Train"]["architecture"],
                                         config["Train"]["pretrained"])
        else:
            self.model = self.get_model(config["Train"]["architecture"])
        self.optimizer = self.get_optimizer(config["Train"]["optimizer"],
                                            config["Train"]["lr_scheduler"]["lr_init"])
        self.lr_scheduler = self.get_scheduler(config["Train"]["lr_scheduler"])
        self.criterion = MonoDepthLoss(self.device)
        self.checkpoints_dir = config["Logging"]["ckpt_dir"]
        self.writer = get_tb_writer(config["Logging"]["tb_logdir"])

    def get_model(self, architecture):
        if architecture == "resnet50":
            return ResNet50Encoder(self.input_shape, self.channels).cuda(self.device)
        elif architecture == "resnet101":
            return ResNet101Encoder(self.input_shape, self.channels).cuda(self.device)
        elif architecture == "resnet152":
            return ResNet152Encoder(self.input_shape, self.channels).cuda(self.device)
        elif architecture == "resnet34":
            return ResNet34Encoder(self.input_shape, self.channels).cuda(self.device)
        else:
            raise NotImplementedError(f"Architecture {architecture} is not implemented. "
                                      f"See documentation for supported architectures")

    def load_model(self, arch, pretrained):

        model = self.get_model(arch)
        self.logger.info(f"Loading weights from {pretrained}...")
        model.load_state_dict(torch.load(pretrained))
        model.eval()

        return model.cuda(self.device)

    def save_model(self, epoch, step):
        """ Save model
        Args:
            step: Number of steps
        """
        if not os.path.isdir(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        ckpt_path = os.path.join(self.checkpoints_dir, f"model_{epoch}_{step}.pt")
        torch.save(self.model.state_dict(), ckpt_path)
        self.logger.info(f"Model saved: {ckpt_path}")

    def get_optimizer(self, opt, lr=0.001):
        """
        Args
            opt: string, optimizer from config file
            model: nn.Module, generated model
            lr: float, specified learning rate
        Returns:
            optimizer
        """
        if opt.lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif opt.lower() == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Not supported optimizer name: {opt}."
                                      f"For supported optimizers see documentation")
        return optimizer

    def get_scheduler(self, lr_scheduler_dict):
        lr_init = lr_scheduler_dict["lr_init"]
        if lr_scheduler_dict["StepDecay"]["use"]:
            lr_scheduler = StepDecayLR(optimizer=self.optimizer,
                                       lr_init=lr_init,
                                       lr_end=lr_scheduler_dict["StepDecay"]["lr_end"],
                                       epoch_steps=lr_scheduler_dict["StepDecay"]["epoch_steps"])
            return lr_scheduler
        else:
            return None

    def train(self):
        params = [p.numel() for p in self.model.parameters() if p.requires_grad]
        total_num_params = 0
        for p in params:
            total_num_params += p
        self.logger.info(f"Number of parameters: {total_num_params}")

        global_step = 0
        self.logger.info(f"---------------- Training Started ----------------")
        for epoch in range(self.epochs):
            epoch += 1
            for image_left, image_right in tqdm(self.train_loader):
                batches = len(self.train_loader)
                image_left = image_left.type(torch.FloatTensor).cuda(self.device)
                image_right = image_right.type(torch.FloatTensor).cuda(self.device)
                disparities = self.model(image_left)

                train_loss = self.criterion(image_left, image_right, disparities)
                train_loss_total = torch.mean(train_loss)

                self.optimizer.zero_grad()
                train_loss_total.backward()
                self.optimizer.step()
                global_step += 1

                self.writer.add_scalar("Train.appearance_matching_loss", train_loss[0].item(),
                                       global_step)
                self.writer.add_scalar("Train.smoothness_loss", train_loss[2].item(),
                                       global_step)
                self.writer.add_scalar("Train.left_right_disparity_consistency_loss", train_loss[2].item(),
                                       global_step)
                self.writer.add_scalar("Train.total_loss", train_loss_total.item(),
                                       global_step)

                if global_step % (batches // self.config["Train"]["evals_per_epoch"]) == 0:
                    self.save_model(epoch, global_step)
                    self.logger.info("Evaluating model...")
                    for image_left, image_right in tqdm(self.val_loader):
                        image_left = image_left.type(torch.FloatTensor).cuda(self.device)
                        image_right = image_right.type(torch.FloatTensor).cuda(self.device)
                        val_disparities = self.model(image_left)
                        val_loss = self.criterion(image_left, image_right, val_disparities)
                        val_loss_total = torch.mean(val_loss)
                        self.writer.add_scalar("Val.appearance_matching_loss", val_loss[0].item(),
                                               global_step)
                        self.writer.add_scalar("Val.smoothness_loss", val_loss[2].item(),
                                               global_step)
                        self.writer.add_scalar("Val.left_right_disparity_consistency_loss", val_loss[2].item(),
                                               global_step)
                        self.writer.add_scalar("Val.total_loss", val_loss_total.item(),
                                               global_step)
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='',
                        help="Config file")
    args = parser.parse_args()

    with open(args.config, 'r') as cfg_file:
        config = yaml.load(cfg_file, Loader=yaml.FullLoader)

    trainer = Trainer(config)
    trainer.train()
