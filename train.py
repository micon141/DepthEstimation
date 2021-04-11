import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn

from utils.utils import get_logger, get_tb_writer, get_device
from dataset import create_dataloaders
from model import ResNet50Encoder, ResNet101Encoder, ResNet152Encoder, ResNet34Encoder
from loss import MonoDepthLoss


class Trainer(object):

    def __init__(self, config):
        self.logger = get_logger()
        self.input_shape = config["Dataset"]["image_size"]
        self.channels = config["Dataset"]["channels"]
        self.train_loader = create_dataloaders(config["Dataset"]["train_data_path"],
                                               config["Dataset"]["image_size"],
                                               config["Train"]["batch_size"],
                                               config["Train"]["augmentation"],
                                               self.logger)
        self.batch_size = config["Train"]["batch_size"]
        self.epochs = config["Train"]["epochs"]
        self.device = get_device(config["Train"]["device"])
        self.model = self.get_model(config["Train"]["architecture"])
        self.optimizer = self.get_optimizer(config["Train"]["optimizer"],
                                            config["Train"]["learning_rate"])
        self.criterion = MonoDepthLoss(self.device)
        self.checkpoints_dir = config["Logging"]["ckpt_dir"]
        # self.writer = get_tb_writer(config["Logging"]["tb_logdir"])

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


    def train(self):
        #torch.autograd.set_detect_anomaly(True)
        params = [p.numel() for p in self.model.parameters() if p.requires_grad]
        total_num_params = 0
        for p in params:
            total_num_params += p
        self.logger.info(f"Number of parameters: {total_num_params}")

        global_step = 0

        self.logger.info(f"---------------- Training Started ----------------")
        for epoch in range(self.epochs):
            epoch += 1
            for batch, (images_left, images_right) in enumerate(self.train_loader):
                batch += 1

                images_left = [image.type(torch.FloatTensor).cuda(self.device) for image in images_left]
                images_right = [image.type(torch.FloatTensor).cuda(self.device) for image in images_right]

                image_left_train = images_left[-1]
                disparities = self.model(image_left_train)

                train_loss = self.criterion(images_left, images_right, disparities, self.device)
                train_loss_total = torch.mean(train_loss)

                if global_step % 10 == 0:
                    print("loss: ", train_loss)
                    print("total_loss: ", train_loss_total)
                self.optimizer.zero_grad()
                train_loss_total.backward(retain_graph=True)
                self.optimizer.step()


                global_step += 1
                


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='',
                        help="Config file")
    args = parser.parse_args()

    with open(args.config, 'r') as cfg_file:
        config = yaml.load(cfg_file, Loader=yaml.FullLoader)

    trainer = Trainer(config)
    trainer.train()
