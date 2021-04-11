import torch
import torch.nn as nn
import numpy as np

from utils.utils import generate_image, disparity_smoothness_loss, \
                        appearance_matching_loss, left_right_disp_consistency_loss

class MonoDepthLoss(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.disparities_left = []
        self.disparities_right = []
        self.images_left_estimated = []
        self.images_right_estimated = []
        self.device = device

    def get_output(self, images_left, images_right, disparities, device):
        self.disparities_left = [torch.unsqueeze(disp[:, 0, :, :], 1) for disp in disparities]
        self.disparities_right = [torch.unsqueeze(disp[:, 1, :, :], 1) for disp in disparities]
        self.disparities_left = [-disparities for disparities in self.disparities_left]
        for imgs_left, imgs_right, disparities_left, disparities_right in \
                zip(images_left, images_right, self.disparities_left, self.disparities_right):
            self.images_right_estimated.append(generate_image(imgs_left, disparities_right))
            self.images_left_estimated.append(generate_image(imgs_right, disparities_left))

    def forward(self, images_left, images_right, disparities, device):
        images_left = [img.cpu() for img in images_left]
        images_right = [img.cpu() for img in images_right]
        disparities = [disp.cpu() for disp in disparities]
        self.get_output(images_left, images_right, disparities, device)

        appearance_matching_loss_left = appearance_matching_loss(images_left,
                                                                 self.images_left_estimated)
        appearance_matching_loss_right = appearance_matching_loss(images_right,
                                                                  self.images_right_estimated)
        appearance_matching_loss_total = appearance_matching_loss_left + appearance_matching_loss_right

        print(f"appearance_matching_loss_total: {appearance_matching_loss_total}")

        smoothness_loss_left = disparity_smoothness_loss(self.disparities_left,
                                                         images_left)
        smoothness_loss_right = disparity_smoothness_loss(self.disparities_right,
                                                          images_right)
        smoothness_loss_total = smoothness_loss_left + smoothness_loss_right

        print(f"smoothness_loss: {smoothness_loss_total}")

        left_right_disp_consistency_loss_right = left_right_disp_consistency_loss(self.disparities_left,
                                                                                  self.disparities_right)
        left_right_disp_consistency_loss_left = left_right_disp_consistency_loss(self.disparities_right,
                                                                                 self.disparities_left)
        left_right_disp_consistency_loss_total = left_right_disp_consistency_loss_left + \
                                                 left_right_disp_consistency_loss_right
        print(f"left_right_disp_consistency_loss_total: {left_right_disp_consistency_loss_total}")

        return torch.stack([appearance_matching_loss_total, smoothness_loss_total,
                            left_right_disp_consistency_loss_total], dim=0)
