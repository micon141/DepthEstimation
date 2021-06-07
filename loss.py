import torch
import torch.nn as nn
import torch.nn.functional as F


class MonoDepthLoss(nn.modules.Module):
    def __init__(self):
        super(MonoDepthLoss, self).__init__()

    def scale_images(self, img, num_scales):
        """
        :param img: Input image
        :param num_scales: Number of scales
        :return: List of scaled images
        """
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.functional.interpolate(img,
                               size=[nh, nw], mode='bilinear',
                               align_corners=True))
        return scaled_imgs

    def gradient(self, img, axis):
        """
        :param img: Input image
        :param axis: Axis for computing gradients
        :return:
        """
        if axis == 'x':
            img = F.pad(img, (0, 1, 0, 0), mode="replicate")
            gx = img[:, :, :, :-1] - img[:, :, :, 1:]
            return gx
        elif axis == 'y':
            img = F.pad(img, (0, 0, 0, 1), mode="replicate")
            gy = img[:, :, :-1, :] - img[:, :, 1:, :]
            return gy
        else:
            print("Not valid axis for calculating gradients")
            exit(1)

    def apply_disparity(self, img, disp):
        """
        :param img: Input image
        :param disp: Predicted disparity
        :return: Output image with applied disparity
        """
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='zeros')

        return output

    def SSIM(self, x, y):
        """
        :param x, y: Input tensors for calculating SSIM
        :return: SSIM
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def disp_smoothness(self, disp, images):
        """
        :param disp: Estimated disparity
        :param images: Inpiut image
        :return:
        """
        disp_gradients_x = [self.gradient(d, 'x') for d in disp]
        disp_gradients_y = [self.gradient(d, 'y') for d in disp]

        image_gradients_x = [self.gradient(img, 'x') for img in images]
        image_gradients_y = [self.gradient(img, 'y') for img in images]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i]
                        for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i]
                        for i in range(4)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])
                for i in range(4)]

    def forward(self, input, left, right, appearance_matching_loss_weight,
                smoothness_loss_weight, lr_consistency_loss_weight):
        """
        :param input: Input image
        :param left: Estimated disparity - left
        :param right: Estimated disparity - right
        :param appearance_matching_loss_weight: loss weight
        :param smoothness_loss_weight: loss weight
        :param lr_consistency_loss_weight: loss weight
        :return: loss value
        """
        left_images = self.scale_images(left, 4)
        right_images = self.scale_images(right, 4)

        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in input]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in input]

        left_est = [self.apply_disparity(right_images[i],
                    -disp_left_est[i]) for i in range(4)]
        right_est = [self.apply_disparity(left_images[i],
                     disp_right_est[i]) for i in range(4)]

        right_left_disp = [self.apply_disparity(disp_right_est[i],
                           -disp_left_est[i]) for i in range(4)]
        left_right_disp = [self.apply_disparity(disp_left_est[i],
                           disp_right_est[i]) for i in range(4)]

        disp_left_smoothness = self.disp_smoothness(disp_left_est,
                                                    left_images)
        disp_right_smoothness = self.disp_smoothness(disp_right_est,
                                                     right_images)

        l1_left = [torch.mean(torch.abs(left_est[i] - left_images[i]))
                   for i in range(4)]
        l1_right = [torch.mean(torch.abs(right_est[i]
                    - right_images[i])) for i in range(4)]

        ssim_left = [torch.mean(self.SSIM(left_est[i],
                     left_images[i])) for i in range(4)]
        ssim_right = [torch.mean(self.SSIM(right_est[i],
                      right_images[i])) for i in range(4)]

        image_loss_left = [appearance_matching_loss_weight * ssim_left[i]
                           + (1 - appearance_matching_loss_weight) * l1_left[i]
                           for i in range(4)]
        image_loss_right = [appearance_matching_loss_weight * ssim_right[i]
                            + (1 - appearance_matching_loss_weight) * l1_right[i]
                            for i in range(4)]
        image_loss = sum(image_loss_left + image_loss_right)

        lr_left_loss = [torch.mean(torch.abs(right_left_disp[i]
                        - disp_left_est[i])) for i in range(4)]
        lr_right_loss = [torch.mean(torch.abs(left_right_disp[i]
                         - disp_right_est[i])) for i in range(4)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        disp_left_loss = [torch.mean(torch.abs(
                          disp_left_smoothness[i])) / 2 ** i
                          for i in range(4)]
        disp_right_loss = [torch.mean(torch.abs(
                           disp_right_smoothness[i])) / 2 ** i
                           for i in range(4)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        loss = [image_loss, smoothness_loss_weight * disp_gradient_loss,
                lr_consistency_loss_weight * lr_loss]

        return loss
