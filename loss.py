import torch
import torch.nn as nn
import torch.nn.functional as F


def scale_image(image, size):
    scaled_image = nn.functional.interpolate(image, size=size, mode='bilinear', align_corners=True)
    return scaled_image


def generate_image(images_batch, disparities):
    batch_size, _, height, width = images_batch.size()
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                                                height, 1).type_as(images_batch)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                                                 width, 1).transpose(1, 2).type_as(images_batch)

    x_shifts = disparities[:, 0, :, :]
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    output = F.grid_sample(images_batch, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')

    return output


def gradient(array, axis='x'):
    if axis == 'x':
        if len(array.shape) == 4:
            x_grad = array[:, :, :, :-1] - array[:, :, :, 1:]
            return x_grad
        else:
            raise ValueError(f"Invalid input shape: {array.shape}. Expected 4D array")
    elif axis == 'y':
        if len(array.shape) == 4:
            y_grad = array[:, :, :-1, :] - array[:, :, 1:, :]
            return y_grad
        else:
            raise ValueError(f"Invalid input shape: {array.shape}. Expected 4D array")
    else:
        raise ValueError(f"Invalid axis name: {axis}. Expected 'x' or 'y'")


def structural_similarity(image_gt, image_est):
    C1 = 0.0001
    C2 = 0.0005

    image_gt_mean = F.avg_pool2d(image_gt, kernel_size=3, stride=1)
    image_est_mean = F.avg_pool2d(image_est, kernel_size=3, stride=1)
    image_gt_sigma = F.avg_pool2d(image_gt ** 2, kernel_size=3, stride=1)
    image_gt_sigma = image_gt_sigma - image_gt_mean ** 2
    image_est_sigma = F.avg_pool2d(image_est ** 2, kernel_size=3, stride=1)
    image_est_sigma = image_est_sigma - image_est_mean ** 2
    image_gt_image_est_sigma = F.avg_pool2d(image_gt * image_est, kernel_size=3, stride=1)
    image_gt_image_est_sigma = image_gt_image_est_sigma - image_est_mean * image_gt_mean

    ssim = ((2 * image_est_mean * image_gt_mean + C1) * (2 * image_gt_image_est_sigma + C2)) /\
           ((image_gt_mean ** 2 + image_est_mean ** 2 + C1) * (image_gt_sigma + image_est_sigma + C2))
    return ssim


def appearance_matching_loss(image_scaled, image_estimated):

    similarity = structural_similarity(image_scaled, image_estimated)
    similarity = torch.mean(similarity)
    return similarity


def disparity_smoothness_loss(disparity, image):

    dx_disparity = gradient(disparity, 'x')
    dy_disparity = gradient(disparity, 'y')

    dx_image = gradient(image, 'x')
    dy_image = gradient(image, 'y')

    dx_disparities_abs = torch.abs(dx_disparity)
    dy_disparities_abs = torch.abs(dy_disparity)

    dx_images_exp = torch.exp(-torch.mean(torch.abs(dx_image), dim=3, keepdim=True))
    dy_images_exp = torch.exp(-torch.mean(torch.abs(dy_image), dim=3, keepdim=True))

    smoothness_loss = torch.mean(dx_disparities_abs * dx_images_exp) + \
                      torch.mean(dy_disparities_abs * dy_images_exp)

    return smoothness_loss


def left_right_disp_consistency_loss(disparity1, disparity2):

    predicted_disparity = disparity1
    projected_disparity = generate_image(predicted_disparity, disparity2)

    left_right_disp_cons = torch.abs(predicted_disparity - projected_disparity)
    left_right_disp_cons = torch.mean(left_right_disp_cons)
    return left_right_disp_cons


class MonoDepthLoss(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.disparities_left = []
        self.disparities_right = []
        self.images_left_estimated = []
        self.images_right_estimated = []
        self.device = device

    def get_disparities(self, disparities):
        self.disparities_left = [torch.unsqueeze(disp[:, 0, :, :], 1) for disp in disparities]
        self.disparities_right = [torch.unsqueeze(disp[:, 1, :, :], 1) for disp in disparities]
        self.disparities_left = [-disparities for disparities in self.disparities_left]

    def forward(self, images_left, images_right, disparities):
        disparities = [disp for disp in disparities]
        self.get_disparities(disparities)

        smoothness_loss_total = []
        appearance_matching_loss_total = []
        left_right_disp_consistency_loss_total = []
        for num in range(len(self.disparities_left)):
            disparity_left = self.disparities_left[num]
            disparity_right = self.disparities_right[num]
            shape = [disparity_left.shape[-2], disparity_left.shape[-1]]

            images_left_scaled = scale_image(images_left, shape)
            images_right_scaled = scale_image(images_right, shape)

            images_left_estimated = generate_image(images_left_scaled, disparity_left)
            images_right_esitamted = generate_image(images_right_scaled, disparity_right)

            appearance_matching_loss_left = appearance_matching_loss(images_left_scaled,
                                                                     images_left_estimated)
            appearance_matching_loss_right = appearance_matching_loss(images_right_scaled,
                                                                      images_right_esitamted)
            appearance_matching_loss_total.append(appearance_matching_loss_left +
                                                  appearance_matching_loss_right)

            smoothness_loss_left = disparity_smoothness_loss(disparity_left,
                                                             images_left_scaled)
            smoothness_loss_right = disparity_smoothness_loss(disparity_right,
                                                              images_right_scaled)
            smoothness_loss_scaled = (smoothness_loss_left + smoothness_loss_right) / ((num + 1) ** 2)
            smoothness_loss_total.append(smoothness_loss_scaled)

            left_right_disp_consistency_loss_right = left_right_disp_consistency_loss(disparity_left,
                                                                                      disparity_right)
            left_right_disp_consistency_loss_left = left_right_disp_consistency_loss(disparity_right,
                                                                                     disparity_left)
            left_right_disp_consistency_loss_total.append(left_right_disp_consistency_loss_left +
                                                          left_right_disp_consistency_loss_right)

        appearance_matching_loss_mean = torch.mean(torch.stack(appearance_matching_loss_total, dim=0))

        smoothness_loss_mean = torch.mean(torch.stack(smoothness_loss_total, dim=0))

        left_right_disp_consistency_loss_mean = torch.mean(torch.stack(left_right_disp_consistency_loss_total, dim=0))

        return torch.stack([appearance_matching_loss_mean, smoothness_loss_mean,
                            left_right_disp_consistency_loss_mean], dim=0)
