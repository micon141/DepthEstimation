import os
import logging
from shutil import rmtree, copyfile
from tensorboardX import SummaryWriter
import torch
import cv2 as cv
import numpy as np
from torch import nn


def get_logger():
    logger = logging.getLogger("Depth Estimation Inspired by Monodepth")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s]-[%(filename)s]: %(message)s ")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def get_tb_writer(tb_logdir, ckpt_dir):
    """
    Args:
        tb_logdir: str, Path to directory fot tensorboard events
        ckpt_dir: str, Path to checkpoints directory
    Return:
        writer: TensorBoard writer
    """
    if '/' in ckpt_dir:
        tb_logdir = os.path.join(tb_logdir, ckpt_dir.split("/")[-1])
    else:
        tb_logdir = os.path.join(tb_logdir, ckpt_dir.split("\\")[-1])
    if os.path.isdir(tb_logdir):
        rmtree(tb_logdir)
    os.mkdir(tb_logdir)
    writer = SummaryWriter(log_dir=tb_logdir)
    return writer


def get_device(device):
    """
    Args:
        device: str, GPU device id
    Return: torch device
    """

    if device == "cpu":
        return torch.device("cpu")
    else:
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"
        c = 1024 ** 2
        x = torch.cuda.get_device_properties(0)
        print("Using GPU")
        print(f"device{device} _CudaDeviceProperties(name='{x.name}'"
              f", total_memory={x.total_memory / c}MB)")
        return torch.device("cuda:0")


def get_disparity(disparities, index):
    disp = []
    for disparity in disparities:
        disp_np = torch.unsqueeze(disparity[:, index, :, :], dim=1).detach().cpu().numpy()
        disp_np = np.transpose(disp_np, (0, 2, 3, 1))
        disp.append(disp_np)
    return disp


def scale_image(image):
    h, w = image.shape[0], image.shape[1]
    scaled_shapes = []
    for num in range(1, 4):
        scaled_shapes.append((w // (2 ** num), h // (2 ** num)))
    images_scaled = [image]
    for scaled_shape in scaled_shapes:
        images_scaled.insert(0, cv.resize(image, tuple(scaled_shape)))

    return images_scaled


def generate_image(images_batch, x_offset):
    # images_batch = images_batch.cpu()
    # x_offset = x_offset.cpu()
    batch = images_batch.shape[0]
    channels = images_batch.shape[1]
    height = images_batch.shape[2]
    width = images_batch.shape[3]

    width_f = float(width)
    height_f = float(height)
    y_t, x_t = torch.meshgrid(torch.linspace(0.0, height_f - 1.0, height),
                              torch.linspace(0.0, width_f - 1.0, width))

    x_t_flat = torch.tile(torch.reshape(x_t, (1, -1)), (batch, 1))
    y_t_flat = torch.tile(torch.reshape(y_t, (1, -1)), (batch, 1))
    x_t_flat = torch.reshape(x_t_flat, [-1])
    y_t_flat = torch.reshape(y_t_flat, [-1])
    x_t_flat = x_t_flat + torch.reshape(x_offset, [-1]) * width_f
    x_t_flat = torch.clip(x_t_flat, 0.0, width_f - 1)

    x0_f = torch.floor(x_t_flat)
    y0_f = torch.floor(y_t_flat)
    x1_f = x0_f + 1

    x0 = x0_f.type(torch.int32)
    y0 = y0_f.type(torch.int32)
    x1 = (torch.minimum(x1_f, torch.tensor(width_f - 1))).type(torch.int32)

    dim2 = width
    dim1 = width * height
    rep = torch.tile(torch.unsqueeze(torch.arange(batch) * dim1, 1), [1, height * width])
    base = torch.reshape(rep, [-1])
    base_y0 = base + y0 * dim2
    idx_l = base_y0 + x0
    idx_r = base_y0 + x1
    images_batch = images_batch.permute(0, 2, 3, 1)
    im_flat = torch.reshape(images_batch, (-1, channels))


    pix_l_list = [torch.take(im_flat[:, num], idx_l) for num in range(channels)]
    pix_l = torch.stack(pix_l_list, dim=1)
    pix_r_list = [torch.take(im_flat[:, num], idx_r) for num in range(channels)]
    pix_r = torch.stack(pix_r_list, dim=1)

    weight_l = torch.unsqueeze(x1_f - x_t_flat, 1)
    weight_r = torch.unsqueeze(x_t_flat - x0_f, 1)

    input_transformed = weight_l * pix_l + weight_r * pix_r
    output = torch.reshape(
        input_transformed, (batch, height, width, channels)).permute(0, 3, 1, 2)

    return output


def gradient(array, axis='x'):
    if axis == 'x':
        if len(array.shape) == 4:
            return array[:, :, :, :-1] - array[:, :, :, 1:]
        else:
            raise ValueError(f"Invalid input shape: {array.shape}. Expected 4D array")
    elif axis == 'y':
        if len(array.shape) == 4:
            return array[:, :, :-1, :] - array[:, :, 1:, :]
        else:
            raise ValueError(f"Invalid input shape: {array.shape}. Expected 4D array")
    else:
        raise ValueError(f"Invalid axis name: {axis}. Expected 'x' or 'y'")


def structural_similarity(image_gt, image_est, kernel_size=3, stride=1):
    C1 = 0.0001
    C2 = 0.0005

    image_gt_mean = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)(image_gt)
    image_est_mean = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)(image_est)

    image_gt_sigma = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)(image_gt ** 2)
    image_gt_sigma -= image_gt_mean ** 2
    image_est_sigma = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)(image_est ** 2)
    image_est_sigma -= image_est_mean ** 2
    image_gt_image_est_sigma = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)(image_gt * image_est)
    image_gt_image_est_sigma -= image_est_mean * image_gt_mean

    ssim = ((2 * image_est_mean * image_gt_mean + C1) * (2 * image_gt_image_est_sigma + C2)) /\
           ((image_gt_mean ** 2 + image_est_mean ** 2 + C1) * (image_gt_sigma + image_est_sigma + C2))
    return ssim


def appearance_matching_loss(images_scaled, images_estimated):

    similarity = [structural_similarity(images_scaled[num], images_estimated[num])
                  for num in range(len(images_scaled))]
    similarity = [torch.mean(sim) for sim in similarity]
    return torch.stack(similarity, dim=0)


def disparity_smoothness_loss(disparities, images_scaled):

    dx_disparities = [gradient(disp, 'x') for disp in disparities]
    dy_disparities = [gradient(disp, 'y') for disp in disparities]

    dx_images = [gradient(image, 'x') for image in images_scaled]
    dy_images = [gradient(image, 'y') for image in images_scaled]

    dx_disparities_abs = [torch.abs(dx_disparity) for dx_disparity in dx_disparities]
    dy_disparities_abs = [torch.abs(dy_disparity) for dy_disparity in dy_disparities]

    dx_images_exp = [torch.exp(-torch.mean(dx_image, dim=3, keepdim=True)) for dx_image in dx_images]
    dy_images_exp = [torch.exp(-torch.mean(dy_image, dim=3, keepdim=True)) for dy_image in dy_images]

    smoothness_loss = [torch.mean(dx_disparities_abs[num] * dx_images_exp[num]) +
                       torch.mean(dy_disparities_abs[num] * dy_images_exp[num])
                       for num in range(len(dx_disparities_abs))]

    return torch.stack(smoothness_loss, dim=0)


def left_right_disp_consistency_loss(disparity1, disparity2):

    dim = len(disparity1)
    predicted_disparity = disparity1
    projected_disparity = [generate_image(predicted_disparity[num], disparity2[num])
                           for num in range(dim)]

    left_right_disp_cons = [torch.abs(predicted_disparity[num] - projected_disparity[num])
                            for num in range(len(predicted_disparity))]
    left_right_disp_cons = [torch.mean(left_right_disp_cons[num])
                            for num in range(len(left_right_disp_cons))]
    return torch.stack(left_right_disp_cons, dim=0)
