import os
import logging
from shutil import rmtree, copyfile
from tensorboardX import SummaryWriter
import torch


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

