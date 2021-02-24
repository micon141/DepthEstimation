import torch
import torch.nn as nn
import torch.nn.functional as F


def get_padd(kernel_size):

    if isinstance(kernel_size, int):
        return kernel_size // 2
    elif isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
        return (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        raise ValueError(f"Not supported type for kernel_size: {type(kernel_size)}. Supported types: int, list, tuple")


class ResBlock2(nn.Module):

    def __init__(self, in_channels, out_channels, strides, kernel_size=None):
        super().__init__()
        if not kernel_size:
            kernel_size = [3, 3]

        self.conv1 = nn.Conv2d(in_channels[0], out_channels[0], kernel_size[0],
                               strides[0], padding=get_padd(kernel_size[0]))
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(in_channels[1], out_channels[1], kernel_size[1],
                               strides[1], padding=get_padd(kernel_size[1]))
        self.bn2 = nn.BatchNorm2d(out_channels[1])

        if in_channels[0] != out_channels[1]:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels[0], out_channels[1], 1, strides[0], bias=False),
                                        nn.BatchNorm2d(out_channels[1]))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResBlock3(nn.Module):

    def __init__(self, in_channels, out_channels, strides, kernel_size=None):
        super().__init__()
        if not kernel_size:
            kernel_size = [1, 3, 1]
        self.conv1 = nn.Conv2d(in_channels[0], out_channels[0], kernel_size[0],
                               strides[0], padding=get_padd(kernel_size[0]))
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(in_channels[1], out_channels[1], kernel_size[1],
                               strides[1], padding=get_padd(kernel_size[1]))
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(in_channels[2], out_channels[2], kernel_size[2],
                               strides[2], padding=get_padd(kernel_size[2]))
        self.bn3 = nn.BatchNorm2d(out_channels[2])

        if in_channels[0] != out_channels[2]:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels[0], out_channels[2], 1, strides[0], bias=False),
                                        nn.BatchNorm2d(out_channels[2]))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class UpsampleNN(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, scale, mode="nearest"):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale, mode=mode)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride, padding=get_padd(kernel_size))

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = F.elu(x)


class ResNet(nn.Module):

    def __init__(self):
        super().__init__()

    def get_block(self, in_channel, out_channels, stride, num_blocks):
        num_conv = len(out_channels)
        in_channels = [[in_channel, *out_channels[:-1]]] + [[out_channels[-1], *out_channels[:-1]]] * (num_blocks - 1)
        out_channels = [out_channels] * num_blocks
        strides = [[stride, *([1] * (num_conv - 1))]] + [[1] * num_conv] * (num_blocks - 1)
        layers = []
        print(in_channels)
        print(out_channels)
        print(strides)
        print("------------------------")
        for in_channel, out_channel, stride in zip(in_channels, out_channels, strides):
            if len(stride) == 2:
                layer = ResBlock2(in_channels=in_channel, out_channels=out_channel, strides=stride)
            else:
                layer = ResBlock3(in_channels=in_channel, out_channels=out_channel, strides=stride)
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.max_pool2d(x1, 3, 2)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x5 = F.avg_pool2d(x5, 1, 4)

        print(f"#######################\nshape: {x5.shape}")
        x6 = self.upconv7(x5)
        return x5


class ResNet50Encoder(ResNet):
    def __init__(self, input_shape, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 7, 2, padding=get_padd(7))
        self.bn1 = nn.BatchNorm2d(64)
        self.block2 = self.get_block(in_channel=64, out_channels=[64, 64, 256], stride=1, num_blocks=3)
        self.block3 = self.get_block(in_channel=256, out_channels=[128, 128, 512], stride=2, num_blocks=4)
        self.block4 = self.get_block(in_channel=512, out_channels=[256, 256, 1024], stride=2, num_blocks=6)
        self.block5 = self.get_block(in_channel=1024, out_channels=[512, 512, 2048], stride=2, num_blocks=3)

        ## Decoder
        downscaling_factor = 16
        dim = (input_shape[0] // downscaling_factor, input_shape[1] // downscaling_factor)
        #self.upconv7 = UpsampleNN(512, 512, 2, 3, dim * 2)


