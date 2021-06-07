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


class MonoDepth(nn.Module):

    def __init__(self):
        super().__init__()


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


class Disp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=get_padd(kernel_size))
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return 0.3 * self.sigmoid(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.elu(x)
        return x


class UpsampleNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv_stride, scale, mode="bilinear"):
        super(UpsampleNN, self).__init__()
        self.scale = scale
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              conv_stride, padding=get_padd(kernel_size))
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)
        x = self.conv(x)
        x = self.batch_norm(x)
        return F.elu(x, inplace=True)


class ResNetDecoder(MonoDepth):

    def __init__(self, scale=(2, 2)):
        super().__init__()
        # Decoder
        self.upconv6 = UpsampleNN(in_channels=1024, out_channels=512, kernel_size=3, conv_stride=1, scale=scale)
        self.iconv6 = ConvLayer(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=get_padd(3))
        self.upconv5 = UpsampleNN(in_channels=512, out_channels=256, kernel_size=3, conv_stride=1, scale=scale)
        self.iconv5 = ConvLayer(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=get_padd(3))
        self.upconv4 = UpsampleNN(in_channels=256, out_channels=128, kernel_size=3, conv_stride=1, scale=scale)
        self.iconv4 = ConvLayer(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=get_padd(3))

        self.disp4 = Disp(in_channels=128, out_channels=2, kernel_size=3, stride=1)
        self.udisp4 = nn.Upsample(scale_factor=scale, mode="nearest")

        self.upconv3 = UpsampleNN(in_channels=128, out_channels=64, kernel_size=3, conv_stride=1, scale=scale)
        self.iconv3 = ConvLayer(in_channels=130, out_channels=64, kernel_size=3, stride=1, padding=get_padd(3))

        self.disp3 = Disp(in_channels=64, out_channels=2, kernel_size=3, stride=1)
        self.udisp3 = nn.Upsample(scale_factor=scale, mode="nearest")

        self.upconv2 = UpsampleNN(in_channels=64, out_channels=32, kernel_size=3, conv_stride=1, scale=scale)
        self.iconv2 = ConvLayer(in_channels=98, out_channels=32, kernel_size=3, stride=1, padding=get_padd(3))

        self.disp2 = Disp(in_channels=32, out_channels=2, kernel_size=3, stride=1)
        self.udisp2 = nn.Upsample(scale_factor=scale, mode="nearest")

        self.upconv1 = UpsampleNN(in_channels=32, out_channels=16, kernel_size=3, conv_stride=1, scale=scale)
        self.iconv1 = ConvLayer(in_channels=18, out_channels=16, kernel_size=3, stride=1, padding=get_padd(3))

        self.disp1 = Disp(in_channels=16, out_channels=2, kernel_size=3, stride=1)

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

        conv1b = F.relu(self.bn1(self.conv1(x)))
        conv2b = F.max_pool2d(conv1b, 2, 2)
        conv3b = self.block2(conv2b)
        conv4b = self.block3(conv3b)
        conv5b = self.block4(conv4b)
        conv6b = self.block5(conv5b)
        up6 = self.upconv6(conv6b)
        i6 = self.iconv6(torch.cat((conv5b, up6), dim=1))
        up5 = self.upconv5(i6)
        i5 = self.iconv5(torch.cat((conv4b, up5), dim=1))
        up4 = self.upconv4(i5)
        i4 = self.iconv4(torch.cat((conv3b, up4), dim=1))
        disp4 = self.disp4(i4)
        up3 = self.upconv3(i4)
        udisp4 = self.udisp4(disp4)
        i3 = self.iconv3(torch.cat((up3, conv2b, udisp4), dim=1))
        disp3 = self.disp3(i3)
        up2 = self.upconv2(i3)
        udisp3 = self.udisp3(disp3)
        i2 = self.iconv2(torch.cat((up2, conv1b, udisp3), dim=1))
        disp2 = self.disp2(i2)
        up1 = self.upconv1(i2)
        udisp2 = self.udisp2(disp2)
        i1_input = torch.cat((up1, udisp2), dim=1)
        i1 = self.iconv1(i1_input)
        disp1 =self.disp1(i1)

        return [disp1, disp2, disp3, disp4]


class ResNet34Encoder(ResNetDecoder):

    def __init__(self, input_shape, channels=3, scale=(2, 2)):
        super().__init__(scale=scale)
        self.conv1 = nn.Conv2d(channels, 64, 7, 2, padding=get_padd(7))
        self.bn1 = nn.BatchNorm2d(64)
        self.block2 = self.get_block(in_channel=64, out_channels=[64, 128], stride=2, num_blocks=3)
        self.block3 = self.get_block(in_channel=128, out_channels=[128, 256], stride=2, num_blocks=4)
        self.block4 = self.get_block(in_channel=256, out_channels=[256, 512], stride=2, num_blocks=6)
        self.block5 = self.get_block(in_channel=512, out_channels=[512, 1024], stride=2, num_blocks=4)


class ResNet50Encoder(ResNetDecoder):
    def __init__(self, input_shape, channels=3, scale=(2, 2)):
        super().__init__(scale=scale)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=7, stride=2, padding=get_padd(7))
        self.bn1 = nn.BatchNorm2d(64)
        self.block2 = self.get_block(in_channel=64, out_channels=[64, 64, 128], stride=2, num_blocks=3)
        self.block3 = self.get_block(in_channel=128, out_channels=[128, 128, 256], stride=2, num_blocks=4)
        self.block4 = self.get_block(in_channel=256, out_channels=[256, 256, 512], stride=2, num_blocks=6)
        self.block5 = self.get_block(in_channel=512, out_channels=[512, 512, 1024], stride=2, num_blocks=3)


class ResNet101Encoder(ResNetDecoder):
    def __init__(self, input_shape, channels=3, scale=(2, 2)):
        super().__init__(scale=scale)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=7, stride=2, padding=get_padd(7))
        self.bn1 = nn.BatchNorm2d(64)
        self.block2 = self.get_block(in_channel=64, out_channels=[64, 64, 128], stride=2, num_blocks=3)
        self.block3 = self.get_block(in_channel=128, out_channels=[128, 128, 256], stride=2, num_blocks=4)
        self.block4 = self.get_block(in_channel=256, out_channels=[256, 256, 512], stride=2, num_blocks=23)
        self.block5 = self.get_block(in_channel=512, out_channels=[512, 512, 1024], stride=2, num_blocks=3)


class ResNet152Encoder(ResNetDecoder):
    def __init__(self, input_shape, channels=3, scale=(2, 2)):
        super().__init__(scale=scale)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=7, stride=2, padding=get_padd(7))
        self.bn1 = nn.BatchNorm2d(64)
        self.block2 = self.get_block(in_channel=64, out_channels=[64, 64, 128], stride=2, num_blocks=3)
        self.block3 = self.get_block(in_channel=128, out_channels=[128, 128, 256], stride=2, num_blocks=8)
        self.block4 = self.get_block(in_channel=256, out_channels=[256, 256, 512], stride=2, num_blocks=36)
        self.block5 = self.get_block(in_channel=512, out_channels=[512, 512, 1024], stride=2, num_blocks=3)
