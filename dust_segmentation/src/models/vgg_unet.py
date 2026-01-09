from typing import Tuple

import torch
from torch import nn
from torchvision.models import VGG19_Weights, vgg19


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class VGG19UNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.encoder = backbone.features

        self.decoder1 = DecoderBlock(512, 512, 512)
        self.decoder2 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder4 = DecoderBlock(128, 64, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = skip2 = skip3 = skip4 = None
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if idx == 3:
                skip1 = x
            elif idx == 8:
                skip2 = x
            elif idx == 17:
                skip3 = x
            elif idx == 26:
                skip4 = x
        bridge = x

        if skip4 is None or skip3 is None or skip2 is None or skip1 is None:
            raise RuntimeError("VGG19 encoder did not produce expected skip connections.")

        x = self.decoder1(bridge, skip4)
        x = self.decoder2(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder4(x, skip1)
        x = self.final_conv(x)
        return self.activation(x)


def build_vgg19_unet(input_shape: Tuple[int, int, int], num_classes: int) -> nn.Module:
    return VGG19UNet(num_classes)
