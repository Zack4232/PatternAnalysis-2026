"""Implement a standard 2D U-Net baseline using PyTorch.

Reference:
Ronneberger et al. (2015), U-Net: Convolutional Networks for
Biomedical Image Segmentation.

This implementation uses padded convolutions and a configurable
base channel count. It is not the project's Improved U-Net model.
"""

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Apply two padded convolutions with ReLU activations."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    """Upsample features and combine them with an encoder skip."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(
            out_channels + skip_channels, out_channels
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError("Decoder and skip spatial shapes do not match.")
        return self.conv(torch.cat((skip, x), dim=1))


class UNet2D(nn.Module):
    """Predict per-pixel logits for inputs divisible by 16."""

    def __init__(self, in_channels=1, num_classes=6, base_channels=32):
        super().__init__()
        if min(in_channels, num_classes, base_channels) < 1:
            raise ValueError("Channel counts must be positive.")

        self.in_channels = in_channels
        b = base_channels

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)
        self.bottleneck = DoubleConv(b * 8, b * 16)

        self.dec4 = UpBlock(b * 16, b * 8, b * 8)
        self.dec3 = UpBlock(b * 8, b * 4, b * 4)
        self.dec2 = UpBlock(b * 4, b * 2, b * 2)
        self.dec1 = UpBlock(b * 2, b, b)
        self.output = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Expected input shape [B, in_channels, H, W].")
        height, width = x.shape[-2:]
        if min(height, width) < 16 or height % 16 or width % 16:
            raise ValueError("Input dimensions must be multiples of 16.")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        x = self.bottleneck(self.pool(e4))

        x = self.dec4(x, e4)
        x = self.dec3(x, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        return self.output(x)
