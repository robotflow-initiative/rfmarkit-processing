"""
File:
    unet_ae.py
Author:
    厉宇桐
Date:
    2021-12-31
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DoubleConv(nn.Module):
    """This module performs convolution twice
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.op = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=(3,), padding=(1,)),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=(3,), padding=(1,)),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.op(x)


class DownSample(nn.Module):
    """This module performs down sampling
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.op(x)


class UpSample(nn.Module):
    """This module performs up sampling [linear|inv conv]
    """

    def __init__(self, in_channels: int, out_channels: int, mode: str = 'linear'):
        super().__init__()
        if mode == 'linear':  # Use linear upsample
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2))
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        """
        Args:
            x: feature input for up sampling
            x2: copied from previous

        Returns:

        """
        x = self.up(x)
        return self.conv(x)


class OutputConv(nn.Module):
    """This module performs output
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.op = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=(1,)))

    def forward(self, x):
        return self.op(x)


class UNetAutoEncoder(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, mode='linear'):
        """
        Args:
            in_channel: Channel of input, usually 3 for rgb images
            out_channel: Number of class to classifier, equals to the channel of output
            mode: Up sample mode, default to linear
        """
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mode = mode

        self.in_conv = DoubleConv(in_channel, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)

        self.up1 = UpSample(512, 256 // self.factor, self.mode)
        self.up2 = UpSample(256 // self.factor, 128 // self.factor, self.mode)
        self.up3 = UpSample(128 // self.factor, 64, self.mode)
        self.out_conv = OutputConv(64, out_channel)

    @property
    def factor(self) -> int:
        return 2 if self.mode == 'linear' else 1

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4)
        x = self.up2(x)
        x = self.up3(x)
        logits = self.out_conv(x)
        return logits


if __name__ == '__main__':
    BATCH_SIZE = 64
    dummy_input = torch.randn(BATCH_SIZE, 3, 200)
    Net = UNetAutoEncoder(in_channel=3, out_channel=3)
    dummy_output = Net(dummy_input)
    print(dummy_output.shape)

