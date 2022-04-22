import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, stride=2, block_type='conv', skip_relu=False, kernel_size=3):
        super().__init__()
        if block_type == 'conv':
            self.conv = nn.Conv2d(in_filters, out_filters, stride=stride, padding=1, kernel_size=kernel_size)
        elif block_type == 'deconv':
            self.conv = nn.ConvTranspose2d(in_filters, out_filters, stride=stride, padding=1, kernel_size=kernel_size,
                                           output_padding=stride - 1)
        self.bn = nn.BatchNorm2d(out_filters)
        self.skip_relu = skip_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if not self.skip_relu:
            x = F.relu(x)
        return x
