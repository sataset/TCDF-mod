import torch
from torch import nn
from torch.nn.functional import softmax


class ADDSTCN(nn.Module):
    """
    Two principles are satisfied:
    1. network produces an output of the same length as an input
    2. there are no leakage from the future into the past

    Those principles are fulfilled using these approaches:
    1. Fully-convolutional network (FCN)
    2. Causal convolution

    1. Using specific dilation and padding depending on the kernel size
        dilation = dilation_c ** level
    2. Using specific padding depending on the kernel size
        padding = (kernel_size - 1) * dilation
    """
    def __init__(self, in_channels, num_levels, kernel_size, dilation_c):
        super().__init__()
        
        out_channels = in_channels

        layers = [TemporalBlock(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                dilation=dilation_c ** level)
                  for level in range(0, num_levels)]
        self.network = nn.Sequential(*layers)

        self.pointwise = nn.Conv1d(out_channels, 1, 1)
        
        nn.init.normal_(self.pointwise.weight, mean=0.0, std=0.1)
        # nn.init.normal_(self.pointwise.weight)
        # nn.init.xavier_normal_(self.pointwise.weight)
        # nn.init.kaiming_normal_(self.pointwise.weight)

        self.fs_attention = torch.nn.Parameter(torch.ones(in_channels, 1))

    def forward(self, x):
        out = self.network(x * softmax(self.fs_attention, dim=0))
        out = self.pointwise(out)
        return out


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, residual=True):
        super().__init__()
        
        self.residual = residual
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              groups=in_channels,
                              kernel_size=kernel_size,
                              dilation=dilation)
        
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.1)
        # nn.init.normal_(self.conv.weight)
        # nn.init.xavier_normal_(self.conv.weight)
        # nn.init.kaiming_normal_(self.conv.weight)

        self.relu = nn.PReLU(out_channels)

    def forward(self, x):
        # left padding with zeros
        # (to fit rule of predicting using only past information)
        out = nn.functional.pad(x, (self.padding, 0))
        out = self.conv(out)
        if self.residual:
            out += x
        return self.relu(out)
