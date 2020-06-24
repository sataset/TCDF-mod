import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


# original TCDF solution
class ADDSTCN(nn.Module):
    def __init__(self, input_size, num_levels, kernel_size, dilation_c, cuda):
        super(ADDSTCN, self).__init__()

        self.dwn = DepthwiseNet(input_size, num_levels,
                                kernel_size=kernel_size, dilation_c=dilation_c)
        self.pointwise = nn.Conv1d(input_size, 1, 1)  # test comment

        self._attention = torch.ones(input_size, 1)
        self._attention = Variable(self._attention, requires_grad=False)

        self.fs_attention = torch.nn.Parameter(self._attention.data)

        if cuda:
            self.dwn = self.dwn.cuda()
            self.pointwise = self.pointwise.cuda()  # test comment
            self._attention = self._attention.cuda()

    def init_weights(self):
        self.pointwise.weight.data.normal_(0, 0.1)  # test comment

    def forward(self, x):
        y1 = self.dwn(x * F.softmax(self.fs_attention, dim=0))
        y1 = self.pointwise(y1)  # test comment
        return y1.transpose(1, 2)  # test comment


# conventional convolution
# class ADDSTCN(nn.Module):
#     def __init__(self, input_size, num_levels, kernel_size, cuda, dilation_c):
#         super(ADDSTCN, self).__init__()

#         self.dwn = DepthwiseNet(input_size, num_levels,
#                                 kernel_size=kernel_size, dilation_c=dilation_c)

#         self._attention = torch.ones(input_size, 1)
#         self._attention = Variable(self._attention, requires_grad=False)

#         self.fs_attention = torch.nn.Parameter(self._attention.data)

#         if cuda:
#             self.dwn = self.dwn.cuda()
#             self._attention = self._attention.cuda()

#     def init_weights(self):
#         pass

#     def forward(self, x):
#         y1 = self.dwn(x * F.softmax(self.fs_attention, dim=0))
#         return y1


# true pointwise implementation
# class ADDSTCN(nn.Module):
#     def __init__(self, input_size, num_levels, kernel_size, cuda, dilation_c):
#         super(ADDSTCN, self).__init__()

#         self.dwn = DepthwiseNet(input_size, num_levels,
#                                 kernel_size=kernel_size, dilation_c=dilation_c)
#         self.pointwise = nn.Conv1d(input_size, 1, 1)

#         self._attention = torch.ones(input_size, 1)
#         self._attention = Variable(self._attention, requires_grad=False)

#         self.fs_attention = torch.nn.Parameter(self._attention.data)

#         if cuda:
#             self.dwn = self.dwn.cuda()
#             self.pointwise = self.pointwise.cuda()
#             self._attention = self._attention.cuda()

#     def init_weights(self):
#         self.pointwise.weight.data.normal_(0, 0.1)

#     def forward(self, x):
#         y1 = self.dwn(x * F.softmax(self.fs_attention, dim=0))
#         y1 = self.pointwise(y1)
#         return y1.transpose(1, 2)

class Chomp1d(nn.Module):
    """PyTorch does not offer native support for causal convolutions,
    so it is implemented (with some inefficiency) by simply using
    a standard convolution with zero padding on both sides,
    and chopping off the end of the sequence."""
    
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class FirstBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(FirstBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding,
                               dilation=dilation, groups=n_outputs)

        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.relu = nn.PReLU(n_inputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net(x)
        return self.relu(out)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
       
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding,
                               dilation=dilation, groups=n_outputs)

        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.relu = nn.PReLU(n_inputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.1)
        

    def forward(self, x):
        out = self.net(x)
        return self.relu(out + x)  # residual connection


class LastBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(LastBlock, self).__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, groups=n_outputs,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.linear = nn.Linear(n_inputs, n_inputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)

        # original TCDF solution
        return self.linear(out.transpose(1, 2) + x.transpose(1, 2)).transpose(1, 2)

        # test
        # return self.linear((out + x).transpose(1, 2))
        
        # true pointwise conv
        # return (out + x)


class DepthwiseNet(nn.Module):
    def __init__(self, num_inputs, num_levels, kernel_size=2, dilation_c=2):
        super(DepthwiseNet, self).__init__()
        layers = []
        in_channels = num_inputs
        out_channels = num_inputs
        dilation_size_last = dilation_c**(num_levels - 1)

        layers += [FirstBlock(in_channels, out_channels, kernel_size,
                              stride=1, dilation=1,
                              padding=(kernel_size - 1))]
        for level in range(1, num_levels - 1):
            dilation_size = dilation_c ** level
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size)]
        layers += [LastBlock(in_channels, out_channels, kernel_size,
                             stride=1, dilation=dilation_size_last,
                             padding=(kernel_size - 1) * dilation_size_last)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
