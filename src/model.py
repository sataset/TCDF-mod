import torch
from torch import nn
from torch.nn.functional import softmax
# from torch.autograd import Variable


class ADDSTCN(nn.Module):
    def __init__(self, input_size, num_levels, kernel_size, dilation_c, cuda):
        super(ADDSTCN, self).__init__()
        
        # num_inputs, num_levels, kernel_size=2, dilation_c=2
        layers = []
        in_channels = input_size
        out_channels = input_size
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
        self.depthwise = nn.Sequential(*layers)

        self.pointwise = nn.Conv1d(input_size, 1, 1)  # comment

        self.fs_attention = torch.nn.Parameter(torch.ones(input_size, 1))
        # self._attention = torch.ones(input_size, 1)
        # self._attention = Variable(self._attention, requires_grad=False)
        # self.fs_attention = torch.nn.Parameter(self._attention.data)

        nn.init.normal_(self.pointwise.weight, mean=0.0, std=0.1)
        # nn.init.normal_(self.pointwise.weight)
        # nn.init.xavier_normal_(self.pointwise.weight)
        # nn.init.kaiming_normal_(self.pointwise.weight)

        if cuda:
            self.depthwise = self.depthwise.cuda()
            self.pointwise = self.pointwise.cuda()  # comment
            self.fs_attention = self.fs_attention.cuda()
            # self._attention = self._attention.cuda()

    def forward(self, x):
        out = self.depthwise(x * softmax(self.fs_attention, dim=0))
        out = self.pointwise(out)  # comment
        return out.transpose(1, 2)  # comment


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
        self.relu = nn.PReLU(n_inputs)
        
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.1)
        # nn.init.normal_(self.conv1.weight)
        # nn.init.xavier_normal_(self.conv1.weight)
        # nn.init.kaiming_normal_(self.conv1.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu(out)
        return out


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
       
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding,
                               dilation=dilation, groups=n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.relu = nn.PReLU(n_inputs)

        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.1)
        # nn.init.normal_(self.conv1.weight)
        # nn.init.xavier_normal_(self.conv1.weight)
        # nn.init.kaiming_normal_(self.conv1.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu(out + x)  # residual connection
        return out


class LastBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(LastBlock, self).__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, groups=n_outputs,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.linear = nn.Linear(n_inputs, n_inputs)
        
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.1)
        # nn.init.normal_(self.conv1.weight)
        # nn.init.xavier_normal_(self.conv1.weight)
        # nn.init.kaiming_normal_(self.conv1.weight)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)

        # original TCDF solution
        out = self.linear(out.transpose(1, 2) + x.transpose(1, 2)).transpose(1, 2)

        # test
        # out = self.linear((out + x).transpose(1, 2))
        
        # true pointwise conv
        # out = (out + x)

        return out
