import torch
import torch.nn as nn

from layers.blocks import ConvBnLRelu

class LS_D(nn.Module):
    def __init__(self, opt):
        super(LS_D, self).__init__()
        self.ndf = opt.ndf

        self.conv1 = ConvBnLRelu(1, self.ndf, 4, 2, 1)
        self.conv2 = ConvBnLRelu(self.ndf, self.ndf * 2, 4, 2, 1)
        self.conv3 = ConvBnLRelu(self.ndf * 2, self.ndf * 4, 4, 2, 1)
        self.conv4 = ConvBnLRelu(self.ndf * 4, self.ndf * 8, 4, 2, 1)
        self.conv5 = ConvBnLRelu(self.ndf * 8, self.ndf * 16, 4, 2, 1)
        self.conv6 = ConvBnLRelu(self.ndf * 16, self.ndf * 16, 4, 2, 1)
        self.conv7 = ConvBnLRelu(self.ndf * 16, 1, 3, 1, 0)
        #1 x 1 x 1

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)

        return out.view(-1, 1)