import torch
import torch.nn as nn

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.activation = nn.ReLU(inplace=True)

        nn.init.xavier_normal_(self.conv.weight.data)
        self.conv.bias.data.zero_()
    
    def forward(self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv(x)
        return self.activation(x)


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.activation = nn.ReLU(inplace=True)

        nn.init.xavier_normal_(self.conv.weight.data)
        self.conv.bias.data.zero_()
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
    
    def forward(self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)


class ConvBnLRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, lrelu_w=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lrelu_w = lrelu_w
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.activation = nn.LeakyReLU(self.lrelu_w, inplace=True)

        nn.init.xavier_normal_(self.conv.weight.data)
        self.conv.bias.data.zero_()
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
    
    def forward(self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)

# class ConvPreactivationRelu(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
#         self.activation = nn.ReLU(inplace=False)
#         self.bn = nn.BatchNorm2d(self.out_channels)

#         nn.init.xavier_normal_(self.conv.weight.data)
#         self.conv.bias.data.zero_()
#         self.bn.weight.data.fill_(1)
#         self.bn.bias.data.zero_()

#     def forward(self,
#         x: torch.Tensor
#     ) -> torch.Tensor:
#         x = self.activation(x)
#         x = self.conv(x)
#         return self.bn(x)

# class ResConv(nn.Module):
#     def __init__(self, ndf, norm):
#         super(ResConv, self).__init__()
#         """
#         Args:
#             ndf: constant number from channels
#         """
#         self.ndf = ndf
#         self.norm = norm
#         self.conv1 = ConvPreactivationRelu(self.ndf, self.ndf * 2, 3, 1, 1, self.norm)
#         self.conv2 = ConvPreactivationRelu(self.ndf * 2 , self.ndf * 2, 3, 1, 1, self.norm)
#         self.resconv = ConvPreactivationRelu(self.ndf , self.ndf * 2, 1, 1, 0, self.norm)

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#         residual = self.resconv(residual)

#         return out + residual


# class Interpolate(nn.Module):
#     def __init__(self, size, mode):
#         super(Interpolate, self).__init__()
#         """
#         Args:
#             size: expected size after interpolation
#             mode: interpolation type (e.g. bilinear, nearest)
#         """
#         self.interp = nn.functional.interpolate
#         self.size = size
#         self.mode = mode
        
#     def forward(self, x):
#         out = self.interp(x, size=self.size, mode=self.mode) #, align_corners=False
        
#         return out