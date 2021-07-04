import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import AdaINDecoder
from layers import ConvRelu

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise):
        return x + self.weight * noise

class NoiseNet(nn.Module):
    def __init__(self, anatomy_out_channels):
        super(NoiseNet, self).__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.conv1 = ConvRelu(self.anatomy_out_channels, 128, 3, 1, 1)
        self.conv2 = ConvRelu(128, 128, 3, 1, 1)
        self.conv3 = ConvRelu(128, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, self.anatomy_out_channels, 3, 1, 1)
        self.noise1 = NoiseInjection(128)
        self.noise2 = NoiseInjection(128)
        self.noise3 = NoiseInjection(128)
        
        nn.init.xavier_normal_(self.conv4.weight.data)
        self.conv4.bias.data.zero_()

    def forward(self, a, noise):
        out = self.conv1(a)
        out = self.noise1(out, noise)
        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.conv3(out)
        out = self.noise3(out, noise)
        out = self.conv4(out)
        out = F.gumbel_softmax(out,hard=True,dim=1)

        return out


class NIGenerator(nn.Module):
    def __init__(self, opt):
        super(NIGenerator, self).__init__()
        self.noise_injection = NoiseNet(opt.anatomy_out_channels)
        self.decoder = AdaINDecoder(opt.anatomy_out_channels)

    def forward(self, a, z, noise):
        noisy_a = self.noise_injection(a, noise)
        reco = self.decoder(noisy_a, z)

        return reco, noisy_a