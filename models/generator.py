import torch.nn as nn
import torch.nn.functional as F

from layers.blocks import ConvRelu
from layers.adain import adaptive_instance_normalization

class AdaINDecoder(nn.Module):
    def __init__(self, anatomy_out_channels):
        super(AdaINDecoder, self).__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.conv1 = ConvRelu(self.anatomy_out_channels, 128, 3, 1, 1)
        self.conv2 = ConvRelu(128, 64, 3, 1, 1)
        self.conv3 = ConvRelu(64, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 1, 3, 1, 1)

        nn.init.xavier_normal_(self.conv4.weight.data)
        self.conv4.bias.data.zero_()

    def forward(self, a, z):
        out = adaptive_instance_normalization(a, z)
        out = self.conv1(out)
        out = adaptive_instance_normalization(out, z)
        out = self.conv2(out)
        out = adaptive_instance_normalization(out, z)
        out = self.conv3(out)
        out = adaptive_instance_normalization(out, z)
        out = F.tanh(self.conv4(out))

        return out

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.decoder = AdaINDecoder(opt.anatomy_out_channels)

    def forward(self, a, z):
        reco = self.decoder(a, z)

        return reco