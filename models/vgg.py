import torch.nn as nn
import torch.nn.functional as F

from layers.blocks import ConvBnRelu

class VGG(nn.Module):
    def __init__(self, opt):
        super(VGG, self).__init__()
        self.ndf = 16
        self.num_classes = opt.num_classes

        self.drop = nn.Dropout(p=0.5)
        self.conv1 = ConvBnRelu(1, self.ndf, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = ConvBnRelu(self.ndf, self.ndf * 2, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBnRelu(self.ndf * 2, self.ndf * 8, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = ConvBnRelu(self.ndf * 8, self.ndf * 16, 3, 1, 1)
        self.conv4_1 = ConvBnRelu(self.ndf * 16, self.ndf * 16, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvBnRelu(self.ndf * 16, self.ndf * 16, 3, 1, 1)
        self.conv5_1 = ConvBnRelu(self.ndf * 16, self.ndf * 16, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(12544, 512) #7x7x512
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, self.num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.pool3(out)
        out = self.conv4(out)
        out = self.conv4_1(out)
        out = self.pool4(out)
        out = self.conv5(out)
        out = self.conv5_1(out)
        out = self.pool5(out)
        out = self.relu(self.linear1(out.view(-1, out.shape[1]*out.shape[2]*out.shape[3])))
        out = self.drop(out)
        out_fid = self.linear2(out)
        out = self.relu(out_fid)
        out = self.drop(out)
        out = self.linear3(out)
        out = F.log_softmax(out)

        return out, out_fid