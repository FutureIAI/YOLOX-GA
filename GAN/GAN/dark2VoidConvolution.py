import torch
import torch.nn as nn

import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from thop import profile

class dark2VoidConv(nn.Module):
    def __init__(self,channels):
        super(dark2VoidConv, self).__init__()
        self.ConvDilaNet = nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=channels//2,kernel_size=1,stride=1,padding=0,dilation=2),
            nn.BatchNorm2d(channels//2,0.8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=channels//2, out_channels=channels // 2, kernel_size=1, stride=1, padding=0, dilation=2),
            nn.BatchNorm2d(channels // 2, 0.8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=channels//2, out_channels=channels // 2, kernel_size=1, stride=1, padding=0, dilation=2),
            nn.BatchNorm2d(channels // 2, 0.8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=channels // 2, out_channels=channels, kernel_size=1, stride=1, padding=0,dilation=2),
            nn.BatchNorm2d(channels, 0.8),
            nn.LeakyReLU(0.2)
        )
        pass
    def forward(self,inputs):
        return self.ConvDilaNet(inputs)
        pass

    pass

if __name__ == '__main__':
    inputs = torch.randn(1, 128, 160, 160)
    G = dark2VoidConv(128)
    flops, params = profile(G, inputs=(inputs,))
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
    output = G(inputs)
    print(output.shape)