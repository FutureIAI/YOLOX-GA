import torch
import torch.nn as nn
import torchvision
# from thop import profile

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out


class ResBlock_CBAM(nn.Module):
    def __init__(self,in_places, places, stride=1,downsampling=False, expansion = 4):
        super(ResBlock_CBAM,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )
        self.cbam = CBAM(channel=places*self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        # print(x.shape)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)


        out += residual
        out = self.relu(out)
        return out

class ResBlock_CBAM2(nn.Module):
    def __init__(self,in_places, out_places, stride=1,downsampling=False, expansion = 4):
        super(ResBlock_CBAM2,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=in_places//2,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(in_places//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_places//2, out_channels=in_places//2, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(in_places//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_places//2, out_channels=in_places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_places),
        )
        self.cbam = CBAM(channel=in_places)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(in_places)
            )
            pass

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=out_places, kernel_size=3, stride=1, padding=1,
                                  bias=False),
            nn.BatchNorm2d(out_places)
        )


        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        # print(x.shape)
        out = self.cbam(out)
        if self.downsampling:
            residual = self.downsample(x)


        out += residual

        out = self.conv(out)
        out = self.relu(out)
        return out

if __name__ == '__main__':
    inputs = torch.randn(1, 32, 100, 100)
    # self_attention = ResBlock_CBAM(32,64,stride=2,downsampling=True)
    self_attention = CBAM(32)
    flops, params = profile(self_attention, inputs=(inputs,))
    # print(output.shape)
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
    output = self_attention(inputs)
    print(output.shape)
    pass


# model = ResBlock_CBAM(in_places=16, places=4)
# print(model)
#
# input = torch.randn(1, 16, 64, 64)
# out = model(input)
# print(out.shape)
