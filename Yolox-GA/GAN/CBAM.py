import torch
import torch.nn as nn
import torchvision
# from thop import profile



class DEPTHWISECONV(nn.Module):
    '''
    深度可分离卷积
    '''
    def __init__(self,in_ch,out_ch, k = 3, s = 1, p = 1,norm = False):
        super(DEPTHWISECONV, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=k,
                                    stride=s,
                                    padding=p,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_ch, 0.8),
            nn.LeakyReLU(0.2)
        )
        self.is_norm = norm
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        if self.norm:
            out = self.norm(out)
        return out

    def __iter__(self):
        pass
    def __next__(self):
        pass



class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()

        # self.deepconv = DEPTHWISECONV(channel, channel, k=3, s=2, p=1, norm=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # x = self.deepconv(x)

        avgout = self.shared_MLP(self.avg_pool(x))
        # print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.conv2d = DEPTHWISECONV(2, 1, k=3, s=1, p=1)
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

