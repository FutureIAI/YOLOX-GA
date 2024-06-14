import torch
import torch.nn as nn
import random
from thop import profile
from thop import clever_format
def getRamdomInt(begin,end):
    a = []
    for i in range(begin,end):
        a.append(i)

    return random.sample(a,3)[2]


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


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)

        # self.conv2d = DEPTHWISECONV(in_ch=2,out_ch=1,k=7,s=1,p=3)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)

        out = self.sigmoid(self.conv2d(out))

        return out
class MulSapitialAttention(nn.Module):
    def __init__(self,in_channel):
        super(MulSapitialAttention, self).__init__()

        self.spitial_attention1 = SpatialAttentionModule()
        self.spitial_attention2 = SpatialAttentionModule()
        self.spitial_attention3 = SpatialAttentionModule()
        self.spitial_attention4 = SpatialAttentionModule()
        self.spitial_attention5 = SpatialAttentionModule()
        self.spitial_attention6 = SpatialAttentionModule()
        self.spitial_attention7 = SpatialAttentionModule()
        self.spitial_attention8 = SpatialAttentionModule()


        # 减少计算量
        self.conv = DEPTHWISECONV(in_channel,in_channel//2,k=4,s=2,p=1)
        self.norm = nn.Sequential(
            # nn.Conv2d(in_channel,in_channel//2,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(in_channel//2,affine=True),
            nn.LeakyReLU(0.2,inplace=True)
        )
        pass

    def forward(self,x):

        # 多分支空间注意力头
        sa1 = self.spitial_attention1(x)
        sa2 = self.spitial_attention2(x)
        sa3 = self.spitial_attention3(x)
        sa4 = self.spitial_attention4(x)
        sa5 = self.spitial_attention5(x)
        sa6 = self.spitial_attention6(x)
        sa7 = self.spitial_attention7(x)
        sa8 = self.spitial_attention8(x)

        # 随机设置某一通道的值为0
        sa_cat = torch.cat((sa1,sa2,sa3,sa4,sa5,sa6,sa7,sa8),dim=1)
        # print(sa_cat.shape)
        end = sa_cat.shape[1]
        randomInt = getRamdomInt(0,end)
        sa_cat[:,randomInt,:,:] = 0
        max_attention = torch.max(sa_cat,dim=1)

        # 将多分支的空间注意力图合并为最大空间注意力图
        max_attention = torch.unsqueeze(max_attention[0],dim=1)

        out = max_attention * x

        out = self.conv(out)
        # print(out.shape)
        out = self.norm(out)
        return out
        pass


    pass


