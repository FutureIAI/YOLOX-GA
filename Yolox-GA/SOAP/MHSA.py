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
class TransDEPTHWISECONV(nn.Module):
    '''
    转置深度可分离卷积
    '''
    def __init__(self,in_ch,out_ch, k = 3, s = 1, p = 1,norm = False):
        super(TransDEPTHWISECONV, self).__init__()
        self.depth_conv = nn.ConvTranspose2d(in_channels=in_ch,out_channels=in_ch,kernel_size=k,stride=s,padding=p,groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=1,stride=1,padding=0,groups=1)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_ch, 0.8),
            nn.LeakyReLU(0.2)
        )
        self.is_norm = norm
        pass
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        if self.norm:
            out = self.norm(out)
        return out
        pass

    pass
class selfattention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*input.shape)

        return self.gamma * out + input


class MulHeadSelfAttention(nn.Module):
    def __init__(self,in_channel):
        super(MulHeadSelfAttention, self).__init__()

        # print(in_channel)

        # 自注意力
        self.sa1 = selfattention(in_channel)
        # 反卷积，将尺寸大小提高一倍
        self.dconv = TransDEPTHWISECONV(in_ch=in_channel,out_ch=in_channel*2,k=4,s=2,p=1)
        self.norm = nn.Sequential(
            nn.InstanceNorm2d(in_channel*2,affine=True),
            nn.SiLU(inplace=False)
        )





        pass
    def forward(self,x):

        sa1 = self.sa1(x)
        out = self.dconv(sa1)
        out = self.norm(out)
        return out
        pass

    pass




