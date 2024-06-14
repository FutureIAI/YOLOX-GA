import torch
import torch.nn as nn
import random
from thop import profile
from thop import clever_format
from LA.MHSA import selfattention
class SCat(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(SCat, self).__init__()

        # self.attenA = selfattention(int(in_channel/2))
        # self.attenB = selfattention(int(in_channel/2))

        self.channel = int(in_channel/2)

        # 自注意力1
        self.query_1 = nn.Conv2d(self.channel, self.channel // 8, kernel_size=1, stride=1)
        self.key_1 = nn.Conv2d(self.channel, self.channel // 8, kernel_size=1, stride=1)
        self.value_1 = nn.Conv2d(self.channel, self.channel, kernel_size=1, stride=1)
        self.gamma_1 = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.



        # 自注意力2
        self.query_2 = nn.Conv2d(self.channel, self.channel // 8, kernel_size=1, stride=1)
        self.key_2 = nn.Conv2d(self.channel, self.channel // 8, kernel_size=1, stride=1)
        self.value_2 = nn.Conv2d(self.channel, self.channel, kernel_size=1, stride=1)
        self.gamma_2 = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.

        self.softmax = nn.Softmax(dim=-1)


        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2)
        )
        pass
    def forward(self,x):

        A,B = x.chunk(2,1)

        # 注意力1
        batch_size_1, channels_1, height_1, width_1 = A.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q_1 = self.query_1(A).view(batch_size_1, -1, height_1 * width_1).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k_1 = self.key_1(A).view(batch_size_1, -1, height_1 * width_1)
        # input: B, C, H, W -> v: B, C, H * W
        v_1 = self.value_1(A).view(batch_size_1, -1, height_1 * width_1)



        # 注意力2
        batch_size_2, channels_2, height_2, width_2 = B.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q_2 = self.query_2(B).view(batch_size_2, -1, height_2 * width_2).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k_2 = self.key_2(B).view(batch_size_2, -1, height_2 * width_2)
        # input: B, C, H, W -> v: B, C, H * W
        v_2 = self.value_2(B).view(batch_size_2, -1, height_2 * width_2)


        # 注意力1
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix_1 = torch.bmm(q_2, k_1)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix_1 = self.softmax(attn_matrix_1)  # 经过一个softmax进行缩放权重大小.
        out_1 = torch.bmm(v_1, attn_matrix_1.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out_1 = out_1.view(*A.shape)


        # 注意力2
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix_2 = torch.bmm(q_1, k_2)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix_2 = self.softmax(attn_matrix_2)  # 经过一个softmax进行缩放权重大小.
        out_2 = torch.bmm(v_2, attn_matrix_2.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out_2 = out_2.view(*B.shape)


        res = self.conv(torch.cat((out_1,out_2),dim=1))
        return res
        pass
    pass

