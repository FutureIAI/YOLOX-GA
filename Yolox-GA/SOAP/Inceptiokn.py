import torch
import torch.nn.functional as F
from LA.DEPTHCONV import DEPTHWISECONV




class InceptionA(torch.nn.Module):
    def __init__(self, in_channels,out_channels):
        super(InceptionA, self).__init__()

        self.branch1x1 = DEPTHWISECONV(in_ch=in_channels,out_ch=16,k=1,s=1,p=0)


        self.branch5x5_1 = DEPTHWISECONV(in_ch=in_channels,out_ch=16,k=1,s=1,p=0)
        self.branch5x5_2 = DEPTHWISECONV(in_ch=16,out_ch=24,k=5,s=1,p=2)



        self.branch3x3_1 = DEPTHWISECONV(in_ch=in_channels,out_ch=16,k=1,s=1,p=0)
        self.branch3x3_2 = DEPTHWISECONV(in_ch=16,out_ch=24,k=3,p=1)
        self.branch3x3_3 = DEPTHWISECONV(in_ch=24,out_ch=24,k=3,p=1)


        self.branch_pool = DEPTHWISECONV(in_ch=in_channels,out_ch=24,k=1,p=0)

        self.cat_conv = DEPTHWISECONV(in_ch=88,out_ch=out_channels,k=1,s=1,p=0,norm=True)


    def forward(self, x):
        brach1x1 = self.branch1x1(x)

        brach5x5 = self.branch5x5_1(x)
        brach5x5 = self.branch5x5_2(brach5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        print(brach1x1.shape,brach5x5.shape,branch3x3.shape,branch_pool.shape)

        output = [brach1x1, brach5x5, branch3x3, branch_pool]
        output = torch.cat(output, dim=1)
        result = self.cat_conv(output)

        # return torch.cat(output, dim=1)
        return result
    pass


class InceptionB(torch.nn.Module):
    def __init__(self, in_channels,out_channels):
        super(InceptionB, self).__init__()

        self.branch1x1 = DEPTHWISECONV(in_ch=in_channels,out_ch=16,k=1,s=1,p=0)
        self.branch7x7 = DEPTHWISECONV(in_ch=16,out_ch=16,k=7,s=2,p=3)


        self.branch5x5_1 = DEPTHWISECONV(in_ch=in_channels,out_ch=24,k=1,s=1,p=0)



        self.branch3x3_1 = DEPTHWISECONV(in_ch=in_channels,out_ch=16,k=1,s=1,p=0)
        self.branch3x3_2 = DEPTHWISECONV(in_ch=16,out_ch=24,k=3,p=1)
        self.branch3x3_3 = DEPTHWISECONV(in_ch=24,out_ch=24,k=3,s=2,p=1)


        self.branch_pool = DEPTHWISECONV(in_ch=in_channels,out_ch=24,k=1,p=0)

        self.cat_conv = DEPTHWISECONV(in_ch=88,out_ch=out_channels,k=1,s=1,p=0,norm=True)


    def forward(self, x):
        brach1x1 = self.branch1x1(x)
        brach1x1 = self.branch7x7(brach1x1)

        brach5x5 = F.max_pool2d(x,kernel_size=3, stride=2, padding=1)
        brach5x5 = self.branch5x5_1(brach5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        output = [brach1x1, brach5x5, branch3x3, branch_pool]
        output = torch.cat(output, dim=1)
        result = self.cat_conv(output)

        return result
    pass

