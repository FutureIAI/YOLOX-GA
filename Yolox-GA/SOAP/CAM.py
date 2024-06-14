import torch
import torch.nn as nn



class CAM_Module(nn.Module):
    def __init__(self,in_dim):
        super(CAM_Module,self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim,in_dim//2,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(in_dim // 2, affine=True),
            nn.SiLU(inplace=False)
        )

    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        out = self.conv(out)
        return out
    pass

