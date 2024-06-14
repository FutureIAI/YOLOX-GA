import torch
import torch.nn as nn


from GAN.CBAM import CBAM
from thop import profile
from thop import clever_format



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
    装置深度可分离卷积
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


class DeepConvGenertor(nn.Module):
    def __init__(self,channels=256):
        super(DeepConvGenertor, self).__init__()

        self.cbam1 = CBAM(channels)

        # -------------------------------------#
        # 下采样阶段
        # -------------------------------------#
        self.downsample = DEPTHWISECONV(channels,64,k=4,s=2,p=1,norm=False)
        self.downsample_norm = nn.Sequential(
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2)
        )
        self.downsample_deepconv1 = DEPTHWISECONV(64,32,norm=True)
        # self.downsample_deepconv2 = DEPTHWISECONV(64,32,norm=True)

        self.cbam2 = CBAM(32)

        # -------------------------------------#
        # 上采样阶段
        # -------------------------------------#
        self.upsample_deepconv1 = DEPTHWISECONV(32,64,norm=True)
        self.upsample1 = TransDEPTHWISECONV(64,128,4,2,1)
        self.upsample1_norm = nn.Sequential(
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2)
        )

        self.upsample2 = TransDEPTHWISECONV(128, 64, 4, 2, 1)
        self.upsample2_norm = nn.Sequential(
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2)
        )

        self.upsample3 = TransDEPTHWISECONV(64, 32, 4, 2, 1)
        self.upsample3_norm = nn.Sequential(
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2)
        )
        self.cbam3 = CBAM(32)

        self.deepConv = DEPTHWISECONV(32,channels//2)
        self.tanh = nn.Tanh()

        pass

    def forward(self,x):

        x = self.cbam1(x)
        x = self.downsample(x)
        x = self.downsample_norm(x)
        x = self.downsample_deepconv1(x)
        # x = self.downsample_deepconv2(x)
        # print(x.shape)
        x = self.cbam2(x)
        # print(x.shape)
        x = self.upsample_deepconv1(x)
        x = self.upsample1(x)
        x = self.upsample1_norm(x)
        x = self.upsample2(x)
        x = self.upsample2_norm(x)
        x = self.upsample3(x)
        x = self.upsample3_norm(x)
        x = self.cbam3(x)
        x = self.deepConv(x)
        x = self.tanh(x)
        return x

    pass



class Generator(nn.Module):
    def __init__(self, channels=256):
        super(Generator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers
        def Conv(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 3, stride=1, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        # self.model = nn.Sequential(
        #     *downsample(channels, 64, normalize=False),
        #     *downsample(64, 64),
        #     *downsample(64, 128),
        #     *downsample(128, 256),
        #     *downsample(256, 512),
        #     nn.Conv2d(512, 4000, 1),
        #     *upsample(4000, 512),
        #     *upsample(512, 256),
        #     *upsample(256, 128),
        #     *upsample(128, 64),
        #     *upsample(64, 32),
        #     *upsample(32, 32),
        #     nn.Conv2d(32, int(channels/2), 3, 1, 1),
        #
        #     nn.Tanh()
        # )
        # self.model = nn.Sequential(
        #     *downsample(channels, 128, normalize=False),
        #     *downsample(128, 64),
        #     *downsample(64, 32),
        #     # *downsample(128, 256),
        #     # *downsample(256, 512),
        #     nn.Conv2d(32, 32, 1),
        #     *upsample(32, 64),
        #     *upsample(64, 128),
        #     *upsample(128, 256),
        #     *upsample(256, 256),
        #     # *upsample(64, 32),
        #     # *upsample(32, 32),
        #     nn.Conv2d(256, int(channels / 2), 3, 1, 1),
        #
        #     nn.Tanh()
        # )
        self.cbam1 = CBAM(channels)
        self.downsample = nn.Sequential(
            *downsample(channels, 128, normalize=False),

            *Conv(128,64),
            *Conv(64,32)
        )

        # self.deepconv1 =  DEPTHWISECONV(128,64)
        # self.deepconv2 = DEPTHWISECONV(64,32)

        self.cbam2 = CBAM(32)
        self.upsample = nn.Sequential(

            *Conv(32,64),
            *upsample(64, 128),
            *upsample(128, 256),
            *upsample(256, 256)
        )
        self.cbam3 = CBAM(256)
        self.conv = nn.Sequential(
            nn.Conv2d(256, int(channels / 2), 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.cbam1(x)
        x = self.downsample(x)
        # x = self.deepconv1(x)
        # x = self.deepconv2(x)

        x = self.cbam2(x)
        x = self.upsample(x)
        x = self.cbam3(x)
        x = self.conv(x)
        return x

class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(

            #  256*128*128 ---> 256*64*64
            nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2,inplace=True),
            #  256*64*64 ---> 256*32*32
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),


            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # # State (256x16x16)
            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(512, affine=True),
            # nn.LeakyReLU(0.2, inplace=True),
            #
            # # State (512x8x8)
            # nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(1024, affine=True),
            # nn.LeakyReLU(0.2, inplace=True)
        )
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )


    def forward(self, x):
        # x = self.main_module(x)
        # x = self.output(x)
        # return x.view(-1)
        # print(x.shape)
        # print(x.shape)
        x = self.main_module(x)
        return self.output(x)


    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     inputs = torch.ones(1, 128, 160, 160)
#     inputs = inputs.to(device)
#     model = Discriminator(128)
#     output = model(inputs)
#     print(output.shape)
#     # model = DCGAN_D(128,256,64)
#
#     # model = model.to(device)
#     # output = model(inputs)
#     # print(output.shape)
#     # print(output.mean().shape)
#     pass

if __name__ == '__main__':
    inputs = torch.randn(1, 64, 80, 80)
    # # inputs = inputs.cuda(0)
    G = DeepConvGenertor(64)
    output = G(inputs)
    print(output.shape)
    #
    flops, params = profile(G, inputs=(inputs,))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops:', flops)
    print('params:', params)
    # # output = G(inputs)
    # # print(output.shape)

    # real = torch.randn(1,64,40,40)
    # D = Discriminator(64)
    # output = D(real)
    # print(output.shape)

    pass

