import torch
import torch.nn as nn
class Generator(nn.Module):
    '''
    宽和高翻倍，通道数不变
    '''
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

        self.model = nn.Sequential(
            *downsample(channels, 128, normalize=False),
            *downsample(128, 64),
            *downsample(64, 32),
            nn.Conv2d(32, 32, 1),
            *upsample(32, 64),
            *upsample(64, 128),
            *upsample(128, 256),
            *upsample(256, 256),
            nn.Conv2d(256, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()

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

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

        )


        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        # print(x.shape)
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


