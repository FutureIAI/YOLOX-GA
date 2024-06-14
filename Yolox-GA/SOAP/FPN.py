import torch
import torch.nn as nn

class SimpleBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super(SimpleBackbone, self).__init__()

        # 简单的骨干网络，这里使用几个卷积层代替
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class FPN(nn.Module):
    def __init__(self, in_channels=3):
        super(FPN, self).__init__()

        # 创建简单的骨干网络
        self.backbone = SimpleBackbone(in_channels)

        # 降采样和卷积处理特征金字塔
        self.pyramid = nn.ModuleList([
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.Conv2d(64, 256, kernel_size=1)
        ])

        # 上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # 获取底层特征
        features = self.backbone(x)

        # 构建特征金字塔
        pyramid_features = [self.pyramid[0](features)]
        for i in range(1, len(features)):
            pyramid_features.insert(0, self.pyramid[i](self.upsample(pyramid_features[0]) + features[-1 - i]))

        return pyramid_features

# 使用FPN
input_tensor = torch.randn((1, 32, 224, 224))  # 输入张量，通道数为6，图像大小为224x224
fpn = FPN(in_channels=32)
output_features = fpn(input_tensor)

# 输出特征金字塔的形状
for i, feature in enumerate(output_features):
    print(f"Level {i + 2} feature shape: {feature.shape}")
