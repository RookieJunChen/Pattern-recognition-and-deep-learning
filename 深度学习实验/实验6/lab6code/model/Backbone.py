import torch.nn as nn
import torch.nn.functional as F

from .SENet import SEBlock


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, is_SE=False):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel[0], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel[0], outchannel[0], kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel[0], outchannel[1], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel[1])
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel[1]:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel[1], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel[1])
            )

        self.se_block = None
        if is_SE:
            self.se_block = SEBlock(outchannel[1])


    def forward(self, x):
        out = self.left(x)
        if self.se_block:
            out = self.se_block(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Input : batches of 224x224x3 images
# Output : a Tensor of length 10 (our number of classes)
class ResNet(nn.Module):
    def __init__(self, name=101, is_backbone=False, ResidualBlock=ResidualBlock, num_classes=1000, is_SE=False):
        super(ResNet, self).__init__()
        layer3_num = 23 if name == 101 else 6
        self.is_backbone = is_backbone
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self.make_layer(ResidualBlock, [64, 256], 3, stride=1, is_SE=is_SE)
        self.layer2 = self.make_layer(ResidualBlock, [128, 512], 4, stride=2, is_SE=is_SE)
        self.layer3 = self.make_layer(ResidualBlock, [256, 1024], layer3_num, stride=2, is_SE=is_SE)
        self.layer4 = self.make_layer(ResidualBlock, [512, 2048], 3, stride=2, is_SE=is_SE)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def make_layer(self, block, channels, num_blocks, stride, is_SE=False):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, is_SE))
            self.inchannel = channels[1]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        d1 = self.layer1(out)
        d2 = self.layer2(d1)
        d3 = self.layer3(d2)
        d4 = self.layer4(d3)

        if self.is_backbone:
            return [d2, d3, d4]

        x = self.avgpool(d4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



if __name__ == '__main__':
    import torch
    net = ResNet(is_backbone=True)
    img = torch.ones(2,3,608,608)
    out = net(img)
    print(out[0].shape, out[1].shape, out[2].shape)
