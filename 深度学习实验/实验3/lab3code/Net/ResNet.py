import torch.nn as nn
import torch.nn.functional as F

from Net.SENet import SEBlock


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, is_SE=False):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

        self.se_block = None
        if is_SE:
            self.se_block = SEBlock(outchannel)

    def forward(self, x):
        out = self.left(x)
        if self.se_block:
            out = self.se_block(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Input : batches of 32x32x3 images
# Output : a Tensor of length 10 (our number of classes)
class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10, is_SE=False):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1, is_SE=is_SE)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2, is_SE=is_SE)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2, is_SE=is_SE)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2, is_SE=is_SE)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride, is_SE=False):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, is_SE))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)


def ResNet18_SE():
    return ResNet(ResidualBlock, is_SE=True)
