import torch.nn as nn
import torch


class Alexnet(nn.Module):
    """
    本人所构造的Alexnet,参考了torchvision.models.alexnet
    1. 在Conv2d层中就卷积核大小、步长、填充长度进行了改变
    2. 在池化层中改变了卷积核大小，避免后续图片过小
    3. 删除了第一个池化层
    """

    def __init__(self, num_classes=10):
        super(Alexnet, self).__init__()
        # 卷积层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 缩小卷积核，步长、填充
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),  # inplace=True，覆盖操作，节省空间
            # nn.MaxPool2d(kernel_size=2),  # 32 -> 16
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 32 -> 16
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 16 -> 8
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # 8 -> 4
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)  # 进入全连接层之前展开为一维
        out = self.classifier(out)
        return out
