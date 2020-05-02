import torch.nn as nn

from Net.SENet import SEBlock
from Net.VGG import VGG


class VGG_SE(VGG):
    """
    基于VGG构造VGG_SE
    """

    def __init__(self, num_classes=10, init_weights=True):
        super(VGG_SE, self).__init__(num_classes=num_classes, init_weights=init_weights)
        # 卷积层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 缩小卷积核，步长、填充
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            SEBlock(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),  # inplace=True，覆盖操作，节省空间
            SEBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            SEBlock(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            SEBlock(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            SEBlock(256),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            SEBlock(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            SEBlock(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            SEBlock(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            SEBlock(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            SEBlock(512),
            nn.MaxPool2d(kernel_size=2),
        )
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()
