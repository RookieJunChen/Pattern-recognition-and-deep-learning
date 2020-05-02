import torch.nn as nn
import torch


class VGG(nn.Module):
    """
    本人所构造的VGG
    1.为了适应32*32的图像，去掉了其中两个池化层
    2.为了提升效果，在每个卷积层后面加了BN层
    3.最后的全连接层输入结点数为512*4*4
    """

    def __init__(self, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        # 卷积层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 缩小卷积核，步长、填充
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(inplace=True),  # inplace=True，覆盖操作，节省空间
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256, affine=True),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
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

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)  # 进入全连接层之前展开为一维
        out = self.classifier(out)
        return out

    # 初始化参数
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
