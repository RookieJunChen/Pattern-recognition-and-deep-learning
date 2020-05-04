import torch
import torch.nn as nn
import torch.nn.functional as F


# Squeeze and Excitation Block Module
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)  # Squeeze
        w = self.fc(w)
        w = torch.sigmoid(w)

        return x * w  # Scale
