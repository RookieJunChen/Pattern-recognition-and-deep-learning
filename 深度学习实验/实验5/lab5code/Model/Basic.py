import torch.nn as nn


class Generator(nn.Module):
    """
    个人构造的生成器模型
    """

    def __init__(self, noise_size, g_middle_size, g_output_size):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise_size, g_middle_size),
            nn.ReLU(True),
            nn.Linear(g_middle_size, g_middle_size),
            nn.ReLU(True),
            nn.Linear(g_middle_size, g_middle_size),
            nn.ReLU(True),
            nn.Linear(g_middle_size, g_output_size),
        )

    def forward(self, x):
        output = self.gen(x)
        return output


class Discriminator(nn.Module):
    """
    个人构造的判别器模型
    """

    def __init__(self, g_output_size, d_middle_size, wgan=False):
        super(Discriminator, self).__init__()
        self.wgan = wgan
        self.disc = nn.Sequential(
            nn.Linear(g_output_size, d_middle_size),
            nn.ReLU(True),
            nn.Linear(d_middle_size, d_middle_size),
            nn.ReLU(True),
            nn.Linear(d_middle_size, d_middle_size),
            nn.ReLU(True),
            nn.Linear(d_middle_size, 1),    # 最后输出一维，判断是或否
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.disc(x)
        # WGAN与GAN的主要区别在于WGAN去掉了最后一层的sigmoid函数
        if not self.wgan:
            output = self.sigmoid(output)
        return output
