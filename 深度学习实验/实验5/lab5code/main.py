import argparse
import torch
from Model.Basic import Generator, Discriminator
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch import autograd
import os
import MyDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


def get_args():
    """
    解析命令行参数
    返回参数列表
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=400, help='batch size')
    parser.add_argument('--gos', type=int, default=2, help='the size of the output from generator')
    parser.add_argument('--noise_size', type=int, default=10, help='size of noise')
    parser.add_argument('--gms', type=int, default=1000)
    parser.add_argument('--dms', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=200, help='the number of epochs to train')
    parser.add_argument('--lrD', type=float, default=0.00018, help='learning rate for Discriminator')
    parser.add_argument('--lrG', type=float, default=0.00018, help='learning rate for Generator')
    parser.add_argument("--b1", type=float, default=0.5, help="Adam: decay of first order momentum of gradient")
    parser.add_argument('--gpu', dest='gpu', default=True, type=bool, help='use gpu or cpu')
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--model', type=str, default='wgan', help='Which model to use, default is gan')
    parser.add_argument('--results_dir', default=None, help='Where to store samples and models')
    parser.add_argument('--optim', type=str, default='rmsprop', help='Which optimizer to use, default is rmsprop')
    args = parser.parse_args()
    return args


def gen_label(batch_size):
    """
    用于生成真与假两个label
    """
    real = torch.ones((batch_size,)).view(-1, 1)
    fake = torch.zeros((batch_size,)).view(-1, 1)
    return real, fake


def visualization(gen_net, disc_net, real, epoch, save_path, device, model='gan'):
    """
    用于可视化的函数
    """
    # 关闭训练模式
    gen_net.eval()
    disc_net.eval()

    # 随机生成噪声
    noise = torch.randn(1000, opt.noise_size).to(device)

    # Generator生成数据
    fake = gen_net(noise)
    fake_np = fake.cpu().detach().numpy()

    # 画图需要用到的原始数据和生成数据的最大最小范围
    x_low, x_high = min(np.min(fake_np[:, 0]), -0.5), max(np.max(fake_np[:, 0]), 1.5)
    y_low, y_high = min(np.min(fake_np[:, 1]), 0), max(np.max(fake_np[:, 1]), 1)

    # 采样
    a_x = np.linspace(x_low, x_high, 200)
    a_y = np.linspace(y_low, y_high, 200)
    u = [[x, y] for y in a_y[::-1] for x in a_x[::-1]]
    u = np.array(u)
    u2tensor = torch.FloatTensor(u).cuda().to(device)

    # 判别器计算
    out = disc_net(u2tensor)
    out2np = out.cpu().detach().numpy()

    # 绘制判别器的结果(黑白热度图)，存储在设定的文件路径中
    plt.cla()
    plt.clf()
    plt.axis('off')
    disc_path = os.path.join(save_path, 'Discriminator')
    disc_path = os.path.join(disc_path, model + '_' + opt.optim)
    if not os.path.exists(disc_path):
        os.makedirs(disc_path)
    plt.imshow(out2np.reshape(200, 200), extent=[x_low, x_high, y_low, y_high], cmap='gray')
    plt.colorbar()
    plt.savefig(os.path.join(disc_path, 'epoch{}.png'.format(epoch)))

    # 绘制生成器的结果，存储在设定的文件路径中
    plt.cla()
    plt.clf()
    plt.axis('off')
    # if model == 'gan':
    #     c = ['w' if x >= 0.4999 else 'black' for x in out2np]
    #     plt.scatter(u[:, 0], u[:, 1],
    #                 c=c, alpha=0.3, marker='s')
    # else:
    #     plt.imshow(out2np.reshape(200, 200), extent=[x_low, x_high, y_low, y_high], cmap='gray')
    #     plt.colorbar()
    plt.imshow(out2np.reshape(200, 200), extent=[x_low, x_high, y_low, y_high], cmap='gray')
    plt.colorbar()
    plt.scatter(real[:, 0], real[:, 1], c='b')
    plt.scatter(fake_np[:, 0], fake_np[:, 1], c='r')
    g_path = os.path.join(save_path, 'Generator')
    g_path = os.path.join(g_path, model + '_' + opt.optim)
    if not os.path.exists(g_path):
        os.makedirs(g_path)
    plt.savefig(os.path.join(g_path, 'epoch{}.png'.format(epoch)))


def cal_gradient_penalty(disc_net, device, real, fake):
    """
    用于计算WGAN-GP引入的gradient penalty
    """
    # 系数alpha
    alpha = torch.rand(real.size(0), 1)
    alpha = alpha.expand(real.size())
    alpha = alpha.to(device)

    # 按公式计算x
    interpolates = alpha * real + ((1 - alpha) * fake)

    # 为得到梯度先计算y
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = disc_net(interpolates)

    # 计算梯度
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    # 利用梯度计算出gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


if __name__ == "__main__":
    opt = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() and opt.gpu else 'cpu')

    writer = SummaryWriter(log_dir="./mylog")

    if opt.results_dir is None:
        opt.results_dir = './piclogs'
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)

    # 构造后面需要使用的数据集有数据
    point_dataset = MyDataset.PointDataset(device)
    real_data = point_dataset.get_data()
    data_loader = DataLoader(dataset=point_dataset, batch_size=opt.batch_size, shuffle=True)

    # 根据命令行参数选择构建哪种模型
    if opt.model == 'wgan':
        gen_net = Generator(opt.noise_size, opt.gms, opt.gos).to(device)
        disc_net = Discriminator(opt.gos, opt.dms, wgan=True).to(device)
    elif opt.model == 'wgan-gp':
        gen_net = Generator(opt.noise_size, opt.gms, opt.gos).to(device)
        disc_net = Discriminator(opt.gos, opt.dms, wgan=True).to(device)
    else:
        gen_net = Generator(opt.noise_size, opt.gms, opt.gos).to(device)
        disc_net = Discriminator(opt.gos, opt.dms).to(device)

    # 选用BCELoss为损失函数（判断单个类别不需要softmax）
    criterion = nn.BCELoss().to(device)

    # 根据命令行参数选择优化器
    if opt.optim == 'adam':
        optimizer_D = optim.Adam(disc_net.parameters(), lr=opt.lrD, betas=(opt.b1, 0.999))
        optimizer_G = optim.Adam(gen_net.parameters(), lr=opt.lrG, betas=(opt.b1, 0.999))
    else:
        optimizer_D = optim.RMSprop(disc_net.parameters(), lr=opt.lrD)
        optimizer_G = optim.RMSprop(gen_net.parameters(), lr=opt.lrG)

    try:
        # 用maxmin算法进行训练
        for epoch in range(opt.epochs):
            # 开启训练模式
            gen_net.train()
            disc_net.train()
            # 用于累计一个epoch的loss
            G_epochloss = 0.0
            D_epochloss = 0.0
            for i, real_img in enumerate(data_loader, 0):

                batch_size = real_img.size(0)
                # 生成标签
                reallabel, fakelabel = gen_label(batch_size)
                # 生成随机噪声
                noise = torch.randn(batch_size, opt.noise_size).to(device)

                # 拷贝数据至GPU
                real_img = real_img.cuda().to(device)
                reallabel = reallabel.cuda().to(device)
                fakelabel = fakelabel.cuda().to(device)

                """
                首先要进行maxmin算法的maximize判别器Loss的部分
                """
                # WGAN需要将判别器的参数绝对值截断到不超过一个固定常数c
                if opt.model == 'wgan':
                    for p in disc_net.parameters():
                        p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
                disc_net.zero_grad()

                # 优化过程根据GAN、WGAN、WGAN-GP三种模型的不同而异。另外，为了能和之前求最小值的优化过程一致，这里我们选用损失值的相反数作为优化目标，即
                # maximize A <==> min -A
                if opt.model == 'wgan':
                    # WGAN相较于GAN，判别器最后一层去掉sigmoid函数，故直接求期望即可，不必使用损失函数
                    D_Loss_real = disc_net(real_img).mean()
                    fake = gen_net(noise)
                    D_Loss_fake = disc_net(fake).mean()
                    D_Loss = -(D_Loss_real - D_Loss_fake)
                    # 反向传播
                    D_Loss.backward()
                elif opt.model == 'wgan-gp':
                    # WGAN-GP此处与WGAN同
                    D_Loss_real = disc_net(real_img).mean()
                    fake = gen_net(noise)
                    D_Loss_fake = disc_net(fake).mean()
                    # WGAN-GP相较于WGAN引入了gradient penalty限制梯度
                    gradient_penalty = cal_gradient_penalty(disc_net, device, real_img.data, fake.data)
                    D_Loss = -(D_Loss_real - D_Loss_fake) + gradient_penalty * 0.1
                    # 反向传播
                    D_Loss.backward()
                else:
                    # 与上面两个不同的是，GAN的公式是maximize log(D(x)) + log(1 - D(G(z)))
                    D_Loss_real = criterion(disc_net(real_img), reallabel)
                    fake = gen_net(noise)
                    D_Loss_fake = criterion(disc_net(fake.detach()), fakelabel)
                    D_Loss = D_Loss_real + D_Loss_fake
                    # 反向传播
                    D_Loss.backward()
                D_epochloss += D_Loss.item()
                # 优化
                optimizer_D.step()

                """
                接着要进行maxmin算法的minimize生成器Loss的部分
                """
                # 将梯度缓存置0
                gen_net.zero_grad()
                # 生成放入generator中的噪声
                noise = torch.randn(batch_size, opt.noise_size).to(device)
                fake = gen_net(noise)
                # 分模型的细节与上述原理相同
                if opt.model == 'wgan':
                    G_Loss = -disc_net(fake).mean()
                    G_Loss.backward()
                elif opt.model == 'wgan-gp':
                    G_Loss = -disc_net(fake).mean()
                    G_Loss.backward()
                else:
                    G_Loss = criterion(disc_net(fake), reallabel)
                    G_Loss.backward()
                G_epochloss += G_Loss.item()
                optimizer_G.step()

            # 绘图
            writer.add_scalar(opt.model + '_' + opt.optim + '_' + 'train/G_loss', -G_epochloss / len(data_loader),
                              epoch + 1, walltime=epoch + 1)
            writer.add_scalar(opt.model + '_' + opt.optim + '_' + 'train/D_loss', -D_epochloss / len(data_loader),
                              epoch + 1, walltime=epoch + 1)

            print('Epoch:[%d/%d]\tLoss_D: %.8f \tLoss_G: %.8f'
                  % (epoch + 1, opt.epochs, -D_epochloss / len(data_loader), G_epochloss / len(data_loader)))

            # 每5轮可视化一次
            if (epoch + 1) % 5 == 0:
                visualization(gen_net, disc_net, real_data, epoch + 1, opt.results_dir, device, model=opt.model)
    # ctrl + C 可停止训练并保存
    except KeyboardInterrupt:
        writer.close()
        exit(0)
    writer.close()
