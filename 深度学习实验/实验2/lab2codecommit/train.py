import os
import torch
from optparse import OptionParser
import torch.nn as nn
from tensorboardX import SummaryWriter
import MyCIFAR10
import Alexnet


def get_args():
    """
    解析命令行参数
    返回参数列表
    """
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=20, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batchsize', default=50,
                      type='int', help='batch size')
    parser.add_option('-l', '--lr', dest='lr', default=3e-4,
                      type='float', help='learning rate')
    (options, args) = parser.parse_args()
    return options


def train(epochs, batch_size, lr, log_interval=200):
    writer = SummaryWriter()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构造训练集
    cifar10 = MyCIFAR10.MyCIFAR10('./data/cifar-10-batches-py', train=True)
    train_loader = torch.utils.data.DataLoader(dataset=cifar10, batch_size=batch_size, shuffle=True)

    # 构建模型并开启dropout训练模式
    model = Alexnet.Alexnet().to(device)
    model.train()

    # 用交叉熵作为损失，Adam优化器作为优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    try:
        total_step = len(train_loader)
        for epoch in range(epochs):
            trainloss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                # 若GPU可用，拷贝数据至GPU
                images = images.to(device)
                labels = labels.to(device)
                # 将梯度缓存置0
                optimizer.zero_grad()
                # 执行一次前向传播
                output = model(images)
                # 计算loss
                loss = criterion(output, labels)
                trainloss += loss
                # 反向传播
                loss.backward()
                # 更新权值
                optimizer.step()
                # 打印loss信息
                if (i + 1) % log_interval == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, epochs, i + 1, total_step, loss.item()))
            # 每个epoch计算一次平均Loss
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, trainloss / len(train_loader)))
            # write to tensorboard
            writer.add_scalar('scalar/TrainLoss', trainloss/len(train_loader), epoch, walltime=epoch)
        writer.close()
    # ctrl + C 可停止训练并保存
    except KeyboardInterrupt:
        print("Save.....")
        torch.save(model.state_dict(), os.path.join('./checkpoints', 'Interrupt.ckpt'))
        exit(0)
    return model


if __name__ == '__main__':
    args = get_args()

    model = train(epochs=args.epochs, batch_size=args.batchsize, lr=args.lr)

    # 保存模型
    save_dir = './checkpoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, 'basic.ckpt'))