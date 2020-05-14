import torch
import argparse
import os
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from MyDataSet import TextDataset
from Model import RNNforTextsen


def get_args():
    """
    解析命令行参数
    返回参数列表
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', default=True, type=bool,
                        help='use gpu or cpu ?')
    parser.add_argument('--epochs', dest='epochs', default=1000, type=int,
                        help='number of epochs')
    parser.add_argument('--batch_size', dest='batchsize', default=256, type=int,
                        help='batch size')
    parser.add_argument('--lr', dest='lr', default=1e-3, type=float,
                        help='learning rate')
    args = parser.parse_args()
    return args


def test_accuracy(model, data_loader, criterion, device):
    """
    测试模型在数据j集上的正确率
    """
    # 开启测试模式
    model.eval()
    test_loss, correct = 0, 0
    for data, label in data_loader:
        # 拷贝数据到GPU
        data, label = data.to(device), label.to(device)
        # 前向传播
        outputs = model(data)
        # 计算loss
        test_loss += criterion(outputs, label).item()
        # 获得分类结果
        _, prediction = torch.max(outputs.data, 1)
        # 计算正确个数
        correct += prediction.eq(label.data).sum().item()
    test_loss /= len(data_loader)
    accuracy = 100. * correct / len(data_loader.dataset)

    print('\nTest Average Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), accuracy))
    return test_loss, accuracy


def train(epochs=1000, lr=0.005, device=torch.device('cuda'), log_interval=10):
    """
    用于训练模型的函数
    """

    writer = SummaryWriter()

    # 构建模型
    model = RNNforTextsen.RNN(50, 64, 2, bidirectional=True).to(device)

    # 设置Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 设置计时器动态调整learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5, last_epoch=-1)

    # 以交叉熵作为损失函数
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # 构造训练集与测试集
    train_dataset = TextDataset.TextDataset(device, train=True)
    test_dataset = TextDataset.TextDataset(device, train=False)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=200,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=20)

    # 训练
    for epoch in range(epochs):
        # 开启训练模式
        model.train()

        train_loss = 0.0
        for index, (data, label) in enumerate(train_loader):
            # 拷贝数据至GPU
            data, label = data.to(device), label.to(device)
            # 将梯度缓存置0
            optimizer.zero_grad()
            # 执行一次前向传播
            outputs = model(data)
            # 计算loss
            loss = criterion(outputs, label)
            # 反向传播
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if index % log_interval == 0:
                print('Train Epoch:[{}/{}]\tStep:[{}/{}]({:.0f}%)\tLoss: {:.6f}'.format(
                    epoch, epochs, index * len(data), len(train_loader.dataset), 100.0 * index / len(train_loader),
                    loss.data.item()))
        print('Train Epoch:[{}/{}]\tAverage Loss: {:.4f}'.format(
            epoch, epochs, train_loss / len(train_loader)))

        # 在测试集上测试
        test_loss, accuracy = test_accuracy(model, test_loader, criterion, device)
        # 绘图
        writer.add_scalars('Sen/Display', {'train_loss': train_loss, 'test_loss': test_loss, 'accuracy': accuracy},
                           epoch + 1, walltime=epoch + 1)
        # 计时器计步
        scheduler.step()
        writer.close()
    return model


if __name__ == '__main__':
    args = get_args()
    # 配置device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    # 训练
    model = train(epochs=args.epochs, lr=args.lr, device=device)

    # 保存模型
    save_dir = './checkpoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, 'basic.ckpt'))
