import os
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import MyCIFAR10
from Net import Alexnet, VGG
import test
# import torch.utils.tensorboard


def get_args():
    """
    解析命令行参数
    返回参数列表
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', default=True, type=bool,
                        help='use gpu or cpu ?')
    parser.add_argument('--epochs', dest='epochs', default=10, type=int,
                        help='number of epochs')
    parser.add_argument('--batch_size', dest='batchsize', default=256, type=int,
                        help='batch size')
    parser.add_argument('--lr', dest='lr', default=1e-1, type=float,
                        help='learning rate')
    args = parser.parse_args()
    return args


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'w+')
    for i in range(len(data)):
        s = str(data[i]) + '\n'  # 每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


def train(epochs, batch_size, lr, log_interval=200):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
    # cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)
    #
    transform_train = transforms.Compose([
        # transforms.Resize(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(cifar_norm_mean, cifar_norm_std),
    ])

    transform_test = transforms.Compose([
        # transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize(cifar_norm_mean, cifar_norm_std),
    ])

    # 构造训练集
    cifar10 = MyCIFAR10.MyCIFAR10('./data/cifar-10-batches-py', device, train=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset=cifar10, batch_size=batch_size, shuffle=True)
    cifar10 = MyCIFAR10.MyCIFAR10('./data/cifar-10-batches-py', device, train=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(dataset=cifar10, batch_size=batch_size, shuffle=False)

    # 构建模型并开启dropout训练模式
    model = VGG.VGG().to(device)
    model.train()

    # 用交叉熵作为损失，Adam优化器作为优化器
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5, last_epoch=-1)

    trainlog = []
    testlog = []
    accuracylog = []

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
            trainlog.append((trainloss / len(train_loader)).item())
            testcost, accuracy = test.validate(model, test_loader, device, showloss=True)
            testlog.append(testcost.item())
            accuracylog.append(accuracy)
            model.train()
            # scheduler.step()
    # ctrl + C 可停止训练并保存
    except KeyboardInterrupt:
        print("Save.....")
        torch.save(model.state_dict(), os.path.join('./checkpoints', 'Interrupt.ckpt'))
        text_save('trainlogs.txt', trainlog)
        text_save('testlogs.txt', testlog)
        text_save('accuracylogs.txt', accuracylog)
        exit(0)
    text_save('trainlogs.txt', trainlog)
    text_save('testlogs.txt', testlog)
    text_save('accuracylogs.txt', accuracylog)
    return model


if __name__ == '__main__':
    args = get_args()

    model = train(epochs=args.epochs, batch_size=args.batchsize, lr=args.lr)

    # 保存模型
    save_dir = './checkpoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, 'basic.ckpt'))
