from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from MyDataSet import SineDataset
from Model import RNNforSine


# 用于绘图的函数
def draw(yi, color):
    # 已知用明线
    plt.plot(np.arange(data.size(1)), yi[:data.size(1)], color, linewidth=2.0)
    # 预测用虚线
    plt.plot(np.arange(data.size(1), data.size(1) + future), yi[data.size(1):], color + ':', linewidth=2.0)


# 设置writer
writer = SummaryWriter()

# 配置device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 生成训练/测试数据集
train_dataset = SineDataset.SineDataset(train=True)
test_dataset = SineDataset.SineDataset(train=False)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=97,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=3)

# 建立模型
model = RNNforSine.RNN(1, 51, 1).to(device)
model.double()

# 用MSELoss作为损失函数
criterion = nn.MSELoss().to(device)

# 使用Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.05)

for i in range(1000):
    for index, (data, label) in enumerate(train_loader):
        # 若GPU可用，拷贝数据至GPU
        data, label = data.to(device), label.to(device)
        # 将梯度缓存置0
        optimizer.zero_grad()
        # 执行一次前向传播
        out = model(data)
        # 计算loss并绘图
        loss = criterion(out, label)
        writer.add_scalar('sineadam/trainloss', loss.item(), i + 1, walltime=i + 1)
        print('Epoch[{}/{}], Loss = {}'.format(i + 1, 1000, loss.item()))
        # 反向传播
        loss.backward()
        optimizer.step()
    # 测试模型,并向前预测1000个点
    with torch.no_grad():  # 测试时不需要计算梯度，也不进行反向传播
        for index, (test_data, test_label) in enumerate(test_loader):
            # 若GPU可用，拷贝数据至GPU
            test_data, test_label = test_data.to(device), test_label.to(device)
            # 向前预测1000个点
            future = 1000
            prediction = model(test_data, future=future)
            # 用预测结果与真实值求损失
            loss = criterion(prediction[:, :-future], test_label)
            writer.add_scalar('sineadam/testloss', loss.item(), i + 1, walltime=i + 1)
            print('Epoch[{}/{}], Test_Loss= {}'.format(i + 1, 1000, loss.item()))
            # 记录预测值，在后续操作中用于绘图
            y = prediction.detach().cpu().numpy()
    # 50个epoch画一次图
    if (i + 1) % 50 != 0:
        continue
    # 绘制结果图
    plt.figure(figsize=(30, 10))
    plt.title('Prediction', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # 三个测试的数据分别用三种颜色来画
    draw(y[0], 'r')
    draw(y[1], 'g')
    draw(y[2], 'b')
    plt.savefig('./pic/sineadam' + str(int((i + 1) / 50)) + '.png')
    plt.show()

writer.close()
