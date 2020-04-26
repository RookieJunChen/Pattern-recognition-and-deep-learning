import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import MLP

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28 * 28    # MNIST数据集中的每张图片由28x28个像素点构成
hidden_size = 500   # 隐藏层大小
num_classes = 10    # 手写数字识别，总共有10个类别的数字
num_epochs = 40  # 将num_epochs提高到40
batch_size = 100    # 每一个batch的大小
learning_rate = 0.001   # 学习率

# 读取训练集
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 构造MLP模型
model = MLP.ThreelayerMLP.NeuralNet(input_size, hidden_size, num_classes).to(device)
# dropout训练，训练阶段开启随机采样，所有模型共享参数
model.train()


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 后向优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每一百步打印一次
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 保存训练完的模型
torch.save(model, 'threelayernd.ckpt')

