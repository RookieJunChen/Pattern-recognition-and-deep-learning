import Alexnet
import MyCIFAR10
import os
import torch
import torch.nn as nn


# 用于展示测试集上测试准确率及Loss的函数
def validate(model, test_loader, device, showloss=False):
    # 关闭dropout开启测试模式
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # 测试准确度
    with torch.no_grad():
        correct = 0
        total = 0
        Loss = 0.0
        for images, labels in test_loader:
            # 若GPU可用，拷贝数据至GPU
            images = images.to(device)
            labels = labels.to(device)
            # 将图像输入Alexnet中并得到结果
            outputs = model(images)
            # 如果需要展示Loss，就计算并累加
            if showloss:
                loss = criterion(outputs, labels)
                Loss += loss
            # 获得概率最大的下标，即分类结果
            _, predicted = torch.max(outputs.data, 1)
            # 计算正确个数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # 如果需要展示Loss，则打印出Loss
        if showloss:
            print('Loss in test_loader: {:.4f}'.format(Loss / len(test_loader)))
        # 打印测试准确率
        print('Accuracy of the network on the {} test images: {} %'.format(len(images) * len(test_loader),
                                                                           100 * correct / total))


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 构造测试集
cifar10 = MyCIFAR10.MyCIFAR10('./data/cifar-10-batches-py', train=False)
test_loader = torch.utils.data.DataLoader(dataset=cifar10, batch_size=50, shuffle=False)

# 加载模型
model_dir = './checkpoints/final.ckpt'
model = Alexnet.Alexnet().to(device)
if not os.path.exists(model_dir):
    print('model not found')
else:
    print('load exist model')
    model.load_state_dict(torch.load(model_dir))

# 展示准确率
validate(model, test_loader, device)
