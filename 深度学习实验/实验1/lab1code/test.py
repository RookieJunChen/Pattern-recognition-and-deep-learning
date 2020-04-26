import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取并构造测试集
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# 测试三层MLP
model = torch.load('threelayer.ckpt')
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of three-layer-network on the 10000 test images: {} %'.format(100 * correct / total))

# 测试四层MLP
model = torch.load('fourlayer.ckpt')
model.eval()    # 去掉dropout用于正常测试
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of four-layer-network on the 10000 test images: {} %'.format(100 * correct / total))

# 测试五层MLP
model = torch.load('fivelayer.ckpt')
model.eval()    # 去掉dropout用于正常测试
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of five-layer-network on the 10000 test images: {} %'.format(100 * correct / total))