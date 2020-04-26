import numpy as np
import os
import os.path
import pickle
import torch
from torch.utils.data import Dataset


class MyCIFAR10(Dataset):
    """
    根据CIFAR-10定义的个人数据集类
    继承自Dataset类，因此能够被torch.utils.data.DataLoader使用，从而更高效地在训练和测试中迭代
    """

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(MyCIFAR10, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = None
        self.labels = []

        # 根据CIFAR-10官网上下载的数据，训练集分为5个batch文件，每个里有10000张32*32的图片；测试集只有1个batch文件，里面有10000张32*32的图片
        train_lists = ['data_batch_1',
                       'data_batch_2',
                       'data_batch_3',
                       'data_batch_4',
                       'data_batch_5']
        test_lists = ['test_batch']

        # 根据train是否为True来选择测试集或训练集
        if train:
            lists = train_lists
        else:
            lists = test_lists

        # 读取数据集，构造类中的图像集和标签
        for list in lists:
            filename = os.path.join(root, list)
            with open(filename, 'rb') as f:  # 这里需要'rb' + 'latin1'才能读取
                datadict = pickle.load(f, encoding='latin1')
                X = datadict['data'].reshape(-1, 3, 32, 32)
                Y = datadict['labels']
                if self.imgs is None:
                    self.imgs = np.vstack(X).reshape(-1, 3, 32, 32)
                else:
                    self.imgs = np.vstack((self.imgs, X)).reshape(-1, 3, 32, 32)
                self.labels = self.labels + Y
        self.imgs = torch.from_numpy(self.imgs).type(torch.FloatTensor)

    # 继承的Dataset类需要实现两个方法之一：__getitem__(self, index)
    def __getitem__(self, index):
        img, label = self.imgs[index], self.labels[index]

        # img = Image.fromarray(img)
        # img = torch.from_numpy(img).type(torch.FloatTensor)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    # 继承的Dataset类需要实现两个方法之一：__len__(self)
    def __len__(self):
        return len(self.imgs)


