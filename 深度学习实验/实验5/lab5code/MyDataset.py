from torch.utils.data import Dataset
import scipy.io as sciio
import numpy as np
import torch
import os


class PointDataset(Dataset):
    """
    自己构建的数据集类
    继承自Dataset类，因此能够被torch.utils.data.DataLoader使用，从而更高效地在训练和测试中迭代
    """

    def __init__(self, device, transform=None, target_transform=None):
        super(PointDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = None
        self.device = device
        # points.mat放置于根目录下
        path = './points.mat'
        if not os.path.exists(path):
            print('Data does not exist!')
            exit()
        # 读取数据
        points = sciio.loadmat(path)
        # 转化为Tensor
        self.data = np.vstack((points['xx'])).astype('float32')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).to(self.device)

    def __len__(self):
        return len(self.data)

    def get_data(self):
        return self.data


