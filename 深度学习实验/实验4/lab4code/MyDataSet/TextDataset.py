import torch
import numpy as np
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, device, train=True, transform=None, target_transform=None):
        super(TextDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = None
        self.device = device
        if train:
            self.data = np.load('data/glove_train.npy', allow_pickle=True)
        else:
            self.data = np.load('data/glove_test.npy', allow_pickle=True)

    def __getitem__(self, index):
        item, label = self.data[index]
        return torch.FloatTensor(item).cuda(self.device), label

    def __len__(self):
        return len(self.data)
