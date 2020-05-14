import torch
from torch.utils.data import Dataset
import sine_wave_generation


class SineDataset(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        super(SineDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = None
        data = sine_wave_generation.generate()
        # 选取97个训练集，3个测试集
        if train:
            self.data = data[3:, :]
        else:
            self.data = data[:3, :]
        self.input = torch.from_numpy(self.data[:, :-1])  # 前999个数据作为input
        self.target = torch.from_numpy(self.data[:, 1:])  # 前999个input的后面一个数据作为target

    def __getitem__(self, index):
        return self.input[index].view(-1, 1), self.target[index].view(-1, 1)

    def __len__(self):
        return len(self.input)

# data = sine_wave_generation.generate()
# input = torch.from_numpy(data[:, 1:])
# print(input[2])
# print(input[2].view(-1, 1))
