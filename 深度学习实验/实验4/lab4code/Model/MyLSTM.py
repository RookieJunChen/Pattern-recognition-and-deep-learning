from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
import math
from enum import IntEnum


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class LSTM_batchfirst(nn.Module):
    """
    自己构造的LSTM
    等价于nn.LSTM中batch_first=True的效果
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 输入门i_t
        self.W_i = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_i = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(torch.Tensor(hidden_size))
        # 遗忘门f_t
        self.W_f = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_f = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(torch.Tensor(hidden_size))
        # 候选内部状态g_t
        self.W_g = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_g = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(torch.Tensor(hidden_size))
        # 输出门o_t
        self.W_o = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_o = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(torch.Tensor(hidden_size))

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def _init_states(self, x):
        h_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=x.dtype).to(x.device)
        c_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=x.dtype).to(x.device)
        return h_t, c_t

    def forward(self, x, init_states=None):
        """
        在这里我定义x的输入格式是(batch, sequence, feature)
        """
        batch_size, seq_size, _ = x.size()
        hidden_seq = []

        # 状态初始化
        if init_states is None:
            h_t, c_t = self._init_states(x)
        else:
            h_t, c_t = init_states

        # 按时间顺序迭代
        for t in range(seq_size):
            x_t = x[:, t, :]
            # 更新门组件及内部候选状态（Tips:Pytorch中@用于矩阵相乘，*用于逐个元素相乘）
            i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)
            g_t = torch.tanh(x_t @ self.W_g + h_t @ self.U_g + self.b_g)
            o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)
            # 记忆单元和隐藏单元更新
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t)
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)

# def reset_weigths(model):
#     """reset weights
#     """
#     for weight in model.parameters():
#         nn.init.constant_(weight, 0.5)


# # test
# inputs = torch.ones(10, 1, 10).cuda()
# h0 = torch.ones(1, 10, 20).cuda()
# c0 = torch.ones(1, 10, 20).cuda()
# print(h0.shape, h0)
# print(c0.shape, c0)
# print(inputs.shape, inputs)
#
# # test naive_lstm with input_size=10, hidden_size=20
# naive_lstm = NaiveLSTM(10, 20).cuda()
# naive_lstm.double()
# reset_weigths(naive_lstm)
#
# output1, (hn1, cn1) = naive_lstm(inputs.double(), (h0.double(), c0.double()))
#
# print(hn1.shape, cn1.shape, output1.shape)
# print(hn1)
# print(cn1)
# print(output1)
#
# lstm = nn.LSTM(10, 20, batch_first=True).cuda()
# lstm.double()
# reset_weigths(lstm)
# output2, (hn2, cn2) = lstm(inputs.double(), (h0.double(), c0.double()))
# print(hn2.shape, cn2.shape, output2.shape)
# print(hn2)
# print(cn2)
# print(output2)
