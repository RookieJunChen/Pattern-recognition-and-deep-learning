import torch
import torch.nn as nn
from Model import MyLSTM


class RNN(nn.Module):
    """
    用于sin预测的RNN模型
    由一个自己编写的LSTM网络 + 全连接层构成
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = MyLSTM.LSTM_batchfirst(input_size, hidden_size)
        self.lstm.double()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, future=0):
        outputs = []
        # 初始化参数
        h_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=torch.double).to(x.device)
        c_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=torch.double).to(x.device)

        # 对于输入中的点，用真实值预测下一个时间点的值
        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            out, (h_t, c_t) = self.lstm(input_t, (h_t, c_t))
            output = self.linear(out)
            outputs += [output]

        # 预测时才会使用，对于预测值，使用上一个time_step预测的点作为输入，一直不断往后预测
        for i in range(future):
            out, (h_t, c_t) = self.lstm(output, (h_t, c_t))
            output = self.linear(out)
            out = self.dropout(out)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

