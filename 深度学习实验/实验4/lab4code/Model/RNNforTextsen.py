import torch.nn as nn
from Model import MyLSTM


class RNN(nn.Module):
    """
    用于文本情感分析的RNN模型
    由一个自己编写的LSTM网络 + 全连接层构成
    """

    def __init__(self, input_size, hidden_size, output_size, bidirectional=False):
        super(RNN, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        if self.bidirectional:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        else:
            self.lstm = MyLSTM.LSTM_batchfirst(input_size, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, output_size) if not self.bidirectional else \
            nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        out, hidden_state = self.lstm(x)
        out = self.dropout(out)
        # 取最后一个time_step的输出作为输出
        out = self.linear(out[:, -1, :])
        return out
