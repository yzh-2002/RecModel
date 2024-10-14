import torch
from torch.nn import functional as F


class GRU(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(GRU, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # 更新门权重
        self.W_xr = torch.nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.01)
        self.W_hr = torch.nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_r = torch.nn.Parameter(torch.zeros(hidden_size))
        # 重置门权重
        self.W_xz = torch.nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.01)
        self.W_hz = torch.nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_z = torch.nn.Parameter(torch.zeros(hidden_size))
        # 候选隐状态权重
        self.W_xh = torch.nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.01)
        self.W_hh = torch.nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = torch.nn.Parameter(torch.zeros(hidden_size))
        # 输出层参数
        self.W_ho = torch.nn.Parameter(torch.randn(hidden_size, vocab_size) * 0.01)
        self.b_o = torch.nn.Parameter(torch.zeros(vocab_size))

    def init_state(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size)).to('cuda')

    def forward(self, x, state):
        """
        :param state: shape(batch_size, hidden_size)
        :param x: shape(sequence_length,batch_size, vocab_size)
        """
        outputs = []
        x = F.one_hot(x.T, self.vocab_size).type(torch.float32)
        for t in x:  # t为每个时间步的输入
            r = torch.sigmoid(torch.mm(t, self.W_xr) + torch.mm(state, self.W_hr) + self.b_r)
            z = torch.sigmoid(torch.mm(t, self.W_xz) + torch.mm(state, self.W_hz) + self.b_z)
            # * 逐元素相乘，也即Hadamard积
            candidate_state = torch.tanh(torch.mm(t, self.W_xh) + torch.mm(r * state, self.W_hh) + self.b_h)
            state = z * state + (1 - z) * candidate_state
            output = torch.mm(state, self.W_ho) + self.b_o
            outputs.append(output)
        # 返回形状等同于输入张量x
        return torch.cat(outputs, dim=0), state
