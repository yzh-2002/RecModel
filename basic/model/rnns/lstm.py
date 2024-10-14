import torch
from torch.nn import functional as F


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # 输入门权重
        self.W_xi = torch.nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.01)
        self.W_hi = torch.nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_i = torch.nn.Parameter(torch.zeros(hidden_size))
        # 遗忘门权重
        self.W_xf = torch.nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.01)
        self.W_hf = torch.nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_f = torch.nn.Parameter(torch.zeros(hidden_size))
        # 输出门权重
        self.W_xo = torch.nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.01)
        self.W_ho = torch.nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_o = torch.nn.Parameter(torch.zeros(hidden_size))
        # 候选记忆单元权重
        self.W_xc = torch.nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.01)
        self.W_hc = torch.nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_c = torch.nn.Parameter(torch.zeros(hidden_size))
        # 输出层参数
        self.W_hq = torch.nn.Parameter(torch.randn(hidden_size, vocab_size) * 0.01)
        self.b_q = torch.nn.Parameter(torch.zeros(vocab_size))

    def init_state(self, batch_size):
        state = torch.zeros((batch_size, self.hidden_size)).to('cuda')
        cell = torch.zeros((batch_size, self.hidden_size)).to('cuda')
        return (state, cell)

    def forward(self, x, state, cell):
        """
        :param cell: shape(batch_size, hidden_size)
        :param state: shape(batch_size, hidden_size)
        :param x: shape(sequence_length,batch_size, vocab_size)
        """
        outputs = []
        x = F.one_hot(x.T, self.vocab_size).type(torch.float32)
        for t in x:  # t为每个时间步的输入
            i = torch.sigmoid(torch.mm(t, self.W_xi) + torch.mm(state, self.W_hi) + self.b_i)
            f = torch.sigmoid(torch.mm(t, self.W_xf) + torch.mm(state, self.W_hf) + self.b_f)
            o = torch.sigmoid(torch.mm(t, self.W_xo) + torch.mm(state, self.W_ho) + self.b_o)
            candidate_cell = torch.tanh(torch.mm(t, self.W_xc) + torch.mm(state, self.W_hc) + self.b_c)
            cell = f * cell + i * candidate_cell
            state = o * torch.tanh(cell)
            output = torch.mm(state, self.W_hq) + self.b_q
            outputs.append(output)
        # 返回形状等同于输入张量x
        return torch.cat(outputs, dim=0), state, cell
