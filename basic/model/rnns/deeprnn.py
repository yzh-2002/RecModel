import torch
from torch.nn import functional as F


class DeepRNN(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, layer_size):
        super(DeepRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        # 输入层参数（第一层隐状态输入为x，后续隐状态输入为上一层隐状态，其shape不一样，故需要单独设置一层输入层权重）
        self.W_xi = torch.nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.01)
        # 隐藏层参数
        self.W_xh = torch.nn.Parameter(torch.randn(layer_size - 1, hidden_size, hidden_size) * 0.01)
        self.W_hh = torch.nn.Parameter(torch.randn(layer_size, hidden_size, hidden_size) * 0.01)
        self.b_h = torch.nn.Parameter(torch.zeros(layer_size, hidden_size))
        # 输出层参数
        self.W_ho = torch.nn.Parameter(torch.randn(hidden_size, vocab_size) * 0.01)
        self.b_o = torch.nn.Parameter(torch.zeros(vocab_size))

    def init_state(self, batch_size):
        # TODO:直接创建一个shape(layer_size,batch_size, hidden_size)的张量 后续修改每层state时会报错：涉及梯度计算的变量被原地修改
        # 循环创建的可正常运行
        return [torch.zeros((batch_size, self.hidden_size)).to('cuda') for _ in range(self.layer_size)]

    def forward(self, x, state):
        """
        :param state: list [(batch_size, hidden_size)...] len=layer_size
        :param x: shape(sequence_length,batch_size, vocab_size)
        """
        outputs = []
        x = F.one_hot(x.T, self.vocab_size).type(torch.float32)
        for t in x:  # t为每个时间步的输入
            state[0] = torch.tanh(torch.mm(t, self.W_xi) + torch.mm(state[0], self.W_hh[0]) + self.b_h[0])
            for i in range(1, self.layer_size):
                state[i] = torch.tanh(
                    torch.mm(state[i - 1], self.W_xh[i - 1]) + torch.mm(state[i], self.W_hh[i]) + self.b_h[i])
            # 输出仅依赖于最后一层隐状态
            output = torch.mm(state[self.layer_size - 1], self.W_ho) + self.b_o
            outputs.append(output)
        # 返回形状等同于输入张量x
        return torch.cat(outputs, dim=0), state
