import torch
from torch.nn import functional as F


class BidirectionalRNN(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(BidirectionalRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # 隐藏层参数
        # 正向隐状态参数
        self.W_xh_forward = torch.nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.01)
        self.W_hh_forward = torch.nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h_forward = torch.nn.Parameter(torch.zeros(hidden_size))
        # 反向隐状态参数
        self.W_xh_backward = torch.nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.01)
        self.W_hh_backward = torch.nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h_backward = torch.nn.Parameter(torch.zeros(hidden_size))

        # 输出层参数
        self.W_ho = torch.nn.Parameter(torch.randn(2 * hidden_size, vocab_size) * 0.01)
        self.b_o = torch.nn.Parameter(torch.zeros(vocab_size))

    def init_state(self, batch_size):
        # state[0]:前向隐状态 state[1]:后向隐状态
        return [torch.zeros((batch_size, self.hidden_size)).to('cuda'),
                torch.zeros((batch_size, self.hidden_size)).to('cuda')]

    def forward(self, x, state):
        """
        :param state: shape(batch_size, hidden_size)
        :param x: shape(sequence_length,batch_size, vocab_size)
        """
        forward_state_seq = []
        backward_state_seq = []
        x = F.one_hot(x.T, self.vocab_size).type(torch.float32)
        # 正向计算
        for t in x:
            state[0] = torch.tanh(
                torch.mm(t, self.W_xh_forward) + torch.mm(state[0], self.W_hh_forward) + self.b_h_forward)
            forward_state_seq.append(state[0])
        # 反向计算
        for t in torch.flip(x, [0]):
            state[1] = torch.tanh(
                torch.mm(t, self.W_xh_backward) + torch.mm(state[1], self.W_hh_backward) + self.b_h_backward)
            backward_state_seq.append(state[1])
        # 合并正向和反向隐状态 shape(sequence_length,batch_size, 2*hidden_size)
        final_state = [torch.cat((forward, backward), dim=1) for forward, backward in
                       zip(forward_state_seq, backward_state_seq[::-1])]
        outputs = [torch.mm(state, self.W_ho) + self.b_o for state in final_state]
        # 返回形状等同于输入张量x
        return torch.cat(outputs, dim=0), state
