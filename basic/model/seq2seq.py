import torch
from ..abstract import Encoder, Decoder


class Seq2seqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2seqEncoder, self).__init__(**kwargs)
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        # encoder可以采用双向RNN
        self.rnn = torch.nn.GRU(embed_size, num_hiddens, num_layers,
                                dropout=dropout, bidirectional=False)

    def forward(self, X, *args):
        # X shape: (batch_size, seq_len)
        X = self.embedding(X)  # X shape: (batch_size, seq_len, embed_size)
        X = X.permute(1, 0, 2)  # RNN中为便于访问每个时间步，需将seq_len放在第一维
        output, state = self.rnn(X)
        # state shape: ( (2)*num_layers, batch_size, 2 * num_hiddens) 最后时间步各层的隐状态组成的张量
        # output：不同于先前手写的RNN，nn.GRU并没有输出层，此处output为所有时间步的最后一层隐状态组成的张量 (seq_len, batch_size, (2) * num_hiddens)
        return output, state


class Seq2seqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2seqDecoder, self).__init__(**kwargs)
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        # decoder的输入包含了encoder最后一层的隐藏状态
        # 如果encoder采用双向RNN，则decoder的输入维度为embed_size + 2 * num_hiddens
        self.rnn = torch.nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                                dropout=dropout)
        self.dense = torch.nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # enc_outputs: (output, state)
        return enc_outputs[1]

    def forward(self, X, enc_state, state=None):
        if state is None:
            state = enc_state
        # X shape: (seq_len, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # state 取最后一个隐藏层的状态
        # repeat第一个维度复制为X.shape[0]，其余维度保持不变
        # TODO: 此处的state是只取Encoder输出的隐状态还是取后续Decoder中的隐状态？
        context = enc_state[-1].repeat(X.shape[0], 1, 1)  # ( seq_len, batch_size, 2 * num_hiddens)
        X_and_context = torch.cat((X, context), 2)  # (seq_len, batch_size, embed_size + 2 * num_hiddens)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)  # (batch_size, seq_len, vocab_size)
        return output, enc_state, state
