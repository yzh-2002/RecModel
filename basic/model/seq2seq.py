import torch
from ..abstract import Encoder, Decoder
from basic.model.transformer.attention import AdditiveAttention


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


class Seq2seqAttentionDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2seqAttentionDecoder, self).__init__(**kwargs)
        # 虽然query和key维度相等，但由于加性注意力有参数，一般效果要好于缩放点积注意力
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.rnn = torch.nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                                dropout=dropout)
        self.dense = torch.nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # enc_outputs: (output, state)
        # 先前只需要state，但注意力机制需要各个时间步的输出，故output也需要传出
        outputs, state = enc_outputs
        # 转置之后，outputs shape: ( batch_size, seq_len, num_hiddens)
        return outputs.permute(1, 0, 2), state, enc_valid_lens

    def forward(self, X, enc_outputs, state, enc_valid_lens):
        # forward函数的参数除X外，后续参数应该和init_state返回值保持一致
        outputs = []  # 保存所有时间步的输出
        X = self.embedding(X).permute(1, 0, 2)
        # 因为每一步的`query`会发生变化，所以此处需要循环，不能直接扔进dnn
        for x in X:  # x shape: (batch_size, embed_size)
            query = torch.unsqueeze(state[-1], dim=1)  # (batch_size, 1（query个数）, num_hiddens)
            # context shape: (batch_size, 1, num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # 此处拼接的不再是context是经注意力机制输出的状态，而不是Encoder的最终隐状态
            x = torch.cat((torch.unsqueeze(x, dim=1), context), dim=-1)
            output, state = self.rnn(x.permute(1, 0, 2), state)
            outputs.append(output)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), enc_outputs, state, enc_valid_lens
