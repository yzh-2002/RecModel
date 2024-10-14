import torch
from basic.loss import masked_softmax


class AdditiveAttention(torch.nn.Module):
    def __init__(self, key_size, query_size, hidden_size, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = torch.nn.Linear(key_size, hidden_size, bias=False)
        self.W_q = torch.nn.Linear(query_size, hidden_size, bias=False)
        self.W_v = torch.nn.Linear(hidden_size, 1, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, queries, keys, values, valid_lens):
        # valid_lens:也即对于query，只考虑前valid_lens个key-value对，剩余的是padding出来的
        # `queries`: (`batch_size`, query个数, `query_size`)
        # 经线性层之后：(`batch_size`, query个数, `hidden_size`)，key同理
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 两个张量由于第二个维度不同，无法直接相加，故先扩展维度，再相加（广播机制）
        # features: (`batch_size`, query个数, key个数, `hidden_size`)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # scores: (`batch_size`, query个数, key个数)
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values: (`batch_size`, key个数, value的维度)，key与value要一一对应，故个数相等
        # 返回值: (`batch_size`, query个数, value的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(torch.nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, queries, keys, values, valid_lens):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / torch.sqrt(torch.tensor(d, dtype=torch.float32))
        # attention_weights: (`batch_size`, query个数, key个数)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
