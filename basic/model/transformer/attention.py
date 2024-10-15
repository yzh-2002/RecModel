import torch
from basic.loss import masked_softmax
from basic.model.transformer.utils import transpose_qkv, transpose_output


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


class MultiHeadAttention(torch.nn.Module):
    """多头注意力"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        # 此处实际简化了实现，也即保证query，key，size经过线性变换之后的维度相等，均为 num_hiddens / num_heads
        # 虽然将多个head的参数合并到一起，但由于初始化时各个head的参数不一致，所以会学到不一样的东西
        self.W_q = torch.nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = torch.nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = torch.nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = torch.nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # 并行计算多头注意力
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # valid_lens的形状:(batch_size,) or (batch_size, query个数)
            # 将其广播到形状为 (batch_size*num_heads, ) or (...) 后
            # 其与valid_lens.repeat()区别：后者按整个张量的结构重复，前者按元素重复
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，query的个数，num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，query的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
