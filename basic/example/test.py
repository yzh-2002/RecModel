import torch
from basic.model.attention.attention import AdditiveAttention

queries, keys = torch.normal(0, 1, (2, 1, 2)), torch.ones((2, 2, 3))
# values的小批量，两个值矩阵是相同的
values = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4).repeat(
    2, 1, 1)
valid_lens = torch.tensor([2, 2])

attention = AdditiveAttention(key_size=3, query_size=2, hidden_size=2, dropout=0.1)
attention.eval()
attention(queries, keys, values, valid_lens)
