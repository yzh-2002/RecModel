import torch


class SumPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        """
        :param x: shape(batch_size,sequence_len,embed_dim)
        :param mask: shape(batch_size,1,sequence_len)
        :return: shape(batch_size,embed_dim)
        """
        if mask is None:
            return torch.sum(x, dim=1)
        else:
            # batch matrix multiplication
            # (batch,n,m) (batch,m,p)=>(batch,n,p)
            return torch.bmm(mask, x).squeeze(1)


class AveragePooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return torch.mean(x, dim=1)
        else:
            sum_pooling_matrix = torch.bmm(mask, x).squeeze()  # (batch_size,embed_dim)
            non_padding_length = mask.sum(dim=-1)  # (batch_size,)
            return sum_pooling_matrix / (non_padding_length.float() + 1e-16)


class ConcatPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        return x
