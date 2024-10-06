import numpy as np
import torch


class DenseFeature:
    """非离散型特征

    Args:
        name (str): 特征唯一标识
    """

    def __init__(self, name):
        self.name = name
        self.embed_dim = 1


def auto_embed_dim(category_num):
    """根据离散特征种类数量计算合适的emb向量维度

    参考论文：Deep & Cross Network for Ad Click Predictions.(ADKDD'17)
    Args:
        category_num (int): 略
    Returns:
        embed_dim
    """
    return np.floor(6 * np.pow(category_num, 0.25))


class SparseFeature:
    """离散型特征

    Args:
        name (str): 略
        vocab_size (int): 离散特征种类数量
        embed_dim (int): emb向量维度，自定义
        shared_with (str): TODO
        padding_idx (int,optional): TODO
    """

    def __init__(self, name, vocab_size, embed_dim, padding_idx, shared_with=None):
        self.name = name
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim if embed_dim else auto_embed_dim(vocab_size)
        self.shared_with = shared_with
        self.padding_idx = padding_idx
        self.embed = None

    def get_embedding_layer(self):
        if self.embed is None:
            embed = torch.nn.Embedding(self.vocab_size, self.embed_dim)
            # normal_和normal的区别：前者原地操作，后者创建并返回一个新张量，不改变现有张量
            torch.nn.init.normal_(embed.weight, mean=0.0, std=1.0)
            self.embed = embed
        return self.embed


class SequenceFeature:
    def __init__(self, name, vocab_size, embed_dim, pooling, padding_idx, shared_with=None):
        self.name = name
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim if embed_dim else auto_embed_dim(vocab_size)
        self.pooling = pooling
        self.shared_with = shared_with
        self.padding_idx = padding_idx
        self.embed = None

    def get_embedding_layer(self):
        if self.embed is None:
            embed = torch.nn.Embedding(self.vocab_size, self.embed_dim)
            torch.nn.init.normal_(embed.weight, mean=0.0, std=1.0)
            self.embed = embed
        return self.embed
