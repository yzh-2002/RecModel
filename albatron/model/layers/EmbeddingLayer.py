import torch

from ..features import DenseFeature, SparseFeature, SequenceFeature
from .pooling import SumPooling, AveragePooling, ConcatPooling
from .utils import InputMask


class EmbeddingLayer(torch.nn.Module):
    """
    通用的Embedding Layer：存储各个特征的embedding_table

    Args:
        features (list): 内置Feature class列表，需要为每一个特征创建一个embedding_table

    """

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.embed_dict = torch.nn.ModuleDict()

        # 将各个特征的初始Embedding Layer添加到embed_dict中
        for feature in features:
            if feature.name in self.embed_dict:
                continue
            if isinstance(feature, SparseFeature) and feature.shared_with is None:
                self.embed_dict[feature.name] = feature.get_embedding_layer()
            elif isinstance(feature, SequenceFeature) and feature.shared_with is None:
                self.embed_dict[feature.name] = feature.get_embedding_layer()
            elif isinstance(feature, DenseFeature):
                pass

    def forward(self, x, features):
        """
        :param x: (dict) {feature_name:feature_value}，由`generate_model_input`返回的召回模型数据集
            如果feature是sequence类型,则其value shape:(batch_size,sequence_len)
            如果feature是sparse/dense类型,则value shape:(batch_size,)
        :param features: 内置Feature Class列表,用于告知模型当前需要对哪些特征进行嵌入
        :param squeeze_dim: TODO:按道理讲都应该合并
        :return:
        """
        sparse_emb = []
        dense_emb = []
        for feature in features:
            if isinstance(feature, SparseFeature):
                if feature.shared_with is None:
                    # (batch_size,embed_dim)=>(batch_size,1,embed_dim),便于后续和SequenceFeature的emb向量拼接
                    sparse_emb.append(self.embed_dict[feature.name](x[feature.name].long()).unsqueeze(1))
                else:
                    sparse_emb.append(self.embed_dict[feature.shared_with](x[feature.name].long()).unsqueeze(1))
            elif isinstance(feature, SequenceFeature):
                # 默认为均分池化
                pooling_layer = AveragePooling()
                if feature.pooling == "sum":
                    pooling_layer = SumPooling()
                elif feature.pooling == "mean":
                    pooling_layer = AveragePooling()
                elif feature.pooling == "concat":
                    pooling_layer = ConcatPooling()
                feature_mask = InputMask()(x, feature)  # shape(batch_size,1,sequence_len)
                if feature.shared_with is None:
                    # sum/average返回值shape:(batch_size,1,embed_dim)
                    # concat返回值shape:(batch_size,sequence_len,embed_dim)
                    # concat未消除padding的影响...
                    sparse_emb.append(
                        pooling_layer(self.embed_dict[feature.name](x[feature.name].long()), feature_mask).unsqueeze(1))
                else:
                    sparse_emb.append(pooling_layer(self.embed_dict[feature.shared_with](x[feature.name].long()),
                                                    feature_mask).unsqueeze(1))
            else:  # DenseFeature处理
                dense_emb.append(x[feature.name].float().unsqueeze(1))  # (batch_size,1)

        dense_value, sparse_value = None, None
        if len(dense_emb) > 0:
            dense_value = torch.cat(dense_emb, dim=1)  # (batch,feature_num)
        if len(sparse_emb) > 0:
            sparse_value = torch.cat(sparse_emb, dim=1)  # (batch,feature_num,embed_dim)

        if dense_value is not None and sparse_value is not None:
            return torch.cat((sparse_value.flatten(start_dim=1), dense_value), dim=1)
        elif dense_value is not None:
            return dense_value
        elif sparse_value is not None:
            return sparse_value.flatten(start_dim=1)  # (batch,feature_num*embed_dim)
