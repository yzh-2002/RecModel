import torch
from ..features import SparseFeature, SequenceFeature


class InputMask(torch.nn.Module):
    """序列特征往往需要经过padding/truncate操作,尤其是padding操作向特征中加入了一些无效的占位符
        训练时需要使用mask机制来屏蔽这些无效的位置,避免其影响

        常用于self-attention机制中,此处多用于序列特征pooling时屏蔽无效位置

    """

    def __init__(self):
        super().__init__()

    def forward(self, x, features):
        """此处的输入等价于EmbeddingLayer的输入
        """
        mask = []
        # TODO:每次应该仅处理一个feature吧???
        if not isinstance(features, list):
            features = [features]
        for feature in features:
            # TODO:为什么还能SparseFeature,不应该只有SequenceFeature嘛?
            if isinstance(feature, SparseFeature) or isinstance(feature, SequenceFeature):
                if feature.padding_idx is not None:
                    feature_mask = x[feature.name].long() != feature.padding_idx
                else:
                    feature_mask = x[feature.name].long() != -1
                # feature_mask.shape (batch_size,1) or (batch_size,1,sequence_len)
                mask.append(feature_mask.unsqueeze(1).float())
        # cat函数的参数为一个tensor list,按照list中tensor指定维度拼接在一起并返回
        return torch.cat(mask, dim=1)
