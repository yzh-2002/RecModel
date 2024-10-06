import torch
import torch.nn.functional as F
from ..layers.EmbeddingLayer import EmbeddingLayer
from ..layers.MLP import MLP


class DSSM(torch.nn.Module):
    # TODO:目前仅支持Pointwise，其他训练方式会报错
    def __init__(self, user_features, item_features, user_params, item_params, mode=None, temperature=1.0):
        """
        :param user_features: 用户特征类型,用于构建EmbeddingLayer结构
        :param item_features: 同上
        :param user_params: 用户塔参数,用于构建MLP结构
        :param item_params: 同上
        :param mode: 训练时设置为None，推理时设置为user/item
        :param temperature: 温度系数
        """
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.temperature = temperature
        self.embedding = EmbeddingLayer(user_features + item_features)
        # 用户/物品塔MLP层的输入向量维度等于其各个特征经EmbeddingLayer层的embed_dim拼接在一起的维度
        self.user_dims = sum([feature.embed_dim for feature in user_features])
        self.item_dims = sum([feature.embed_dim for feature in item_features])
        # *variable用于将可迭代对象解包传递,**variable用于将字典解包传递
        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)
        self.item_mlp = MLP(self.item_dims, output_layer=False, **item_params)
        self.mode = mode

    def forward(self, x):
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)
        if self.mode == 'user':
            return user_embedding
        if self.mode == 'item':
            return item_embedding
        y = F.cosine_similarity(user_embedding, item_embedding, dim=1)
        return torch.sigmoid(y / self.temperature)

    def user_tower(self, x):
        # 推理阶段，x不包含另一侧的特征，故需要在此处判断，不然会报错
        if self.mode == 'item':
            return None
        input_user = self.embedding(x, self.user_features)
        user_embedding = self.user_mlp(input_user)
        user_embedding = F.normalize(user_embedding, p=2, dim=1)  # 对指定dim进行L2正则，也即使其模长为1
        return user_embedding

    def item_tower(self, x):
        if self.mode == 'user':
            return None
        input_item = self.embedding(x, self.item_features)
        item_embedding = self.item_mlp(input_item)
        item_embedding = F.normalize(item_embedding, p=2, dim=1)
        return item_embedding
