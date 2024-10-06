import torch
from ..activation import activation_layer


class MLP(torch.nn.Module):
    """Multi Layer Perceptron Module,简单来讲就是多层全连接网络
    Args:
        input_dim (int): 第一层线性层的输入维度
        output_layer (bool): 是否作为网络最后一个模块,如果是,需要添加一层Linear(*,1)
        dims (list): 隐含层的输出维度
        dropout (float):
        activation (str): 激活层函数选择
    """

    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0, activation="relu"):
        super().__init__()
        if dims is None:
            dims = []
        layers = list()
        for dim in dims:
            layers.append(torch.nn.Linear(input_dim, dim))
            layers.append(torch.nn.BatchNorm1d(dim))
            layers.append(activation_layer(activation))
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
