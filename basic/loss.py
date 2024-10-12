import torch


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(torch.nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        # 创建一个与label形状相同的全1张量
        weights = torch.ones_like(label)
        # 将超过有效长度的位置置为0，也即mask
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'  # 设置torch.nn.CrossEntropyLoss的reduction属性为'none'，也即不对输出进行任何默认汇总操作
        # output shape: (`batch_size`, `num_steps`)
        output = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        # 只有有效长度部分的损失参与求和
        return (output * weights).mean(dim=1)  # shape: (`batch_size`,)
