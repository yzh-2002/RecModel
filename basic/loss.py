import torch
import torch.nn.functional as F


def sequence_mask(X, valid_len, value=0.):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                          value=-1e6)
        return F.softmax(X.reshape(shape), dim=-1)


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
