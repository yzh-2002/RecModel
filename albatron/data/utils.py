import numpy as np


def pad_or_truncate_sequence(sequences, max_len, padding='pre', truncate="pre", value=0.):
    """填充或截断序列,使其保持相同长度(深度学习要求输入数据具有固定的形状)

    :param sequences: 待处理的序列
    :param max_len: 序列最大长度
    :param padding: 填充策略
    :param truncate: 截断策略
    :param value: 填充的值
    """
    assert padding in ["pre", "post"], "Invalid padding={}.".format(padding)
    assert truncate in ["pre", "post"], "Invalid truncating={}.".format(truncate)

    if max_len is None:
        max_len = max(len(x) for x in sequences)
    arr = np.full((len(sequences), max_len), value)
    for idx, x in enumerate(sequences):
        if len(x) == 0:
            continue
        if truncate == 'pre':
            # 截取前面的,保留后面max_len个元素
            trunc = x[-max_len:]
        else:
            trunc = x[:max_len]
        trunc = np.asarray(trunc)
        if padding == 'pre':
            arr[idx, -len(trunc):] = trunc
        else:
            arr[idx, :len(trunc)] = trunc
    return arr
