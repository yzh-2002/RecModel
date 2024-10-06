from torch.utils.data import Dataset


class PointWiseDataset(Dataset):
    """带有标签的数据集,主要用于pointwise训练方式
    """

    def __init__(self, x, y):
        """
        :param x: dict(feature_key,feature_value(list))
        :param y: list
        """
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}, self.y[index]

    def __len__(self):
        return len(self.y)


class PairOrListWiseDataset(Dataset):
    """不带标签的数据集,例如:pairwise和listwise
    """

    def __init__(self, x):
        super().__init__()
        self.x = x

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}

    def __len__(self):
        return len(self.x[list(self.x.keys())[0]])
