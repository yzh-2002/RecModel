from torch.utils.data import DataLoader
from .dataset import PointWiseDataset, PairOrListWiseDataset


class RecallData:
    def __init__(self, x, y=None):
        super().__init__()
        if y is None:
            y = []
        if len(y) != 0:  # pointwise
            self.dataset = PointWiseDataset(x, y)
        else:  # pairwise or listwise 没有标签
            self.dataset = PairOrListWiseDataset(x)

    def generate_dataloader(self, batch_size, test_user, all_item, num_workers=8):
        """ 生成召回过程所需DataLoader
        :param batch_size:
        :param test_user: 模型训练完后需要在测试集用户上验证模型效果
        :param all_item: 对测试用户在全体物料池上进行召回，计算MAP,NDCG等召回验证指标
        :param num_workers:
        :return:
        """
        trainDataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # 注意此处，无需传入标签，原因在于验证方式不是通过标签计算auc，而是通过用户/物品embedding向量计算相似度，所以只传特征即可
        test_user_dataset = PairOrListWiseDataset(test_user)
        all_item_dataset = PairOrListWiseDataset(all_item)
        test_user_dataloader = DataLoader(test_user_dataset, batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers)
        all_item_dataloader = DataLoader(all_item_dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers)
        return trainDataloader, test_user_dataloader, all_item_dataloader
