import torch, tqdm, os
from .loss import BPRLoss
from sklearn.metrics import roc_auc_score


class RecallTrainer:
    """召回模型通用训练器

    Args:
        model: 召回模型
        mode: 训练方式，与`generate_sequence_feature`中mode保持一致
        optimizer: 优化器，负责梯度计算，模型参数更新
            虽然Adam优化器自带学习率调增过程，但调度器提供了更全局的学习率管理
        scheduler: 调度器，用于动态调整学习率
        n_epoch:
        early_stop:
    """

    def __init__(self, model, mode, optimizer, n_epoch, device, model_path="./"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.mode = mode
        if self.mode == 0:
            self.criterion = torch.nn.BCELoss()  # 二元分类交叉熵损失函数
        elif self.mode == 1:
            # TODO:后续更换成`Triplet Hinge Loss` or `Triplet Logistic Loss`
            self.criterion = BPRLoss()
        elif self.mode == 2:
            self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.evaluate_fn = roc_auc_score
        self.n_epoch = n_epoch
        self.model_path = model_path
        # self.early_stopper = EarlyStopper(patience=early_stop)

    def train_one_epoch(self, data_loader, log_interval=10):
        """
        :param data_loader:
        :param log_interval: 每间隔多少batch输出一次日志
        :return:
        """
        self.model.train()
        total_loss = 0
        progress_bar = tqdm.tqdm(data_loader, desc="train")
        for i, (x, y) in enumerate(progress_bar):
            # 先将数据移至GPU
            x = {k: v.to(self.device) for k, v in x.items()}
            y = y.to(self.device)
            # 对于Pairwise和Listwise，其y为None
            if self.mode == 0:
                y = y.float()  # BCELoss需要float类型
            if self.mode == 1:
                pos_score, neg_score = self.model(x)
                loss = self.criterion(pos_score, neg_score)
            else:
                y_pred = self.model(x)
                # FIXME:Listwise的y为None？？？？
                loss = self.criterion(y_pred, y)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                progress_bar.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

    def fit(self, train_loader, valid_loader=None):
        for epoch in range(self.n_epoch):
            print("epoch:", epoch)
            self.train_one_epoch(train_loader)
            # TODO:此处可用调度器更新学习率
            if valid_loader:
                pass
        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pth"))

    def evaluate(self, data_loader):
        # data_loader:验证集数据集
        self.model.eval()
        targets, predicts = [], []
        with torch.no_grad():
            progress_bar = tqdm.tqdm(data_loader, desc="validation")
            for i, (x, y) in enumerate(progress_bar):
                x = {k: v.to(self.device) for k, v in x.items()}
                y = y.to(self.device)
                y_pred = self.model(x)
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        return self.evaluate_fn(targets, predicts)

    def predict(self, data_loader):
        self.model.eval()
        predicts = []
        with torch.no_grad():
            progress_bar = tqdm.tqdm(data_loader, desc="predict")
            for i, (x, y) in enumerate(progress_bar):
                x = {k: v.to(self.device) for k, v in x.items()}
                y = y.to(self.device)
                y_pred = self.model(x)
                predicts.extend(y_pred)
        return predicts

    def inference_embedding(self, mode, data_loader):
        """从已完成训练的模型推导出data_loader中的embedding向量（经过mlp层的最终向量）
        :param mode: 用户塔/物品塔
        :param data_loader:
        """
        assert mode in ["user", "item"], "Invalid mode={}.".format(mode)
        self.model.mode = mode
        self.model.load_state_dict(torch.load(os.path.join(self.model_path, "model.pth"), weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()
        predicts = []
        with torch.no_grad():
            progress_bar = tqdm.tqdm(data_loader, desc="%s inference" % mode)
            for i, x in enumerate(progress_bar):
                x = {k: v.to(self.device) for k, v in x.items()}
                y_pred = self.model(x)  # (batch_size,{mode}_params["dims"][-1])
                predicts.append(y_pred.detach())  # 返回一个新tensor，与计算图脱离联系
        return torch.cat(predicts, dim=0)
