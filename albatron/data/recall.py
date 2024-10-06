import tqdm, random
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from collections import Counter, OrderedDict
from .utils import pad_or_truncate_sequence


def generate_negative_sample(items_cnt, ratio, method):
    """负采样
    :param items_cnt: （dict）物料出现次数
    :param ratio: 负样本个数，>=1
    :param method: 负采样方法
        0：（默认）随机负采样
        1："popularity sampling method used in word2vec"
        2："popularity sampling method by `log(count+1)+1e-6`"
        3："tencent RALM sampling"
    :return:
    """
    negative_items = None
    items_set = [item for item, count in items_cnt.items()]
    if method == 0:
        negative_items = np.random.choice(items_set, size=ratio, replace=True)
    elif method == 1:
        p_sel = {item: count ** 0.75 for item, count in items_cnt.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        negative_items = np.random.choice(items_set, size=ratio, replace=True, p=p_value)
    elif method == 2:
        p_sel = {item: np.log(count + 1) + 1e-6 for item, count in items_cnt.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        negative_items = np.random.choice(items_set, size=ratio, replace=True, p=p_value)
    elif method == 3:
        p_sel = {item: (np.log(k + 2) - np.log(k + 1)) / np.log(len(items_cnt) + 1) for item, k in
                 items_cnt.items()}
        p_value = np.array(list(p_sel.values())) / sum(p_sel.values())
        negative_items = np.random.choice(items_set, size=ratio, replace=False, p=p_value)
    return negative_items


def generate_seq_feature(data, user_col, item_col, time_col, sample_method, negative_ratio, items_attr_cols=None,
                         mode=0,
                         min_item=0):
    """召回阶段生成序列特征，该步骤与数据集划分强关联
    :param data:(pd.DataFrame) 原始数据
    :param user_col: user_id列对应名称
    :param item_col: 同上
    :param time_col: 同上
    :param items_attr_cols: 一般情况下Sequence序列由item id构成，此处可指定items其他属性用于构建序列特征
    :param sample_method: 负样本采样方法
    :param negative_ratio: 负样本与正样本的比例
    :param mode: 训练方式
        0：pointwise
        1：pairwise
        2：listwise
    :param min_item: 用户历史行为记录低于此值，判定为冷启动用户
    :return:
        train_set: (pd.DataFrame) 训练数据集（仅包含序列特征）
        test_set: 测试数据集
    """

    if items_attr_cols is None:
        items_attr_cols = []
    if mode == 2:
        assert negative_ratio > 0
    elif mode == 1:
        negative_ratio = 1
    train_set, test_set = [], []
    cold_user_num = 0
    data.sort_values(time_col, inplace=True)  # data按时间从前到后

    items_cnt = Counter(data[item_col].tolist())
    # TODO:为什么需要按顺序排列呢?
    items_cnt_order = OrderedDict(sorted(items_cnt.items(), key=lambda x: x[1], reverse=True))
    # TODO:此处负样本生成的个数貌似有点随意???
    negative_list = generate_negative_sample(items_cnt_order, ratio=data.shape[0] * negative_ratio,
                                             method=sample_method)
    negative_idx = 0
    last_col = None
    for uid, history_list in tqdm.tqdm(data.groupby(user_col), desc="generate sequence feature"):
        # history_list (pd.DataFrame)，仅包含uid用户的历史行为记录
        item_list = history_list[item_col].tolist()
        if len(item_list) < min_item:
            cold_user_num += 1
            continue
        for i in range(1, len(item_list)):  # 采用滑动窗口尽可能多的构建序列特征
            history_item = item_list[:i]
            # 样本构成：user_id,item_id,history_item_id,history_item_id_len,history_item_attr1,2...,last_col
            # 此处item_id为正样本，history_item_id_len用于后续序列特征的pad/truncate操作
            # last_col根据训练方式的不同其内容不同：
            # pointwise:独立看待每个样本,无需负样本,此时last_col为label(0/1),但需要往训练集中添加一些负样本,后续处理时再说明
            # pairwise:需要一个负样本
            # listwise:需要多个负样本
            sample = [uid, item_list[i], history_item, len(history_item)]
            if len(items_attr_cols) > 0:
                for attr_col in items_attr_cols:
                    sample.append(history_list[attr_col].tolist()[:i])
            if i != len(item_list) - 1:
                if mode == 0:
                    last_col = "label"
                    train_set.append(sample + [1])
                    # 训练集中添加一些负样本,正负样本比例最好1:2/3
                    for _ in range(negative_ratio):
                        sample[1] = negative_list[negative_idx]
                        negative_idx += 1
                        train_set.append(sample + [0])
                elif mode == 1:
                    last_col = "negative_item"
                    train_set.append(sample + [negative_list[negative_idx]])
                    negative_idx += 1
                elif mode == 2:
                    last_col = "negative_items"
                    sample.append(negative_list[negative_idx:negative_idx + negative_ratio])
                    negative_idx += negative_ratio
                    train_set.append(sample)
            else:  # 用户最后一次点击记录应该作为测试集数据
                test_set.append(sample + [1])
    random.shuffle(train_set)
    random.shuffle(test_set)
    print("n_train: %d, n_test: %d" % (len(train_set), len(test_set)))
    print("%d cold start user droped " % (cold_user_num))

    history_item_attr_cols = ["history_" + col for col in items_attr_cols]
    df_train = pd.DataFrame(train_set,
                            columns=[user_col, item_col, "history_" + item_col,
                                     "history_len"] + history_item_attr_cols + [last_col])
    df_test = pd.DataFrame(test_set,
                           columns=[user_col, item_col, "history_" + item_col,
                                    "history_len"] + history_item_attr_cols + [last_col])

    return df_train, df_test


def generate_model_input(df, user_features, user_col, item_features, item_col, sequence_max_len, padding="pre",
                         truncate="pre"):
    """召回模型最终数据集：
        1. pad or truncate序列特征
        2. 拼接序列特征与用户/物品的其他特征

    :param df: 经generate_seq_feature 生成的包含序列特征的测试集或训练集数据
    :param user_features: 用户侧的除序列特征外其他特征
    :param user_col: 用户侧col名称,用于拼接
    :param item_features: 同上
    :param item_col: 同上
    :param sequence_max_len: 序列特征最大长度
    :param padding: padding策略
    :param truncate: truncate策略
    :return:
    """
    df = pd.merge(df, user_features, on=user_col, how="left")  # left:保证左侧数据的所有行,在右侧寻找匹配行拼接,右侧找不到则填充NaN
    df = pd.merge(df, item_features, on=item_col, how='left')
    for col in tqdm.tqdm(df.columns.tolist(), desc="padding or truncating sequence feature"):
        if col.startswith("history_") and col != "history_len":
            df[col] = pad_or_truncate_sequence(df[col], max_len=sequence_max_len, value=0, padding=padding,
                                               truncate=truncate).tolist()
    # 将pd.DataFrame转换为dict
    input_dict = df.to_dict("list")
    for key in input_dict.keys():
        # np.array和np.asarray的区别:前者总是创建一个新的numpy数组,
        # 后者尝试转换为一个numpy数组,如果其参数就是,便返回引用
        input_dict[key] = np.asarray(input_dict[key])
    return input_dict


class Annoy:
    def __init__(self, metric='angular', n_trees=10, search_k=-1):
        """
        :param n_trees: 越大，查询精度越高，但查询速度和内存销毁会变高
        :param metric: 距离度量
        """
        self.annoy = None
        self.metric = metric
        self.n_trees = n_trees
        self.search_k = search_k  # -1代表使用默认搜索策略，搜索次数自动设置

    def fit(self, X):
        # 为向量构建索引
        self.annoy = AnnoyIndex(X.shape[1], metric=self.metric)
        for i, x in enumerate(X):
            self.annoy.add_item(i, x.tolist())
        self.annoy.build(self.n_trees)

    def query(self, v, n):
        # 查询v最近的n个向量
        # 返回二元组列表，第一个元素是向量索引，第二个元素是距离
        return self.annoy.get_nns_by_vector(v, n, self.search_k, include_distances=True)
