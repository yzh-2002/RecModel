"""
数据集：MovieLens-1M：https://grouplens.org/datasets/movielens/1m/
"""
import os.path

import pandas as pd
import numpy as np
import torch, collections
from sklearn.preprocessing import LabelEncoder
from albatron.data.recall import generate_seq_feature, generate_model_input, Annoy
from albatron.model.features import SparseFeature, SequenceFeature
from albatron.data.dataloader import RecallData
from albatron.model.recall.dssm import DSSM
from albatron.trainer.recall_trainer import RecallTrainer
from albatron.model.metric import topk_metrics

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
torch.manual_seed(2024)


def recall_evaluation(user_embedding, item_embedding, test_user, all_item, user_map, item_map, user_col='user_id',
                      item_col='movie_id', topk=10):
    """
    :param user_embedding:
    :param item_embedding:
    :param test_user: (dict)
    :param all_item:
    :param user_map/item_map: user/item原始id映射
    :return:
    """
    print("evaluate embedding matching on test data")
    annoy = Annoy()
    annoy.fit(item_embedding)
    # 对于每个测试用户，寻找其最近邻的n个物品
    recall_dicts = collections.defaultdict(dict)  # user_id:predict_item_id
    for user_id, user_emb in zip(test_user[user_col], user_embedding):
        items_idx, items_scores = annoy.query(v=user_emb, n=topk)
        # np.vectorize:将普通 Python 函数转换为可以作用于 NumPy 数组的向量化函数
        recall_dicts[user_map[user_id]] = np.vectorize(item_map.get)(all_item[item_col][items_idx])

    # 对于测试用户，寻找其访问过的物品
    data = pd.DataFrame({user_col: test_user[user_col], item_col: test_user[item_col]})
    data[user_col] = data[user_col].map(user_map)
    data[item_col] = data[item_col].map(item_map)
    # 按照用户id分组，将用户访问过的物品id转换为list
    user_pos_item = data.groupby(user_col).agg(list).reset_index()
    ground_truth = dict(zip(user_pos_item[user_col], user_pos_item[item_col]))  # user_id:history_item_id

    out = topk_metrics(y_true=ground_truth, y_pred=recall_dicts, topKs=[topk])
    print(out)


if __name__ == "__main__":
    file_path = "./movielens-1m/ml-1m.csv"
    data = pd.read_csv(file_path)
    user_col = 'user_id'
    item_col = 'movie_id'
    # TODO:rating title genres未使用
    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    dense_features = []

    # 对原始数据的离散特征进行编码
    feature_max_idx = {}
    for feature in sparse_features:
        lbe = LabelEncoder()
        # fit_transform对指定特征进行标签编码，范围[0,n-1]，+1是为了避免与可能的序列特征padding=0混淆
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
        if feature == user_col:
            user_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}
        if feature == item_col:
            item_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}

    # 定义双塔结构使用的特征
    user_tower_features = ['user_id', 'gender', 'age', 'occupation', 'zip']
    item_tower_features = ['movie_id']
    user_profile = data[user_tower_features].drop_duplicates("user_id")
    item_profile = data[item_tower_features].drop_duplicates('movie_id')

    # DSSM采用pointwise训练方式，正负样本1：3
    # 注意此处测试集的划分，将用户最后一次点击记录作为测试集数据
    train_df, test_df = generate_seq_feature(data, user_col, item_col, time_col="timestamp", items_attr_cols=[],
                                             sample_method=1, mode=0,
                                             negative_ratio=3, min_item=0)
    # 为上述训练数据集拼接上更多用户，物品特征
    train_x = generate_model_input(train_df, user_profile, user_col, item_profile, item_col, sequence_max_len=50)
    train_y = train_x["label"]  # pointwise 才有该标签，pairwise和listwise是nagative_item(s)
    test_x = generate_model_input(test_df, user_profile, user_col, item_profile, item_col, sequence_max_len=50)
    test_y = test_x["label"]

    user_features = [
        SparseFeature(feature_name, vocab_size=feature_max_idx[feature_name], embed_dim=16, padding_idx=0) for
        feature_name
        in user_tower_features
    ]
    user_features += [
        # 该序列名称与`generate_sequence_feature`函数强关联
        SequenceFeature("history_movie_id", vocab_size=feature_max_idx["movie_id"], embed_dim=16, pooling="mean",
                        shared_with="movie_id", padding_idx=0)
    ]
    item_features = [
        SparseFeature(feature_name, vocab_size=feature_max_idx[feature_name], embed_dim=16, padding_idx=0) for
        feature_name
        in item_tower_features
    ]
    # 数据集要求整理为字典形式
    all_items_dict = item_profile.to_dict("list")
    for key in all_items_dict.keys():
        all_items_dict[key] = np.asarray(all_items_dict[key])

    # test_user_dataloader, all_item_dataloader用于验证召回效果
    train_dataloader, test_user_dataloader, all_item_dataloader = RecallData(x=train_x, y=train_y).generate_dataloader(
        batch_size=256, test_user=test_x, all_item=all_items_dict)

    # 定义DSSM结构
    model = DSSM(user_features, item_features, temperature=0.02, user_params={
        "dims": [256, 128, 64],
        "activation": "prelu"
    }, item_params={
        "dims": [256, 128, 64],
        "activation": "prelu"
    })
    # 定义模型训练器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    trainer = RecallTrainer(model, mode=0, optimizer=optimizer, n_epoch=2, device='cuda',
                            model_path=os.path.dirname(__file__))
    trainer.fit(train_dataloader)

    # 测试集上验证训练效果
    user_embedding = trainer.inference_embedding("user", test_user_dataloader)
    item_embedding = trainer.inference_embedding("item", all_item_dataloader)
    recall_evaluation(user_embedding, item_embedding, test_x, all_items_dict, user_map, item_map)
