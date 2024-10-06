import collections

import numpy as np


def ndcg():
    pass


def hit():
    pass


def mrr():
    # 平均倒数排名
    pass


def recall():
    pass


def precision():
    pass


def topk_metrics(y_true, y_pred, topKs):
    """ TOPK 的评估指标通常用于评估推荐系统或信息检索系统的性能，主要关注于系统返回的前 K 个推荐项的质量和相关性
        常见的topk评估指标有：precision,recall,map,ndcg,hit,mrr
    :param y_true(dict): {user_id: [item_id, ...]}
    :param y_pred(dict): 同上
    :param topKs(list):
    :return: dict {metric_name: metric_values}
    """
    assert len(y_true) == len(y_pred)
    pred_list, true_list = [], []
    for user in y_true.keys():
        pred_list.append(y_pred[user])
        true_list.append(y_true[user])
    ndcg_result, mrr_result, hit_result, precision_result, recall_result = [], [], [], [], []
    for topk in topKs:  # 计算每个TopK对应的评估指标
        ncdgs, mrrs, hits, precisions, recalls = 0, 0, 0, 0, 0
        for i in range(len(true_list)):  # 计算每个用户对应指标的值，再取平均
            if len(true_list[i]) != 0:  # 可能存在某些新用户作为测试用户
                dcg = 0
                idcg = 0
                mrr_flag = True  # mrr指标仅关心第一个命中的物品
                mrr = 0
                hit = 0
                for j in range(topk):  # 遍历模型给用户推荐列表的前topk个，进行相应指标计算
                    if pred_list[i][j] in true_list[i]:
                        dcg += 1 / np.log2(j + 2)  # 相关度为1或0，在交互列表出现过则为1，否则为0，log2(i+1)为位置折扣因子
                        if mrr_flag:
                            mrr_flag = False
                            mrr = 1 / (j + 1)
                        hit += 1
                    # 理想的dcg值应为相关度为1的排在最前面
                    # TODO： 此处假设模型推荐的前topk个都是用户真实交互过的物品，但可能不足topk个
                    if j < len(true_list[i]):
                        idcg += 1 / np.log2(j + 2)
                if idcg != 0:
                    ncdgs += dcg / idcg
                mrrs += mrr
                hits += 1 if hit > 0 else 0  # hit rate:推荐列表中包含至少一个用户感兴趣的项目的比例
                recalls += hit / len(true_list[i])
                precisions += hit / topk
        ndcg_result.append(round(ncdgs / len(true_list), 4))
        mrr_result.append(round(mrrs / len(pred_list), 4))
        hit_result.append(round(hits / len(true_list), 4))
        recall_result.append(round(recalls / len(true_list), 4))
        precision_result.append(round(precisions / len(true_list), 4))

    results = collections.defaultdict(list)
    for idx in range(len(topKs)):
        output = f'NDCG@{topKs[idx]}: {ndcg_result[idx]}'
        results['NDCG'].append(output)

        output = f'MRR@{topKs[idx]}: {mrr_result[idx]}'
        results['MRR'].append(output)

        output = f'Recall@{topKs[idx]}: {recall_result[idx]}'
        results['Recall'].append(output)

        output = f'Hit@{topKs[idx]}: {hit_result[idx]}'
        results['Hit'].append(output)

        output = f'Precision@{topKs[idx]}: {precision_result[idx]}'
        results['Precision'].append(output)
    return results
