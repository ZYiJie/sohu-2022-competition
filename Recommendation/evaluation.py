# coding: utf-8
import sys
import time
import traceback
from collections import defaultdict
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT) 
logger = logging.getLogger(__file__)
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import rankdata

@njit
def _auc(actual, pred_ranks):
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual == 1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)


def fast_auc(actual, predicted):
    # https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/208031
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)


def gAUC(labels, preds, group_id_list):
    """Calculate group AUC"""
    group_pred = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        group_id = group_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        group_pred[group_id].append(pred)
        group_truth[group_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for group_id in set(group_id_list):
        truths = group_truth[group_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[group_id] = flag

    total_auc = 0.0
    size = 0.0
    for group_id in group_flag:
        if group_flag[group_id]:
            auc = fast_auc(np.asarray(group_truth[group_id]), np.asarray(group_pred[group_id]))
            total_auc += auc
            size += 1.0
    group_auc = float(total_auc)/size
    return group_auc




def score(result_data, label_data):
    try:
        # 读取数据
        logger.info('Read data')
        result_df = pd.read_csv(result_data, delimiter="\t")
        label_df = pd.read_csv(label_data, delimiter=",")

        # 规范检查
        logger.info('Check result file')
        if result_df.shape[0] != label_df.shape[0]:
            err_msg = "结果文件的行数（%i行）与测试集（%i行）不一致"%(result_df.shape[0], label_df.shape[0])
            res = {
                "ret": 1,
                "err_msg": err_msg,
            }
            logger.error(res)
            return res
        err_cols = []
        result_cols = set(result_df.columns)
        target_cols = ["Id", "result"]
        for col in target_cols:
            if col not in result_cols:
                err_cols.append(col)
        if len(err_cols) > 0:
            err_msg = "结果文件缺少字段/列：%s"%(', '.join(err_cols))
            res = {
                "ret": 2,
                "err_msg": err_msg,
            }
            logger.error(res)
            return res

        # 拼接表
        result_df = result_df.rename(columns={"Id":"testSampleId"})

        df = label_df.merge(result_df, on=['testSampleId'], how='left')
        miss_sample_ids = df[np.isnan(df["result"])]["testSampleId"].tolist()
        if(len(miss_sample_ids)):
            err_msg = "结果文件缺少指定的testSampleId预测结果，比如 %s" % (miss_sample_ids)
            res = {
                "ret": 3,
                "err_msg": err_msg,
            }
            logger.error(res)
            return res

        # 计算分数
        logger.info('Compute score')
        labels = df["label"].tolist()
        logits = df["result"].map(lambda x:float(x)).tolist()
        pvid_list = df["pvId"].tolist()
        auc = gAUC(labels, logits, pvid_list)

        res = {
            "ret": 0,
            "data": {
                "score": auc
            }
        }
        logger.info(res)
    except Exception as e:
        traceback.print_exc()
        res = {
            "ret": 4,
            "err_msg": str(e)
        }    
        logger.error(res)
    return res

def cal_score():
    t = time.time()
    label_data = open('/Users/andy/Desktop/algoCompetition/dataset/hash/test-dataset-answer.csv', 'r')
    predict_data = open('/Users/andy/Desktop/algoCompetition/dataset/hash/sohuBaseline/data/submit/section2.txt', 'r')
    res = score(predict_data, label_data)
    print(res)
    print('Time cost: %.2f s'%(time.time()-t))

    
if __name__ == '__main__':
    cal_score()
    sys.exit(0)
