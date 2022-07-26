# _*_ coding:utf-8 _*_
'''
@file: main.py
'''
import os
from argparse import ArgumentParser
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from generate_rec_feature import deal_feature
from sklearn.metrics import roc_auc_score, f1_score
from deepctr_torch.models import DIFM
from deepctr_torch.inputs import SparseFeat, get_feature_names, VarLenSparseFeat, DenseFeat

def cal_group_auc(labels, preds, user_id_list):
    """Calculate group auc"""

    print('*' * 50)
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0
    #
    for user_id in tqdm(group_flag):
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 5)
    return group_auc


def predict(info):
    res = []
    for k in info["testSampleId"].values:
        res.append((k, random.uniform(0, 1)))
    return res


def infer(args):
    print('======模型推理======')
    newTest_all_feature = pd.read_csv('./tmp/test.csv')
    feature_names = ['province', 'city', 'deviceType', 'logTs_gap', 'Hour', 'bm25_mean', 'co_bm25_mean', 
                     'bm25', 'bm25Len', 'gapMeanEmotion', 'groupLen', 'historyHitSort', 'gongxian']
    test_model_input = {name: newTest_all_feature[name] for name in feature_names}
    
    model = torch.load(os.path.join(args.model_save_path, 'DIFM.h5'))
    testSampleId = pd.read_csv(args.test_input)['testSampleId']
    pred_ans = model.predict(test_model_input, batch_size=1024)
    pred = pd.Series(pred_ans.flatten('F'))
    output_df = pd.concat([testSampleId, pred], axis=1)
    output_path = args.output
    output_df.to_csv(output_path, index=False,sep="\t", header=["Id", "result"])
    print(f'save result at {output_path}')


def train(args):
    print('======数据预处理======')
    #deal_feature() # 形成预处理数据
    train_df = pd.read_csv('./tmp/train.csv')
    test_df = pd.read_csv('./tmp/test.csv')
    
    used_features = ['province', 'city', 'deviceType', 'logTs_gap', 'Hour', 'bm25_mean', 'co_bm25_mean', 
                     'bm25', 'bm25Len', 'gapMeanEmotion', 'groupLen', 'historyHitSort', 'gongxian']
    fixlen_feature_columns = [SparseFeat(feat, 
                                          pd.concat([train_df[feat], test_df[feat]]).nunique(), 
                                          embedding_dim=512) 
                              for feat in used_features]
    target = ['label']

    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns 
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    print('feature names:', feature_names)
    
    # 划分训练测试集
    pvId_list = list(set(train_df['pvId']))
    random.shuffle(pvId_list)
    length = len(pvId_list)
    train_pvId = pvId_list[:int(length*0.95)]
    valid_pvId = pvId_list[int(length*0.95):]
    train = train_df.loc[train_df['pvId'].isin(train_pvId)]
    valid = train_df.loc[train_df['pvId'].isin(valid_pvId)]

    # 模型输入
    train_model_input = {name: train[name] for name in feature_names}
    valid_model_input = {name: valid[name] for name in feature_names}
    test_model_input = {name: test_df[name] for name in feature_names}
    
    # 模型定义和训练
    print('======模型训练======')
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda'

    model = DIFM(linear_feature_columns, dnn_feature_columns, task='binary', device=device)

    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', 'auc'])
    history = model.fit(train_model_input, train[target].values,

                        batch_size=2048, epochs=1, verbose=1, 

    #                     validation_split=0.1)

                       validation_data=(valid_model_input, valid[target].values))
    # 模型验证
    pred_ans = model.predict(valid_model_input, batch_size=1024)
    print("valid AUC", round(roc_auc_score(valid[target].values, pred_ans), 5))
    # print("valid f1_score", round(f1_score(valid[target].values, pred_ans), 5))

    labels = pd.Series(valid[target].values.flatten('F'))
    pred = pd.Series(pred_ans.flatten('F'))
    print(len(labels), len(pred))
    gauc = cal_group_auc(labels, pred, valid['pvId'].astype(str).tolist())
    print("valid GroupAUC", gauc)
    
    # 保存模型
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    save_path = os.path.join(args.model_save_path, 'DIFM.h5')
    torch.save(model, save_path)
    print(f'save model at {save_path}')
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_input", type=str, required=True, help="训练输入文件")
    parser.add_argument("--test_input", type=str, required=True, help="测试输入文件")
    parser.add_argument("--nlp_model_path", type=str, required=True, help="预训练模型地址，我们将使用NLP生成的模型来生成情感特征,并导入")
    parser.add_argument("--output", type=str, required=True, help="结果输出文件")
    parser.add_argument("--model_save_path", type=str, required=True, help="训练好的模型存放地址")
    args = parser.parse_args()
    train(args)
    infer(args)
