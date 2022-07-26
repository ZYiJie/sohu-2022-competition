#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd
import numpy as np
import torch
import os
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import random
import json
from tqdm import tqdm
from collections import Counter
from deepctr_torch.inputs import SparseFeat, get_feature_names, VarLenSparseFeat, DenseFeat
import jieba

class BM25_Model(object):
    def __init__(self, documents_list, k1=2, k2=1, b=0.75):
        self.documents_list = documents_list
        self.documents_number = len(documents_list)
        self.avg_documents_len = sum([len(document) for document in documents_list]) / (self.documents_number+1)
        self.f = []
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                    self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
                             qf[q] * (self.k2 + 1) / (qf[q] + self.k2))

        return score

    def get_documents_score(self, query):
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list
def generate_historyFeature(feature, itemCounter, entityCounter,entity_map, emotionDic):
    print(f'总长={len(feature)}')
    out = [] # bm25、bm25Len、co_bm25、co_bm25Len、gongxian
    result = [] # [现文章平均情感，历史文章平均情感，重叠实体情感差距]
    wrongHisCnt = 0
#     for item in tqdm(feature.sample(1000)[['itemId','logTs','seq', 'pvId']].itertuples()):
    for item in tqdm(feature[['itemId','logTs','seq', 'pvId']].itertuples()):
#         print(item.label)
        itemId = int(item.itemId)
        
        curTime = int(item.logTs)
        seqArr = item.seq
        if type(seqArr)!=list or len(seqArr) == 0:  # 无历史信息
            out.append([-1, 0, -1, 0, 0]) 
            result.append([-2, -2, -2]) # 无历史
            continue

        assert len(seqArr)%2 == 0
        entitys = [] # 历史实体
        timeGaps = []
        items = []
        for i in range(0,len(seqArr),2):
            timeGap = int((curTime-int(seqArr[i+1]))/1000/3600)
            itemId = int(seqArr[i])
            if len(entity_map[itemId]) == 0 or timeGap < 0:
                wrongHisCnt += 1
                continue
            itemCounter[itemId] += 1
            entitys.append(entity_map[itemId])
            timeGaps.append(timeGap)
            items.append(itemId)
            for entity in entity_map[itemId]:
                entityCounter[entity] += 1
#       
        curEntitys = []
        curTemp = entity_map[itemId]
        for entity in curTemp:
            if len(entity) > 5:
                words = jieba.cut(entity, cut_all=False)
                curEntitys.extend(words)
            else:
                curEntitys.append(entity)
        historyEntitys = []
        entity_gongxian_times = 0 ## 共现实体次数
        curTemp = set(curTemp)
        
        ## 情感特征计算
        curEmotion = []
        historyEmotion = []
        emotionGap = []
        for entity in curTemp:
            curEmotion.append(emotionDic[itemId][entity])

        for i in range(0,len(seqArr),2):
            s1 = set(entity_map[int(seqArr[i])]) # 历史文章实体
            for entity in s1:
                emotion = int(emotionDic[int(seqArr[i])][entity])
                historyEmotion.append(emotion)
                if entity in curTemp:
                    emotionGap.append(abs(emotionDic[int(itemId)][entity]-emotion))
        if len(curEmotion) > 0:
            curMean = int(np.mean(curEmotion)*20)
        else:
            curMean = -1
        if len(historyEmotion) > 0:
            historyMean = int(np.mean(historyEmotion)*20)
        else:
            historyMean = -1
        if len(emotionGap) > 0:
            gapMean = int(np.mean(emotionGap)*20)
        else:
            gapMean = -1
        result.append([curMean, historyMean, gapMean])
        ## 历史BM25特征计算
        for group in entitys:
            tempGroup = []
            for entity in group:
                if entity in curTemp:
                    entity_gongxian_times += 1
                if len(entity) > 5:
                    words = jieba.cut(entity, cut_all=False)
                    tempGroup.extend(words)
                else:
                    tempGroup.append(entity)
            historyEntitys.append(tempGroup)
        bm25Model = BM25_Model(historyEntitys)
        
        bm25Arr = bm25Model.get_documents_score(curEntitys)
        bm25Arr = np.abs(bm25Arr)
        bm25 = int(sum(bm25Arr)+0.5) # query 与历史的bm25相似度
        
        co_scoreArr = []
        for i in range(len(historyEntitys)):
            for j in range(i+1,len(historyEntitys)):
                tempScore = bm25Model.get_score(j, historyEntitys[i])
                co_scoreArr.append(tempScore)
        co_scoreArr = np.abs(co_scoreArr)
        co_score = int(sum(co_scoreArr)+0.5)
        out.append([bm25, len(bm25Arr), co_score, len(co_scoreArr), entity_gongxian_times]) 
    print(f'错误历史数据数量={wrongHisCnt}')
    print(f'历史item词典大小={len(itemCounter)}')
    print(f'历史实体词典大小={len(entityCounter)}')
    return out, result

def deal_raw_data(path):
    data = pd.read_csv(path)
    data['time'] = pd.to_datetime(data['logTs'],unit='ms',origin=pd.to_datetime('1970-01-01 08:00:00'))
    data['Hour'] = data['time'].dt.hour
    data['Min'] = data['time'].dt.hour*60+data['time'].dt.minute
    data['seq'] = data['userSeq'].str.split('[;:]').fillna(0)
    
    logTs_min = data.groupby('pvId')['logTs'].min()
    logTs_min = pd.DataFrame({"logTs_min": logTs_min}).reset_index()
    data = pd.merge(data, logTs_min, how='left', on='pvId')
    data['logTs_gap'] = (data['logTs'] - data['logTs_min'])/1000
    data['logTs_gap'] = data['logTs_gap'].astype(int)
    del data['logTs_min']
    print(len(data), len(data.columns))
    return data

def deal_feature():
    recommend_content_entity_paths = ['./data/rec/recommend_content_entity_0317_初赛.txt',
                                     './data/rec/recommend_content_entity_复赛_训练.txt',
                                     './data/rec/recommend_content_entity_复赛_测试.txt',]
    entity_map = {}
    for path in recommend_content_entity_paths:
        with open(path) as f:
            for line in f:
                if len(line.strip()):
                    js = json.loads(line)
                    entity_map[int(js['id'])] = js['entity']
    print('entity_map len = ', len(entity_map))
    
    emotionDic = {}
    with open('./data/rec/sentiment.dic', 'r', encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split('\t')
            if arr[0] == 'id': continue
            emotionDic[int(arr[0])] = json.loads(arr[1])
    print(len(emotionDic))
#     data_paths = ['/home/zyj/sohu/Recommendation/data/rec_data/train-dataset.csv',
#                  '/home/zyj/sohu/Recommendation/data/rec_data/newTrain-dataset.csv',
#                  '/home/zyj/sohu/Recommendation/data/rec_data/newTest-dataset.csv',]
    data_paths = ['./data/rec/train-dataset_初赛.csv',
                 './data/rec/rec-train-dataset_复赛.csv',
                 './data/rec/rec-test-dataset_复赛.csv',]
    itemCounter, entityCounter = Counter(), Counter()
    features = []
    for path in data_paths:
        feature = deal_raw_data(path)
        bm25_df, emotion_df = generate_historyFeature(feature, itemCounter, entityCounter, entity_map, emotionDic)
        bm25_df = pd.DataFrame(bm25_df, columns=['bm25','bm25Len','co_bm25','co_bm25Len','gongxian'])
        emotion_df = pd.DataFrame(emotion_df, columns=['curMeanEmotion', 'historyMeanEmotion', 'gapMeanEmotion'])
        feature = pd.concat([feature, bm25_df, emotion_df],axis=1)
        del feature['seq']
        del feature['userSeq']
        features.append(feature)
    item_df = pd.DataFrame({'itemId':list(itemCounter.keys()), 'historyHit':list(itemCounter.values())})
    used_features = ['province','city','deviceType', 'logTs_gap','Hour',
                 'bm25_mean','co_bm25_mean','bm25', 'bm25Len', 
                 'gapMeanEmotion', 'groupLen','historyHitSort',
                'gongxian']
    new_features = []
    for feature in features:
        feature = pd.merge(feature, item_df, how='left', on='itemId')
        feature['historyHit'] = feature['historyHit'].fillna(0)
#         print(feature)
        #  分组特征（分组长度、情感均差、点击频次排序）
        feature['historyHitSort'] = feature.groupby('pvId').historyHit.rank(ascending=0,method='first')
        feature['historyHitSort'] = feature['historyHitSort'].astype(int)

        groupLen = feature.groupby('pvId')['logTs'].count()
        groupLen = pd.DataFrame({"groupLen": groupLen}).reset_index()
        feature = pd.merge(feature, groupLen, how='left', on='pvId')
        feature['groupLen'] = feature['groupLen'].astype(int)

        groupMeanEmotion = feature.groupby('pvId')['curMeanEmotion'].mean()
        groupMeanEmotion = pd.DataFrame({"groupMeanEmotion": groupMeanEmotion}).reset_index()
        feature = pd.merge(feature, groupMeanEmotion, how='left', on='pvId')

        feature['groupMeanEmotionGap'] = feature['curMeanEmotion'] - feature['groupMeanEmotion']
        
        ## 清洗&规范化
        feature['co_bm25_mean'] = 10*(feature['co_bm25']/(feature['co_bm25Len']+1))
        feature['co_bm25_mean'] = feature['co_bm25_mean'].astype(int)
        feature['bm25_mean'] = 10*(feature['bm25']/(feature['bm25Len']+1))
        feature['bm25_mean'] = feature['bm25_mean'].astype(int)
        feature['historyHitNor'] = 10*np.log10(feature['historyHit']+1)
        feature['historyHitNor'] = feature['historyHit'].astype(int)

        feature['historyHitSort_r'] = feature['groupLen'] - feature['historyHitSort']

        feature['bm25Sort'] = feature.groupby('pvId').bm25.rank(ascending=0,method='first')
        feature['bm25Sort'] = feature['bm25Sort'].astype(int)
        # 清洗过大的值
        feature.loc[feature['logTs_gap']>600, 'logTs_gap'] = 601 
        # 缺失值处理
        feature[used_features] = feature[used_features].fillna('-1', ) 
        if 'label' in feature.columns:
            new_features.append(feature[used_features+['label', 'pvId']])
        else:
            new_features.append(feature[used_features+['pvId']])
        
    train_all_feature = new_features[0]
    newTrain_all_feature = new_features[1]
    newTest_all_feature = new_features[2]
    # 离散特征映射
    for feat in used_features:
        
        lbe = LabelEncoder()
        temp = lbe.fit_transform(pd.concat([train_all_feature[feat], 
                                            newTrain_all_feature[feat], 
                                            newTest_all_feature[feat]]))
        train_all_feature[feat] = temp[:len(train_all_feature)]
        newTrain_all_feature[feat] = temp[len(train_all_feature):len(train_all_feature)+len(newTrain_all_feature)]
        newTest_all_feature[feat] = temp[len(train_all_feature)+len(newTrain_all_feature):]



    fixlen_feature_columns = [SparseFeat(feat, 
                              pd.concat([train_all_feature[feat], 
                                            newTrain_all_feature[feat], 
                                            newTest_all_feature[feat]]).nunique(), embedding_dim=512) 
                              for feat in used_features]
    target = ['label']


    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns 
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    TMP_DIR = './tmp/'
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)
    train = pd.concat([train_all_feature, newTrain_all_feature],axis=0)
    print('========train data========', train.head(), train.nunique(), sep='\n')
    train.to_csv('./tmp/train.csv', index=False)
    print('========test data========', newTest_all_feature.head(), newTest_all_feature.nunique(), sep='\n')
    newTest_all_feature.to_csv('./tmp/test.csv', index=False)
