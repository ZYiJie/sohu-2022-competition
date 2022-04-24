# _*_ coding:utf-8 _*_

import json
import warnings

warnings.filterwarnings('ignore')

def merge_idx(idxArr, span, content):
    assert len(idxArr) >= 1
    if len(idxArr)==1:
        return content[max(0,idxArr[0]-span) : min(len(content),idxArr[0]+span)]
    i = 0
    ret = []
    while True:
        if i>=len(idxArr):break
        temp_i = i
        for j in range(i+1,len(idxArr)):
            if idxArr[j]-idxArr[temp_i] > 2*span:
                temp_i = j-1
                break
            else:
                temp_i = j
        ret.append(content[max(0,idxArr[i]-span) : min(len(content),idxArr[temp_i]+span)])    
        i = temp_i+1
    return '#'.join(ret)
            
def sample_context(entity:str, content:str, length:int):
    cnt = content.count(entity)
    span = int(length/cnt/2)
    assert cnt > 0
    idx = content.find(entity,0)
    idxArr = []
    while idx != -1:
        idxArr.append(idx)
        idx = content.find(entity,idx+1)
    result = merge_idx(idxArr, span, content)
    return result

def get_train_data(input_file):
    corpus = []
    labels = []
    entitys = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = json.loads(line.strip())
            raw_contents = tmp['content']
            raw_entitys = tmp['entity']
            label = int(tmp["label"])
            if label == -2:
                label = 4
            elif label == -1:
                label = 3
            for entity in [raw_entitys]:
                text = raw_contents.strip()
                text = sample_context(entity, text, 230)
                corpus.append(text)
                entitys.append(entity)
                labels.append(label)
    assert len(corpus) == len(labels) == len(entitys)
    return corpus, labels, entitys


def get_test_data(input_file):
    ids = []
    corpus = []
    entitys = []
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tmp = json.loads(line.strip())
            raw_id = tmp['id']
            raw_contents = tmp['content']
            raw_entitys = tmp['entity']
            for entity in [raw_entitys]:
                text = raw_contents.strip()
                text = sample_context(entity, text, 230)
                corpus.append(text)
                ids.append(raw_id)
                entitys.append(entity)
    assert len(corpus) == len(entitys) == len(ids)
    return corpus, entitys, ids
