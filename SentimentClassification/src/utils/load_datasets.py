# _*_ coding:utf-8 _*_

import json
import warnings

warnings.filterwarnings('ignore')

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
                corpus.append(text)
                ids.append(raw_id)
                entitys.append(entity)
    assert len(corpus) == len(entitys) == len(ids)
    return corpus, entitys, ids
