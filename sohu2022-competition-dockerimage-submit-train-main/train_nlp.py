# _*_ coding:utf-8 _*_
'''
@file: main.py
'''

from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import random



def predict(info: dict) -> (str, int):
    tmep_dict = {}
    for entity in info["entity"]:
        tmep_dict[entity] = np.random.randint(-2, 2)
    return str(info["id"]), tmep_dict

def loadNP(path):
    ret = []
    logits = np.load(path,allow_pickle=True)
    for each in logits:
        ret.append(np.array(each))
    return np.array(ret)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def infer(args):
    print('=============推理==============')
    import mlm_model
    mlm_model.CFG.test_file = args.test_input
    
    mlm_model_saved1 = os.path.join(args.model_save_path, 'mlm1')
    mlm_model.CFG.model = args.pretrained_path1
    mlm_model.CFG.output_dir = mlm_model_saved1
    mlm_model.saved_logits(mlm_model_saved1)
    
    mlm_model_saved2 = os.path.join(args.model_save_path, 'mlm2')
    mlm_model.CFG.model = args.pretrained_path2
    mlm_model.CFG.output_dir = mlm_model_saved2
    mlm_model.saved_logits(mlm_model_saved2)
    
    import seqLabelModel
    seqLabelModel.CFG.test_file = args.test_input
    
    seq_model_saved1 = os.path.join(args.model_save_path, 'seq1')
    seqLabelModel.CFG.model = args.pretrained_path1
    seqLabelModel.CFG.output_dir = seq_model_saved1
    seqLabelModel.saved_logits(seq_model_saved1)
    
    seq_model_saved2 = os.path.join(args.model_save_path, 'seq2')
    seqLabelModel.CFG.model = args.pretrained_path2
    seqLabelModel.CFG.output_dir = seq_model_saved2
    seqLabelModel.saved_logits(seq_model_saved2)
#     import maskSeqModel
#     maskSeq_model_saved = os.path.join(args.model_save_path, 'maskSeq')
#     maskSeqModel.CFG.test_file = args.test_input
#     maskSeqModel.CFG.model = args.pretrained_path
#     maskSeqModel.CFG.output_dir = maskSeq_model_saved
#     maskSeqModel.saved_logits(maskSeq_model_saved)
    print('=============输出结果==============')
    model_save_path = args.model_save_path
    test_file = args.test_input
    output_file = args.output
    model_paths = os.listdir(model_save_path)
    logitsArr = []
    for name in model_paths:
        path = os.path.join(model_save_path, name, 'result.npy')
        logits = loadNP(path)
        logitsArr.append(logits)
    logits = (0.25*logitsArr[0]+0.25*logitsArr[1]+0.25*logitsArr[3]+0.25*logitsArr[3])
    with open(output_file, 'w') as fw:
        fw.write("id	result\n")
        with open(test_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                dic = json.loads(line.strip())
                entityLen = len(dic['entity'])
                tmp_result = {}
                for j in range(entityLen):
                    label = logits[i][j].argmax()-2
                    tmp_result[dic['entity'][j]] = label
                fw.write(str(dic['id']) + '	' + json.dumps(tmp_result, ensure_ascii=False, cls=NpEncoder) + '\n')
    print(f'save result at {output_file}')
    


def train(args):
    TMP_DIR = './tmp/'
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)
    with open(args.train_input, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    random.shuffle(lines)
    train_file = './tmp/nlp_train.txt'
    valid_file = './tmp/nlp_valid.txt'
    train_len = int(len(lines)*0.95)
    with open(train_file, 'w', encoding='utf8') as f:
        for line in lines[:train_len]:
            f.write(line)
    with open(valid_file, 'w', encoding='utf8') as f:
        for line in lines[train_len:]:
            f.write(line)
    
    print('=============mlm模型训练==============')
    import mlm_model
    
#     mlm_model.CFG.epochs = 1
    mlm_model.CFG.train_file = train_file
    mlm_model.CFG.valid_file = valid_file
    
    mlm_model_saved = os.path.join(args.model_save_path, 'mlm1')
    mlm_model.CFG.model = args.pretrained_path1
    mlm_model.CFG.output_dir = mlm_model_saved
    mlm_model.train_eval()
    
    mlm_model_saved = os.path.join(args.model_save_path, 'mlm2')
    mlm_model.CFG.model = args.pretrained_path2
    mlm_model.CFG.output_dir = mlm_model_saved
    mlm_model.train_eval()
    
    print('=============seqLabel模型训练==============')
    import seqLabelModel
    
#     seqLabelModel.CFG.epochs = 1
    seqLabelModel.CFG.train_file = train_file
    seqLabelModel.CFG.valid_file = valid_file
    
    seq_model_saved = os.path.join(args.model_save_path, 'seq1')
    seqLabelModel.CFG.model = args.pretrained_path1
    seqLabelModel.CFG.output_dir = seq_model_saved
    seqLabelModel.train_eval()
    
    seq_model_saved = os.path.join(args.model_save_path, 'seq2')
    seqLabelModel.CFG.model = args.pretrained_path2
    seqLabelModel.CFG.output_dir = seq_model_saved
    seqLabelModel.train_eval()
    
#     print('=============maskSeq模型训练==============')
#     import maskSeqModel
#     maskSeq_model_saved = os.path.join(args.model_save_path, 'maskSeq')
#     maskSeqModel.CFG.epochs = 1
#     maskSeqModel.CFG.train_file = train_file
#     maskSeqModel.CFG.valid_file = valid_file
#     maskSeqModel.CFG.test_file = args.test_input
#     maskSeqModel.CFG.model = args.pretrained_path
#     maskSeqModel.CFG.output_dir = maskSeq_model_saved
#     maskSeqModel.train_eval()
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_input", type=str, required=True, help="训练输入文件")
    parser.add_argument("--pretrained_path1", type=str, required=True, help="预训练模型地址，我们将下载选手所提供的预训练模型地址,并导入")
    parser.add_argument("--pretrained_path2", type=str, required=True, help="预训练模型地址，我们将下载选手所提供的预训练模型地址,并导入")
    parser.add_argument("--test_input", type=str, required=True, help="测试输入文件")
    parser.add_argument("--output", type=str, required=True, help="输出文件")
    parser.add_argument("--model_save_path", type=str, required=True,help="训练好的模型存放地址")
    args = parser.parse_args()
    train(args)
    infer(args)
