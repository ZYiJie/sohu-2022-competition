#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import transformers
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
transformers.logging.set_verbosity_error()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


#=================参数设置=================
class CFG:
    apex=True
    num_workers=0
    test_file = '../data/generated_train_data.txt'    
    # test_file = '../data/generated_test_data.txt'
    model="/home/yjw/ZYJ_WorkSpace/PTMs/chinese-roberta-wwm-ext/" 
    # model="/home/yjw/ZYJ_WorkSpace/PTMs/CirBERTa-Chinese-Base/" 
    batch_size=25
    max_len=512       
    out_path = './train.pred.txt'              
    load_model_path = './roberta-base-allData-saved/'
    n_fold=2

import json
import warnings

warnings.filterwarnings('ignore')

    
#=====将官方txt数据转换成我们所需的格式==
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
    if cnt == 0:
        return content
    span = int(length/cnt/2)
    idx = content.find(entity,0)
    idxArr = []
    while idx != -1:
        idxArr.append(idx)
        idx = content.find(entity,idx+1)
    result = merge_idx(idxArr, span, content)
    return result

def split_sentence(content:str):
    for each in '，。；！？':
        content = content.replace(each, each+'##')
    return content.split('##')

def merge_sentences(expand_idxArr, sentenceArr):
    length = len(sentenceArr)
    assert length >= 1
    if length==1: 
        return [[max(0,expand_idxArr[0][0]),min(length-1, expand_idxArr[0][1])]]
    ret = []
    i, j = expand_idxArr[0]
    for x, y in expand_idxArr[1:]:
        if x <= j+1: j = y
        else:
            ret.append([max(0,i), min(j,length)])
            i, j = x, y
    ret.append([max(0,i), min(j,length-1)])
    return ret

def expand_hit_sentences(hitIdxArr, sentenceArr, SPAN=1):
    ret = []
    for each in hitIdxArr:
        ret.append([each-SPAN, each+SPAN])
    ret = merge_sentences(ret, sentenceArr)
    return ret

def sample_sentence_context(entity:str, content:str):
    cnt = content.count(entity)
    if cnt == 0: return content
    
    sentenceArr = split_sentence(content)
    
    hitIdxArr = []
    for idx, sentence in enumerate(sentenceArr):
        if entity in sentence:
            hitIdxArr.append(idx)
    
    if len(hitIdxArr)== 0: return content
    expand_hitIdxs = expand_hit_sentences(hitIdxArr, sentenceArr)
    
    ret = []
    for i, j in expand_hitIdxs:
        for ii in range(i,j+1):
            ret.append(sentenceArr[ii])
    return ''.join(ret)


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
                text = sample_context(entity, text, CFG.max_len-20)
                # text = sample_sentence_context(entity, text)
                corpus.append(text)
                ids.append(raw_id)
                entitys.append(entity)
    assert len(corpus) == len(entitys) == len(ids)
    return corpus, entitys, ids


# 读取txt文件并处理成 文本-情感关键词-情感类别 一一对应的数据
test_corpus, test_entitys, test_ids = get_test_data(input_file=CFG.test_file)


test = {'id':test_ids,'content':test_corpus,'entity':test_entitys}
test = pd.DataFrame(test)



# 载入预训练模型的分词器
tokenizer = AutoTokenizer.from_pretrained(CFG.model)
CFG.tokenizer = tokenizer


# 使用HF tokenzier 对输入（ 文本+ 情感关键词）进行编码，同一处理成CFG里定义的最大长度
def prepare_input(cfg, text, feature_text):
    inputs = cfg.tokenizer(text, feature_text, 
                           add_special_tokens=True,
                           truncation = True,
                           max_length=CFG.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs



# 定义模型结构，该结构是取预训练模型最后一层encoder输出，形状为[batch_size, sequence_length, hidden_size]，
# 在1维取平均，得到[batch_size, hidden_size]的特征向量，传递给分类层得到[batch_size, 5]的向量输出，代表每条文本在五个类别上的得分，最后使用softmax将得分规范化
# 训练过程中额外对 取平均后的输出做了5次dropout，并计算五次loss取平均，该方法可以加速模型收敛，相关思路可参考论文： https://arxiv.org/pdf/1905.09788.pdf
class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        self.fc = nn.Linear(self.config.hidden_size, 5)
        self._init_weights(self.fc)
        self.drop1=nn.Dropout(0.1)
        self.drop2=nn.Dropout(0.2)
        self.drop3=nn.Dropout(0.3)
        self.drop4=nn.Dropout(0.4)
        self.drop5=nn.Dropout(0.5)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = torch.mean(outputs[0], axis=1)
        return last_hidden_states
    
    def loss(self,logits,labels):
        loss_fnc = nn.CrossEntropyLoss()
        # loss_fnc = DiceLoss(smooth = 1, square_denominator = True, with_logits = True,  alpha = 0.01 )
        loss = loss_fnc(logits, labels)
        return loss

    def forward(self, inputs,labels=None):
        feature = self.feature(inputs)
        logits1 = self.fc(self.drop1(feature))
        logits2 = self.fc(self.drop2(feature))
        logits3 = self.fc(self.drop3(feature))
        logits4 = self.fc(self.drop4(feature))
        logits5 = self.fc(self.drop5(feature))
        output = self.fc(feature)
        output = F.softmax(output, dim=1)
        _loss=0
        if labels is not None:
            loss1 = self.loss(logits1,labels)
            loss2 = self.loss(logits2,labels)
            loss3 = self.loss(logits3,labels)
            loss4 = self.loss(logits4,labels)
            loss5 = self.loss(logits5,labels)
            _loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5
            
        return output,_loss



class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.entitys = df['entity'].values
        self.contents = df['content'].values

    def __len__(self):
        return len(self.entitys)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, 
                               self.contents[item], 
                               self.entitys[item])
        return inputs
def test_and_save_reault(device, test_loader, test_ids, result_path):
    raw_preds = []
    test_pred = []
    for fold in range(CFG.n_fold):
        model_path = os.path.join(CFG.load_model_path, f"model_fold{fold}_best.bin")
        print('='*10 + f'load model from {model_path}' + '='*10)
        current_idx = 0
        
        model = CustomModel(CFG, config_path=CFG.load_model_path+'config.pth', pretrained=True)
        model.to(device)
        model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))
        model.eval()
        tk0 = tqdm(test_loader, total=len(test_loader))
        for inputs in tk0:
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            with torch.no_grad():
                y_pred_pa_all,_ = model(inputs)
            batch_pred = (y_pred_pa_all.detach().cpu().numpy())/CFG.n_fold
            if fold == 0:
                raw_preds.append(batch_pred)
            else:
                raw_preds[current_idx] += batch_pred
                current_idx += 1
    for preds in raw_preds:
        for item in preds:
            test_pred.append(item.argmax(-1))
    assert len(test_entitys) == len(test_pred) == len(test_ids)
    result = {}
    for id, entity, pre_lable in zip(test_ids, test_entitys, test_pred):
        if pre_lable == 3:
            pre_lable = int(-1)
        elif pre_lable == 4:
            pre_lable = int(-2)
        else:
            pre_lable = int(pre_lable)
        if id in result.keys():
            result[id][entity] = pre_lable
        else:
            result[id] = {entity: pre_lable}
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("id	result")
        f.write('\n')
        for k, v in result.items():
            f.write(str(k) + '	' + json.dumps(v, ensure_ascii=False) + '\n')
    print(f"保存文件到:{result_path}")



if __name__ == '__main__':
    test_dataset = TestDataset(CFG, test)
    test_loader = DataLoader(test_dataset,
                      batch_size=256,
                      shuffle=False,
                      num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    test_and_save_reault(device, test_loader, test_ids, CFG.out_path)
    print("+++ bert valid done +++")

