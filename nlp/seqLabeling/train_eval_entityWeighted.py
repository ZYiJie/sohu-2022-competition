#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8


import os

OUTPUT_DIR = './allData_roberta-large_batchWeight_saved/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

import pandas as pd
import numpy as np
from collections import Counter
import gc
import json
import time
import random
import warnings
import wandb

warnings.filterwarnings("ignore")

import scipy as sp
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from scipy.special import softmax

import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import transformers
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
# from  dice_loss import  DiceLoss
from  focalloss import  FocalLoss
transformers.logging.set_verbosity_error()
#get_ipython().run_line_magic('env', 'TOKENIZERS_PARALLELISM=true')


# ### 参数

# In[2]:

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


class CFG:
    apex=True
    num_workers=0
    # train_file = '../nlp_data/train.txt'    
    # test_file = '../nlp_data/test.txt'
    # model="/home/yjw/ZYJ_WorkSpace/PTMs/chinese-roberta-wwm-ext-large/" 
    train_file = '../../nlp_data/train.txt'    
    test_file = '../../nlp_data/test.txt'
    model="/home/zyj/PTMs/chinese-roberta-wwm-ext-large/" 
    # model="/home/yjw/ZYJ_WorkSpace/PTMs/chinese-roberta-wwm-ext-large/" 
    # model="/home/yjw/ZYJ_WorkSpace/PTMs/ernie_1.0_skep_large_ch/" 
    scheduler='cosine'                   # ['linear', 'cosine'] # lr scheduler 类型
    batch_scheduler=True                 # 是否每个step结束后更新 lr scheduler
    num_cycles=0.5                       # 如果使用 cosine lr scheduler， 该参数决定学习率曲线的形状，0.5代表半个cosine曲线
    num_warmup_steps=0                   # 模型刚开始训练时，学习率从0到初始最大值的步数
    epochs=15
    last_epoch=-1                        # 从第 last_epoch +1 个epoch开始训练
    encoder_lr=2e-5                      # 预训练模型内部参数的学习率
    decoder_lr=2e-5                      # 自定义输出层的学习率
    batch_size=10
    max_len=512                     
    weight_decay=0.01        
    gradient_accumulation_steps=1        # 梯度累计步数，1代表每个batch更新一次
    # max_grad_norm=1000  
    seed=42 
    n_fold=20                             # 总共划分数据的份数
    trn_fold=[0]                   # 需要训练的折数，比如一共划分了4份，则可以对应训练4个模型，1代表用编号为1的折做验证，其余折做训练
    train=True


# In[6]:


import json
import warnings

warnings.filterwarnings('ignore')
#======生成log文件记录训练输出======
def get_logger(filename=OUTPUT_DIR+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    
    logger.addHandler(handler1)
    if CFG.train:
        handler2 = FileHandler(filename=f"{filename}.log")
        handler2.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler2)
    return logger

LOGGER = get_logger()
#=======设置全局seed保证结果可复现====
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)


# ### content采样函数

# #### mean span content

# In[3]:


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
            
def sample_context_by_list(entitys:list, content:str, length:int):
    '''
    通过entity列表筛选content中对应每个实体位置的前后文
    '''
    cnt = 0
    for entity in entitys:
        cnt += content.count(entity)
    if cnt == 0 or len(content)<=length:
        return content
    span = int(length/cnt/2)
    idxArr = []
    for entity in entitys:
        idx = content.find(entity,0)
        while idx != -1:
            idxArr.append(idx)
            idx = content.find(entity,idx+1)
    idxArr = sorted(idxArr)
    result = merge_idx(idxArr, span, content)
    return result


# #### sentence sample

# In[4]:


class BM25_Model(object):
    def __init__(self, documents_list, k1=2, k2=1, b=0.75):
        self.documents_list = documents_list
        self.documents_number = len(documents_list)
        self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number
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
    
def split_sentence(content:str):
    for each in '。；！!?？':
        content = content.replace(each, each+'##')
    return content.split('##')

def bm25_sample(content, query, length=512):
    """
    bm25相似度打分，然后进行截断，使其<=length
    :param query:
    :param content:
    :param length:
    :return:
    """

    if len(content) <= length:
        return content
    else:
        document_list = split_sentence(content.strip())
        rest_document_list = list()
        for document in document_list:
            if len(document) != 0:
                rest_document_list.append(document)

        document_list = rest_document_list
        # print(document_list)
        model = BM25_Model(document_list)
        scores = model.get_documents_score(query)
        index_scores = []
        for index, score_i in enumerate(scores):
            index_scores.append((index, score_i))
        # print(index_scores)
        index_scores.sort(key=lambda index_score: index_score[1], reverse=True)
        new_content = ""
        save_document = [0] * len(document_list)
        content_length = 0
        for item in index_scores:
            index = item[0]
            if content_length + len(document_list[index]) > length:
                break
            else:
                content_length += len(document_list[index])
                save_document[index] = 1
        # print(index_scores)
        for i, save in enumerate(save_document):
            if save != 0:
                new_content += document_list[i]

        return new_content


# ### 数据format

# In[5]:


def encode_labels(text, entity_items):
    content_label,content_id = [0] * len(text), [0] * len(text)
    entity_label, entity_id = [], []

    for id, (entity, label) in enumerate(entity_items):
        # -2=>1 -1=>2 0=>3 1=>4 2=>5 
        label += 3  
        
        entity_label += [label] * len(entity)
        entity_id += [id+1] * len(entity)

        idx = text.find(entity,0)
        while idx >=0 :
            for i in range(idx,idx+len(entity)):
                content_id[i] = id + 1
                content_label[i] = label
            idx = text.find(entity,idx+1)

    return content_label, entity_label, content_id, entity_id

def create_mask(text, entityArr):
    content_mask = [0] * len(text)
    entity_mask = []
    for id, entity in enumerate(entityArr):
        entity_mask += [id+1] * len(entity)
        
        idx = text.find(entity,0)
        while idx >=0 :
            for i in range(idx,idx+len(entity)):
                content_mask[i] = id + 1
            idx = text.find(entity,idx+1)
    return content_mask, entity_mask

def cal_mean_weight(entitys, entity_weight_dic):
    temp = []
    for entity in entitys:
        temp.append(entity_weight_dic[entity])
    # temp = sorted(temp)
    # return temp[-1] # 最大值
    return np.mean(temp) #均值

def get_train_data(input_file):
    corpus = []
    entitys = []
    content_labels, entity_labels = [], []
    content_ids, entity_ids = [], []
    weights = []
    with open('../data/entity_weight.json','r') as f:
        entity_weight_dic =  json.load(f)
    assert len(entity_weight_dic) >= 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = json.loads(line.strip())
            raw_contents = tmp['content'].strip()
            entityArr = list(tmp['entity'].keys())
            entity_content = ''.join(entityArr)
            text = sample_context_by_list(entityArr, raw_contents, CFG.max_len-2*len(entity_content))
            # text = bm25_sample(raw_contents, ''.join(entityArr), CFG.max_len-2*len(entity_content))
            text = text[:CFG.max_len-len(entity_content)-3]  # [cls]text[sep]entity_content[sep]

            content_label, entity_label, content_id, entity_id = encode_labels(text, tmp['entity'].items())
            weight = cal_mean_weight(entityArr, entity_weight_dic)

            assert len(content_label) == len(text)
            assert len(entity_label) == len(entity_content)

            weights.append(weight)
            content_labels.append(content_label)
            entity_labels.append(entity_label)
            corpus.append(text)
            entitys.append(entity_content)
            content_ids.append(content_id)
            entity_ids.append(entity_id)

    assert len(corpus) == len(content_labels) == len(entitys) == len(entity_labels)
    train = {'content':corpus,'entity':entitys,
                'content_labels':content_labels, 'entity_labels':entity_labels, 
                'content_ids':content_ids, 'entity_ids':entity_ids, 
                'weights':weights}
    train = pd.DataFrame(train)
    return train


def get_test_data(input_file):
    ids = []
    corpus = []
    entitys = []
    content_masks, entity_masks = [], []
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tmp = json.loads(line.strip())
            raw_id = tmp['id']
            raw_contents = tmp['content'].strip()
            entityArr = tmp['entity']
            entity_content = ''.join(entityArr)
            text = sample_context_by_list(entityArr, raw_contents, CFG.max_len-2*len(entity_content))
            # text = bm25_sample(raw_contents, ''.join(entityArr), CFG.max_len-2*len(entity_content))
            text = text[:CFG.max_len-len(entity_content)-3]  # [cls]text[sep]entity_content[sep]
            
            content_mask, entity_mask = create_mask(text, entityArr)
            corpus.append(text)
            ids.append(raw_id)
            entitys.append(entity_content)
            content_masks.append(content_mask)
            entity_masks.append(entity_mask)
    assert len(corpus) == len(entitys) == len(ids) == len(entity_masks)
    test = {'id':ids,'content':corpus,'entity':entitys,'content_masks':content_masks, 'entity_masks':entity_masks}
    test = pd.DataFrame(test)
    return test


# In[6]:


# 测试
# tmp_file = '../nlp_data/train.txt'
# corpus, entitys, content_labels, entity_labels = get_train_data(tmp_file)
# corpus[0], entitys[0], str(content_labels[0]), str(entity_labels[0])


# In[7]:


# # 测试
# tmp_file = '../nlp_data/test.txt'
# corpus, entitys, ids, content_masks, entity_masks = get_test_data(tmp_file)
# corpus[0], entitys[0], ids[0], str(content_masks[0]), str(entity_masks[0])



# In[9]:


# 载入预训练模型的分词器
tokenizer = AutoTokenizer.from_pretrained(CFG.model)
CFG.tokenizer = tokenizer

# 定义模型输入的输入label
def prepare_label(cfg, label1, label2, id1, id2):
    pad_len = cfg.max_len-len(label2)-len(label1)-3
    assert pad_len>=0
    assert len(label1) == len(id1)
    label_all = [0] + label1 + [0] + label2 + [0] + [0] * pad_len
    id_all = [0] + id1 + [0] + id2 + [0] + [0] * pad_len
    return label_all, id_all

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
# 将tokenizer编码完成的输入 以及 对应的情感标签 处理成tensor，供模型训练

class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.entitys = df['entity'].values
        self.contents = df['content'].values
        self.content_labels = df['content_labels'].values
        self.entity_labels = df['entity_labels'].values
        self.content_ids = df['content_ids'].values
        self.entity_ids = df['entity_ids'].values
        self.weights = df['weights'].values

    def __len__(self):
        return len(self.entitys)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, 
                               self.contents[item], 
                               self.entitys[item])
        labels, label_ids = torch.tensor(prepare_label(
                                    self.cfg, 
                                    self.content_labels[item], self.entity_labels[item], 
                                    self.content_ids[item], self.entity_ids[item]), 
                                dtype=torch.long)
        return inputs, labels, label_ids, self.weights[item]


# ### 模型定义

# In[10]:


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
            print('='*10+f' load PTM from {cfg.model} '+'='*10)
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        self.fc = nn.Linear(self.config.hidden_size, 6)
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
        
    def loss(self,logits,labels,weights):
        # loss_fnc = FocalLoss(6)
        # loss_fnc = nn.CrossEntropyLoss()

        # loss_fnc = nn.CrossEntropyLoss(ignore_index=0)

        # loss_fnc = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.1, 2, 1, 0.5, 1, 3])).float() ,
        #                                 size_average=True).cuda()

        loss_fnc = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.1, 2, 1, 0.5, 1, 3])).float() ,
                                        size_average=True,
                                        reduction='none').cuda()
        # loss_fnc = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.1, 2, 1, 0.5, 1, 3])).float() ,
        #                                 size_average=True,
        #                                 ignore_index=0).cuda()
        # loss_fnc = DiceLoss(smooth = 1, square_denominator = True, with_logits = True,  alpha = 0.01 )
        loss = loss_fnc(logits, labels)

        loss = (loss * weights).mean()

        return loss

    def forward(self, inputs, labels=None, weights=None, training=True):
        feature = self.model(**inputs)[0]
        
        if  training:
            logits1 = self.fc(self.drop1(feature))
            logits2 = self.fc(self.drop2(feature))
            logits3 = self.fc(self.drop3(feature))
            logits4 = self.fc(self.drop4(feature))
            logits5 = self.fc(self.drop5(feature))
            _loss=0
            if labels is not None:
                loss1 = self.loss(logits1.permute(0, 2, 1),labels,weights)
                loss2 = self.loss(logits2.permute(0, 2, 1),labels,weights)
                loss3 = self.loss(logits3.permute(0, 2, 1),labels,weights)
                loss4 = self.loss(logits4.permute(0, 2, 1),labels,weights)
                loss5 = self.loss(logits5.permute(0, 2, 1),labels,weights)
                _loss = (loss1 + loss2 + loss3 + loss4 + loss5)/5
                # _loss = loss3
            return _loss
        else:
            output = self.fc(feature)
            # output = F.softmax(output, dim=1)
            return output


# ### 训练代码

# In[11]:

# 模型训练常用工具类，记录指标变化
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def valid_fn(valid_loader, model, device):
    model.eval()
    # preds = []
    valid_true = []
    valid_pred = []
    tk0=tqdm(enumerate(valid_loader),total=len(valid_loader))
    for step, (inputs, labels, label_ids, weights) in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        
        with torch.no_grad():
            y_preds = model(inputs,training=False)

        batch_pred = y_preds.detach().cpu().numpy()
        labels = np.array(labels.cpu())

        for idx, pred_logits in enumerate(batch_pred):
            label = labels[idx]
            id = label_ids[idx]

            # line = tokenizer.decode(inputs['input_ids'][idx])
            # line = line.split(' ')
            # print(len(line), line)
            # print(len(label), label)
            # print(len(id), id)

            for i in range(1, max(id)+1):
                ind = np.where(id==i)
                logits = pred_logits[ind]   # [实体字数, 6] 
                logits = logits[:,1: ]      # [实体字数, 5] 
                # print(i, f'ind={ind}', )
                # print('选中token:', [line[int(n)] for n in list(ind[0])])
                # print('pred:', int(np.mean(softmax(logits, axis=-1), axis=0).argmax(-1)))
                # print('label:', int(np.mean(label[ind]))-1)
                # print('选中pred_logits:', logits.shape, logits)
                # print('选中pred_logits softmax:', logits.shape, softmax(logits, axis=-1))
                # print('选中pred_logits softmax +argmax:', logits.shape, softmax(logits, axis=-1).argmax(-1))
                logits = softmax(logits, axis=-1)                 # 归一化概率 
                valid_true.append(int(np.mean(label[ind])) - 1)   # 注意去掉了一个0
                valid_pred.append(int(np.mean(logits, axis=0).argmax(-1)))
            # print('=================')
            # exit()
    valid_true = np.array(valid_true)
    valid_pred = np.array(valid_pred)
    avg_acc = accuracy_score(valid_true, valid_pred)
    avg_f1s = f1_score(valid_true, valid_pred, average='macro')

    LOGGER.info('Average: Accuracy: {:.3f}%, F1Score: {:.3f}'.format(100 * avg_acc, 100 * avg_f1s))
    LOGGER.info(classification_report(valid_true, valid_pred))

    return avg_acc, avg_f1s


def train_loop(folds, fold):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    LOGGER.info(f'train_loader size = {len(train_loader)} valid_loader size = {len(valid_loader)}')

    # ====================================================
    # model & optimizer
    # ====================================================
    
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, OUTPUT_DIR+'config.pth')
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay, 'initial_lr':encoder_lr},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0, 'initial_lr':encoder_lr},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0, 'initial_lr':decoder_lr}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters)
    
    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler=='linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        else :
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles, last_epoch = ((cfg.last_epoch+1)/cfg.epochs)*num_train_steps
            )
        return scheduler
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)
    
    if torch.cuda.device_count() > 1:
        print("Currently training on", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # ====================================================
    # loop
    # ====================================================
    best_score = 0
    total_step = 0
    for epoch in range(CFG.epochs-1-CFG.last_epoch):

        start_time = time.time()

        # train
        model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
        losses = AverageMeter()
        # start = end = time.time()
        global_step = 0
        grad_norm = 0
        tk0=tqdm(enumerate(train_loader),total=len(train_loader))
        for step, (inputs, labels, label_ids, weights) in tk0:
            total_step += 1
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            batch_size = labels.size(0)
            with torch.cuda.amp.autocast(enabled=CFG.apex):
                loss = model(inputs,labels,weights,training=True)
            if CFG.gradient_accumulation_steps > 1:
                loss = loss / CFG.gradient_accumulation_steps
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            if (step + 1) % 10 == 0:
                wandb.log({'train_loss':loss, 'lr':optimizer.param_groups[0]["lr"], 'flod':fold, 'epoch': epoch})

            if (step + 1) % CFG.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                if CFG.batch_scheduler:
                    scheduler.step()
            tk0.set_postfix(Epoch=epoch+1, Loss=losses.avg,lr=scheduler.get_lr()[0])

            if (step + 1) % int(len(train_loader)/5) == 0: # 一个epoch evaluate 5 次
                # eval
                
                avg_acc, avg_f1s = valid_fn(valid_loader, model, device)
                LOGGER.info(f'EVAL on epoch={epoch+1} step={step+1} - Score: {avg_f1s:.4f}')
                wandb.log({'valid_f1':avg_f1s, 'flod':fold, 'epoch': epoch, 'valid_step':total_step})

                if best_score < avg_f1s:
                    best_score = avg_f1s
                    LOGGER.info(f'Epoch {epoch+1} - Save Best Score: f1: {best_score:.4f} Model')
                    torch.save(model.state_dict(),OUTPUT_DIR+f"model_fold{fold}_best.bin")

        elapsed = time.time() - start_time
        avg_loss = losses.avg
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f} time: {elapsed:.0f}s')
        

    torch.cuda.empty_cache()
    gc.collect()


# ### 生成提交文件

# In[12]:

# 定义模型输入的输入mask
def prepare_mask(cfg, mask1, mask2):
    pad_len = cfg.max_len-len(mask1)-len(mask2)-3
    assert pad_len>=0
    mask_all = [0] + mask1 + [0] + mask2 + [0] + [0] * pad_len
    return mask_all

class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.entitys = df['entity'].values
        self.contents = df['content'].values
        self.content_masks = df['content_masks'].values
        self.entity_masks = df['entity_masks'].values

    def __len__(self):
        return len(self.entitys)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, 
                               self.contents[item], 
                               self.entitys[item])
        label_ids = torch.tensor(prepare_mask(
                                    self.cfg, 
                                    self.content_masks[item],
                                    self.entity_masks[item]), 
                                    dtype=torch.long)
        return inputs, label_ids
def test_and_save_reault(device, test_loader, test_ids, result_path):
    raw_preds = []
    test_pred = []
    for fold in CFG.trn_fold:
        current_idx = 0
        
        model = CustomModel(CFG, config_path=OUTPUT_DIR+'config.pth', pretrained=True)
        model.to(device)
        model_path = os.path.join(OUTPUT_DIR, f"model_fold{fold}_best.bin")
        print(f'=========== load model from {model_path} ===========')
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cuda')))
        model.eval()
        tk0 = tqdm(test_loader, total=len(test_loader))
        for (inputs, label_ids) in tk0:
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            with torch.no_grad():
                y_pred_pa_all = model(inputs,training=False)
            batch_pred = (y_pred_pa_all.detach().cpu().numpy())/len(CFG.trn_fold)  # [batchSize, seqLen, 6]
            for idx, pred_logits in enumerate(batch_pred):
                id = label_ids[idx]
                sample_logits = []  # [样本实体数, 5]
                for i in range(1, max(id)+1):
                    ind = np.where(id==i)
                    logits = pred_logits[ind]   # [实体字数, 6] 
                    logits = logits[:,1: ]      # [实体字数, 5] 
                    logits = softmax(logits, axis=-1)            # 归一化概率 
                    sample_logits.append(np.mean(logits, axis=0))  # [5, ] 
                    # print(i, np.mean(logits, axis=0))
                
                if fold == 0:
                    raw_preds.append(sample_logits)
                else:
                    raw_preds[current_idx] = np.add(raw_preds[current_idx], sample_logits)
                    current_idx += 1
    for item in raw_preds: 
        test_pred.append(np.array(item).argmax(-1))
    assert len(test_pred) == len(test_ids)
    result = {}
    # test_pred [样本数, 样本实体数]
    raw_entity_list = []
    with open(CFG.test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tmp = json.loads(line.strip())
            raw_entity_list.append(tmp['entity'])

    # print(len(raw_entity_list), raw_entity_list)
    for sample_idx, (id, sample_labels) in enumerate(zip(test_ids, test_pred)):
        result[id] = {}
        for idx, pre_lable in enumerate(sample_labels):
            pre_lable = int(pre_lable) - 2 ## 
            
            # print(len(sample_labels), sample_labels)
            # print(raw_entity_list[sample_idx][idx], pre_lable)
            result[id][raw_entity_list[sample_idx][idx]] = pre_lable
            # print('================')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("id	result")
        f.write('\n')
        for k, v in result.items():
            f.write(str(k) + '	' + json.dumps(v, ensure_ascii=False) + '\n')
    print(f"保存文件到:{result_path}")


# ### 主程序

# In[13]:


if __name__ == '__main__':
    if CFG.train:
        print('='*10+' TRAIN MODE '+'='*10)
        
        train_df = get_train_data(input_file=CFG.train_file)
        print('='*10+f' load train data from {CFG.train_file} length = {len(train_df)}'+'='*10)

        Fold = KFold(n_splits=CFG.n_fold, shuffle=True)
        for n, (train_index, val_index) in enumerate(Fold.split(train_df)):
            train_df.loc[val_index, 'fold'] = int(n)
        train_df['fold'] = train_df['fold'].astype(int)
        # print(train.groupby('fold').size())

        wandb.init(project='sohu-2022-SentimentClassification-seqLabeling-new')
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                train_loop(train_df, fold)
        print("+++ bert train done +++")

    else:
        # valid
        print('='*10+' VALID MODE '+'='*10)

        test_df = get_test_data(input_file=CFG.test_file)
        print('='*10+f' load test data from {CFG.test_file} length = {len(test_df)}'+'='*10)
        

        test_dataset = TestDataset(CFG, test_df)
        test_loader = DataLoader(test_dataset,
                        batch_size=256,
                        shuffle=False,
                        num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
        test_and_save_reault(device, test_loader, test_df['id'], OUTPUT_DIR+'output.txt')
        print("+++ bert valid done +++")
