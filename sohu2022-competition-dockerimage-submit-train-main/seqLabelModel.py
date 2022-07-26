#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8


import os
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
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import transformers
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer 
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
# from  dice_loss import  DiceLoss
# from  focalloss import  FocalLoss
from utils import sample_context_by_list, bm25_sample

transformers.logging.set_verbosity_error()
#get_ipython().run_line_magic('env', 'TOKENIZERS_PARALLELISM=true')


# ### 参数

class CFG:
    apex=True
    num_workers=0
    # train_file = '../nlp_data/train.sample.txt'    
    # test_file = '../nlp_data/test.txt'
    # model="/home/yjw/ZYJ_WorkSpace/PTMs//" 
    train_file = '../../nlp_data/final/train.mix.txt'    
    valid_file = '../../nlp_data/final/valid.mix.txt'    
    test_file = '../../nlp_data/test.txt'
    model="/home/zyj/sohu/SentimentClassification/domainAdaption/mask_roberta_saved/epoch16/" 
    output_dir = './tmp/'
    # model="/home/zyj/PTMs/ernie-gram-zh/" 
    # model="/home/zyj/PTMs/chinese-roberta-wwm-ext/" 
    # model="/home/yjw/ZYJ_WorkSpace/PTMs/ernie_1.0_skep_large_ch/" 
    scheduler='cosine'                   # ['linear', 'cosine'] # lr scheduler 类型
    batch_scheduler=True                 # 是否每个step结束后更新 lr scheduler
    num_cycles=0.5                       # 如果使用 cosine lr scheduler， 该参数决定学习率曲线的形状，0.5代表半个cosine曲线
    num_warmup_steps=0                   # 模型刚开始训练时，学习率从0到初始最大值的步数
    epochs=10
    last_epoch=-1                        # 从第 last_epoch +1 个epoch开始训练
    encoder_lr=2e-5                      # 预训练模型内部参数的学习率
    decoder_lr=2e-5                      # 自定义输出层的学习率
    batch_size=32
    max_len=512                     
    weight_decay=0.01        
    gradient_accumulation_steps=1        # 梯度累计步数，1代表每个batch更新一次
    # max_grad_norm=1000  
    seed=42 
    n_fold=4                             # 总共划分数据的份数
    trn_fold=[0]                   # 需要训练的折数，比如一共划分了4份，则可以对应训练4个模型，1代表用编号为1的折做验证，其余折做训练
    train=True

# In[6]:


import json
import warnings

warnings.filterwarnings('ignore')
#======生成log文件记录训练输出======
def get_logger(filename=os.path.join(CFG.output_dir, 'train')):
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


#=======设置全局seed保证结果可复现====
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed=42)

LOGGER = None
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
    # weights = []
    # with open('../data/entity_weight.json','r') as f:
    #     entity_weight_dic =  json.load(f)
    # assert len(entity_weight_dic) >= 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = json.loads(line.strip())
            raw_contents = tmp['content'].strip()
            if type(tmp['entity']) == list:
                entityArr = tmp['entity']
            else:
                entityArr = list(tmp['entity'].keys())
            entity_content = '、'.join(entityArr)
            text = sample_context_by_list(entityArr, raw_contents, CFG.max_len-len(entity_content))
            
            # labels = tmp['entity'].values()
            # if -2 in labels:
            #     augment = 2
            # elif 2 in labels:
            #     augment = 2
            # else:
            #     augment = 1
            # augment = 1
            # texts = bm25_sample(raw_contents, ''.join(entityArr), augment=augment,length=CFG.max_len-len(entity_content))
            # weight = cal_mean_weight(entityArr, entity_weight_dic)

            # text = text[:CFG.max_len-len(entity_content)-3]  # [cls]text[sep]entity_content[sep]
            corpus.append(text)
            entitys.append(tmp['entity']) # type:dict {entyty:label, ...}
            # weights.append(weight)
                
    train = {'content':corpus,'entity':entitys}
    train = pd.DataFrame(train)
    return train


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
# tokenizer = AutoTokenizer.from_pretrained(CFG.model)
# tokenizer = BertTokenizer.from_pretrained(CFG.model)
# CFG.tokenizer = tokenizer

def maxtch_token(token_ids:list, sentence_ids:list):
    # 得到实体的token list在句子的所有开始位置
    ret = []
    startId = token_ids[0]
    for idx, candId in enumerate(sentence_ids):
        if candId == startId and sentence_ids[idx:idx+len(token_ids)] == token_ids:
            ret.append(idx)
    assert len(ret) > 0
    return ret

def tag_entity_span(entity_startIndexs:list, entityLen:int, tag:int, targets:list):
    for index in entity_startIndexs:
        for span in range(entityLen):
            targets[index+span] = tag
    return targets

def getTrainEntityInfo(entityDic, TOKENIZER):
    # entityDic{实体名:标签}
    tagDic = {}  # {实体名:[实体ids, 实体情感标签, 实体在原字典中的次序, 实体ids长度]}
    for idx, (entity,label) in enumerate(entityDic.items()):
        entityIds = TOKENIZER(entity).input_ids[1:-1]
        tagDic[entity] = [
            entityIds,
            int(label)+3,  #-2~2 => 1~5
            idx+1 , ###从1开始
            len(entityIds)
        ]
    '''
    按实体ids长度从短到长排序，后面标注时若出现嵌套实体，会先标注短实体，然后标注长实体覆盖短实体标签
    '''
    return sorted(tagDic.items(), key=lambda x:x[1][-1])

def getTestEntityInfo(entityArr, TOKENIZER):
    # entityArr[实体名]
    tagDic = {}  # {实体名:[实体ids, 实体在原字典中的次序, 实体ids长度]}
    for idx, entity in enumerate(entityArr):
        entityIds = TOKENIZER(entity).input_ids[1:-1]
        tagDic[entity] = [
            entityIds,
            idx+1, ###从1开始
            len(entityIds)
        ]
    '''
    按实体ids长度从短到长排序，后面标注时若出现嵌套实体，会先标注短实体，然后标注长实体覆盖短实体标签
    '''
    return sorted(tagDic.items(), key=lambda x:x[1][-1])

class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.entitys = df['entity'].values        
        self.contents = df['content'].values

    def __len__(self):
        return len(self.entitys)

    def __getitem__(self, item):
        tokenizer = BertTokenizer.from_pretrained(CFG.model)
        # entity_items = list(self.entitys[item].items())
        # # random.shuffle(entity_items) # 打乱entity列表顺序 每次形成不同的标注序列
        if type(self.entitys[item]) == dict:
            entityDic = self.entitys[item]
            text = self.contents[item]
            entity_content = '、'.join(list(entityDic.keys()))
            inputs = tokenizer(text, entity_content, 
                    add_special_tokens=True,
                    truncation = True,
                    max_length=CFG.max_len,
                    padding="max_length",
                    return_offsets_mapping=False)

            labels = [0] * len(inputs.input_ids)
            labels_ids = [0] * len(inputs.input_ids)
            entityInfoItems = getTrainEntityInfo(entityDic, tokenizer)
            for entity, info in entityInfoItems:
                entity_startIndexs = maxtch_token(info[0], inputs.input_ids)
                labels = tag_entity_span(entity_startIndexs, info[-1], info[1], labels)     # 标注label序列
                labels_ids = tag_entity_span(entity_startIndexs, info[-1], info[2], labels_ids) # 标注labels_ids序列
            # print(text)
            # print(entity_content)
            # for each in zip(inputs.input_ids,labels,labels_ids):
            #     print(each)
            # exit()
            assert len(inputs['input_ids']) == len(labels)
            assert len(inputs['input_ids']) == len(labels_ids)

            # 转换为tensor
            for k, v in inputs.items():
                inputs[k] = torch.tensor(v, dtype=torch.long)
            labels_ids = torch.tensor(labels_ids, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)

            return inputs, labels, labels_ids
        else:
            entityArr = self.entitys[item]
            text = self.contents[item]
            entity_content = '、'.join(entityArr)
            inputs = tokenizer(text, entity_content, 
                    add_special_tokens=True,
                    truncation = True,
                    max_length=CFG.max_len,
                    padding="max_length",
                    return_offsets_mapping=False)


            labels_ids = [0] * len(inputs.input_ids)
            entityInfoItems = getTestEntityInfo(entityArr, tokenizer)
            for entity, info in entityInfoItems:
                entity_startIndexs = maxtch_token(info[0], inputs.input_ids)
                labels_ids = tag_entity_span(entity_startIndexs, info[-1], info[1], labels_ids) # 标注labels_ids序列

            assert len(inputs['input_ids']) == len(labels_ids)

            # 转换为tensor
            for k, v in inputs.items():
                inputs[k] = torch.tensor(v, dtype=torch.long)
            labels_ids = torch.tensor(labels_ids, dtype=torch.long)

            return inputs, labels_ids

# ### 模型定义

# In[10]:

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.3, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {} 

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
            self.model = AutoModel.from_config(self.config)
        
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
        
    def loss(self,logits,labels,weights=None):
        # loss_fnc = FocalLoss(6)
        # loss_fnc = nn.CrossEntropyLoss()

        # loss_fnc = nn.CrossEntropyLoss(ignore_index=0)

        loss_fnc = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.1, 2, 1, 0.5, 1, 3])).float() ,
                                        size_average=True).cuda()
        # loss_fnc = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.1, 2, 1, 0.5, 1, 3])).float() ,
        #                                 size_average=True,
        #                                 reduction='none').cuda()

        # loss_fnc = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.1, 2, 1, 0.5, 1, 3])).float() ,
        #                                 size_average=True,
        #                                 ignore_index=0).cuda()
        # loss_fnc = DiceLoss(smooth = 1, square_denominator = True, with_logits = True,  alpha = 0.01 )
        loss = loss_fnc(logits, labels)

        # loss = (loss * weights).mean()
        return loss

    def forward(self, inputs, labels=None, weights=None, training=True):
        feature = self.model(**inputs)[0]
        # out = self.model(**inputs)
        # # first-last-avg
        # avg = torch.cat((out.hidden_states[1].unsqueeze(2), 
        #                  out.hidden_states[-1].unsqueeze(2)), dim=2)                # [batch, seq, 2, emd_dim]
        # feature = F.avg_pool2d(avg.transpose(2, 3),kernel_size=(1,2)).squeeze(-1)   # [batch, seq, emd_dim]

        if  training:
            logits1 = self.fc(self.drop1(feature))
            logits2 = self.fc(self.drop2(feature))
            logits3 = self.fc(self.drop3(feature))
            logits4 = self.fc(self.drop4(feature))
            logits5 = self.fc(self.drop5(feature))
            _loss=0
            if labels is not None:
                loss1 = self.loss(logits1.permute(0, 2, 1), labels, weights)
                loss2 = self.loss(logits2.permute(0, 2, 1), labels, weights)
                loss3 = self.loss(logits3.permute(0, 2, 1), labels, weights)
                loss4 = self.loss(logits4.permute(0, 2, 1), labels, weights)
                loss5 = self.loss(logits5.permute(0, 2, 1), labels, weights)
                _loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
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
    for step, (inputs, labels, label_ids) in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=CFG.apex):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds
    valid_folds = get_train_data(input_file=CFG.valid_file)
    print('='*10+f' load valid data from {CFG.valid_file} length = {len(valid_folds)}'+'='*10)

    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    LOGGER.info(f'train_loader size = {len(train_loader)} valid_loader size = {len(valid_loader)}')

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, os.path.join(CFG.output_dir,'config.pth'))
    
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

    # # 对抗训练器
    # fgm = FGM(model.model)

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
        for step, (inputs, labels, label_ids) in tk0:
            total_step += 1
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            with torch.cuda.amp.autocast(enabled=CFG.apex):
                loss = model(inputs,labels,training=True)
                
            if CFG.gradient_accumulation_steps > 1:
                loss = loss / CFG.gradient_accumulation_steps
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            scaler.scale(loss).backward()
            
            # fgm.attack() # 在embedding上添加对抗扰动
            # with torch.cuda.amp.autocast(enabled=CFG.apex):
            #     loss_adv = model(inputs,labels,weights,training=True)
            # if CFG.gradient_accumulation_steps > 1:
            #     loss_adv = loss_adv / CFG.gradient_accumulation_steps
            # if torch.cuda.device_count() > 1:
            #     loss_adv = loss_adv.mean()
            # scaler.scale(loss_adv).backward()
            # fgm.restore() # 恢复embedding参数
            
            losses.update(loss.item(), batch_size)
            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            # if (step + 1) % 10 == 0:
            #     wandb.log({'train_loss':loss, 'lr':optimizer.param_groups[0]["lr"], 'flod':fold, 'epoch': epoch})

            if (step + 1) % CFG.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                if CFG.batch_scheduler:
                    scheduler.step()
            tk0.set_postfix(Epoch=epoch+1, Loss=losses.avg,lr=scheduler.get_lr()[0])
            

            EVAL_STEP = int(len(train_loader)/3) # 一个epoch evaluate 3 次
            if (step + 1) % EVAL_STEP == 0: 
                # eval
                
                avg_acc, avg_f1s = valid_fn(valid_loader, model, device)
                LOGGER.info(f'EVAL on epoch={epoch+1} step={step+1} - Score: {avg_f1s:.4f}')
                # wandb.log({'valid_f1':avg_f1s, 'flod':fold, 'epoch': epoch, 'valid_step':total_step})

                score_gap = avg_f1s - best_score
                model_saved_path = os.path.join(CFG.output_dir, f"model_best.bin")
                if best_score < avg_f1s:
                    best_score = avg_f1s
                    torch.save(model.state_dict(),model_saved_path)
                    LOGGER.info(f'Epoch {epoch+1} - Save Best Score: f1: {avg_f1s:.4f} Model')
                elif abs(score_gap) <= 0.001:
                    torch.save(model.state_dict(),model_saved_path)
                    LOGGER.info(f'Epoch {epoch+1} - Save Newwer Score: f1: {avg_f1s:.4f} Model')

        elapsed = time.time() - start_time
        avg_loss = losses.avg
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f} time: {elapsed:.0f}s')
        

    torch.cuda.empty_cache()
    gc.collect()


def saved_logits(model_saved_path):
    torch.cuda.empty_cache() 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_path = os.path.join(CFG.output_dir,'result.npy')
    test_df = get_train_data(input_file=CFG.test_file)
    print('='*10+f' load test data from {CFG.test_file} length = {len(test_df)}'+'='*10)
    

    test_dataset = TrainDataset(CFG, test_df)
    test_loader = DataLoader(test_dataset,
                    batch_size=CFG.batch_size,
                    shuffle=False,
                    num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    raw_preds = []

    model = CustomModel(CFG, config_path=os.path.join(model_saved_path,'config.pth'), pretrained=False)
    model_path = os.path.join(model_saved_path, f"model_best.bin")
    print(f'=========== load model from {model_path} ===========')
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    for idx, batch in tk0:
        try:
            inputs, labels, label_ids = batch
        except:
            inputs, label_ids = batch
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_pred_pa_all = model(inputs,training=False)
        batch_pred = y_pred_pa_all.detach().cpu().numpy()  # [batchSize, seqLen, 6]
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
            raw_preds.append(sample_logits)
    
    np.save(result_path, np.array(raw_preds))
    print(f"保存logits到:{result_path} 样本数:{len(raw_preds)}")
    torch.cuda.empty_cache() 

# ### 主程序

# In[13]:

def train_eval():
    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)

    global LOGGER 
    LOGGER = get_logger(filename=os.path.join(CFG.output_dir, 'train'))
    torch.cuda.empty_cache() 
    

    print('='*10+' TRAIN MODE '+'='*10)
    
    train_df = get_train_data(input_file=CFG.train_file)
    print('='*10+f' load train data from {CFG.train_file} length = {len(train_df)}'+'='*10)

    Fold = KFold(n_splits=CFG.n_fold, shuffle=True)
    for n, (train_index, val_index) in enumerate(Fold.split(train_df)):
    # Fold = GroupKFold(n_splits=CFG.n_fold)
    # for n, (train_index, val_index) in enumerate(Fold.split(train_df, train_df['entity'], train_df['flag'])):
        train_df.loc[val_index, 'fold'] = int(n)
    train_df['fold'] = train_df['fold'].astype(int)
    # print(train.groupby('fold').size())

    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            train_loop(train_df, fold)
    print("+++ bert train done +++")
    torch.cuda.empty_cache() 

if __name__ == '__main__':
    if CFG.train:
        train_eval()

    else:
        saved_logits(CFG.output_dir)