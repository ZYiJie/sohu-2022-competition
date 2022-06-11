#!/usr/bin/env python
# coding: utf-8

# In[8]:


#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
from collections import Counter
import json
import time
import random
import warnings
import wandb

warnings.filterwarnings("ignore")

import scipy as sp
from tqdm import tqdm
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
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from utils import sample_context_by_list, bm25_sample
transformers.logging.set_verbosity_error()
# from  dice_loss import  DiceLoss
# from  focalloss import  FocalLoss


# ### 参数

# In[2]:


class CFG:
    train_file = '../../nlp_data/final/train.mix.txt'    
    valid_file = '../../nlp_data/final/valid.mix.txt'    
    test_file = '../../nlp_data/test.txt'
    model="/home/zyj/sohu/SentimentClassification/domainAdaption/mask_ernie_saved/epoch8/" 
    output_dir = './mlm_saved'
    # 训练参数
    device='cuda:0'
    epochs=5
    learning_rate = 2e-5
    batch_size=16
    max_len=512  
    eval_per_epoch = 3
    apex = True
    seed=42 
    # scheduler参数
    scheduler='cosine'                   # ['linear', 'cosine'] # lr scheduler 类型
    last_epoch=-1                        # 从第 last_epoch +1 个epoch开始训练
    batch_scheduler=True                 # 是否每个step结束后更新 lr scheduler
    weight_decay=0.01    
    num_warmup_steps = 0
    num_cycles=0.5                    # 如果使用 cosine lr scheduler， 该参数决定学习率曲线的形状，0.5代表半个cosine曲线
    
    # log参数
    log_step = 5
    wandb = False
    key_metrics = 'macro_f1'
    


# In[3]:


#=======设置全局seed保证结果可复现====
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)


# ### 数据预处理

# #### prepare_input

# In[4]:


MAPPER = {-2:"坏",-1:"差",0:"平",1:"好",2:"棒"}

def prepare_input(content, entitys, labels=None, TOKENIZER=None):
    inputs = TOKENIZER(content,add_special_tokens=True,
                       truncation = True,
                       max_length=CFG.max_len,
                       padding="max_length",
                       return_offsets_mapping=False)
    if labels==None:
        pass
    else: ## 形成标签inputs
        mlm_labels = []
        labels_idx = 0
        for tk in inputs.input_ids:
            if TOKENIZER.convert_ids_to_tokens(tk) == TOKENIZER.mask_token:
                label_token = MAPPER[labels[labels_idx]]
                mlm_labels.append(TOKENIZER.convert_tokens_to_ids(label_token)) # 加入映射后的prompt token id
                labels_idx += 1
            else:
                mlm_labels.append(-100) # 非mask部分
        assert labels_idx==len(labels)
        assert len(mlm_labels)==len(inputs.input_ids)
        inputs['labels'] = mlm_labels
    
    for k,v in inputs.items():
        inputs[k] = torch.tensor(v)
    return inputs


# #### format_line

# In[5]:


def format_line(line, TOKENIZER):
    tmp = json.loads(line.strip())
    raw_contents = tmp['content'].strip()
    if type(tmp['entity']) == dict:
        entityArr = list(tmp['entity'].keys())
        labels = list(tmp['entity'].values())
    elif type(tmp['entity']) == list:
        entityArr = tmp['entity']
        labels = None
    else:
        print('entity type error!')

    prompt = '在这篇新闻中'
    
    for entity in entityArr:
        prompt += f'，{entity}是{TOKENIZER.mask_token}'
    prompt += '。'
    prompt_token_len = len(TOKENIZER(prompt,add_special_tokens=False).input_ids)
    
    text = sample_context_by_list(entityArr, raw_contents, length=CFG.max_len-prompt_token_len)
    text = prompt + text
    
    inputs = prepare_input(text, entityArr, labels, TOKENIZER)
    return inputs


# #### Dataset

# In[6]:


class TrainDataset(Dataset):
    def __init__(self, input_file):
        tokenizer = AutoTokenizer.from_pretrained(CFG.model)
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.inputs = []
        for line in tqdm(lines):
            self.inputs.append(format_line(line.strip(), tokenizer))
        print(f'load data from {input_file} len={len(self.inputs)}')

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item]


# ### 主程序
def saved_logits(model_path):
    assert CFG.device.startswith('cuda') or CFG.device == 'cpu', ValueError("Invalid device.")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    target_ids = tokenizer.convert_tokens_to_ids(list(MAPPER.values()))
    def decode_mlm(arr, target_ids):
        ret = []
        for each in arr:
            ret.append(each[target_ids])
        return np.array(ret)
    test_dataset = TrainDataset(CFG.test_file)
    test_dataloader = DataLoader(test_dataset, batch_size=CFG.batch_size,)
    device = torch.device(CFG.device)
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    tk0 = tqdm(enumerate(test_dataloader),total=len(test_dataloader))
    result = []
    for step, batch in tk0:
        for k,v in batch.items():
            batch[k] = v.to(device)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        batch_logits = output.logits.detach().cpu().numpy()
        batch_input_ids = batch['input_ids'].detach().cpu().numpy()
        for idx, logits in enumerate(batch_logits):
            valid_idx = np.where(batch_input_ids[idx]==tokenizer.mask_token_id)  # 只计算情感标签部分
            if len(valid_idx[0]) > 0:
                logits = decode_mlm(logits[valid_idx], target_ids)
                result.append(softmax(logits, axis=-1))
            else:
                result.append(np.array([]))
    logits_path = os.path.join(model_path,'result.npy')
    np.save(logits_path, np.array(result))
    torch.cuda.empty_cache()  
# #### evaluate

# In[20]:


def compute_metrics(preds,labels) -> dict:
    print(classification_report(labels, preds))
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
        'macro_f1': macro_f1,
    }

def evaluate(model, valid_dataloader, device):
    model.eval()
    tk0 = tqdm(enumerate(valid_dataloader),total=len(valid_dataloader))
    loss = 0
    preds = []
    labels = []
    for step, batch in tk0:
        for k,v in batch.items():
            batch[k] = v.to(device)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            output = model(input_ids=batch['input_ids'], 
                             attention_mask=batch['attention_mask'], 
                             labels=batch['labels'])
        loss += output.loss.item()
        batch_preds = output.logits.argmax(-1).detach().cpu().numpy()
        batch_labels = batch['labels'].detach().cpu().numpy()
        valid_idx = np.where(batch_labels!=-100)  # 只计算情感标签部分
        preds.extend(batch_preds[valid_idx].ravel())
        labels.extend(batch_labels[valid_idx].ravel())
        
    metrics = compute_metrics(preds,labels)
    metrics['valid_loss'] = loss/len(valid_dataloader)
    return metrics


# #### train loop

# In[21]:


def train_eval():
    #### 加载数据和模型
    train_dataset = TrainDataset(CFG.train_file)
    valid_dataset = TrainDataset(CFG.valid_file)
    train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size,)
    valid_dataloader = DataLoader(valid_dataset, batch_size=CFG.batch_size,)

    model = AutoModelForMaskedLM.from_pretrained(CFG.model)

    assert CFG.device.startswith('cuda') or CFG.device == 'cpu', ValueError("Invalid device.")
    device = torch.device(CFG.device)
    best_score = 0
    total_step = 0
    EVAL_STEP = len(train_dataloader)//CFG.eval_per_epoch  # 每轮3次
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    if not len(train_dataloader):
        raise EOFError("Empty train_dataloader.")

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": CFG.weight_decay},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    
    num_train_steps = int(len(train_dataloader) * CFG.epochs)
    if CFG.scheduler=='cosine':
        scheduler = get_cosine_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=CFG.num_warmup_steps, 
                    num_training_steps=num_train_steps, 
                    num_cycles=CFG.num_cycles, 
#                     last_epoch = ((CFG.last_epoch+1)/CFG.epochs)*num_train_steps
                )
    else:
        scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=CFG.num_warmup_steps, num_training_steps=num_train_steps
            )
    
    for cur_epc in range(int(CFG.epochs)):
        training_loss = 0
        print("Epoch: {}".format(cur_epc))
        model.train()
        tk0 = tqdm(enumerate(train_dataloader),total=len(train_dataloader))
        for step, batch in tk0:
            total_step += 1
            for k,v in batch.items():
                batch[k] = v.to(device)
            with torch.cuda.amp.autocast(enabled=CFG.apex):
                loss = model(input_ids=batch['input_ids'], 
                             attention_mask=batch['attention_mask'], 
                             labels=batch['labels']).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if CFG.batch_scheduler:
                scheduler.step()
            training_loss += loss.item()
            tk0.set_postfix(Epoch=cur_epc, Loss=training_loss/(step+1))
            if CFG.wandb and (step + 1) % CFG.log_step == 0:
                wandb.log({'train_loss':loss, 'lr':optimizer.param_groups[0]["lr"], 'epoch': cur_epc},
                          step=total_step)
            if (step + 1) % EVAL_STEP == 0:
                metrics = evaluate(model, valid_dataloader, device)
                print(f"eval metrics = {metrics}")
                if CFG.wandb:
                    wandb.log(metrics, step=total_step)
                if metrics[CFG.key_metrics] >= best_score:
                    best_score = metrics[CFG.key_metrics]
                    model_save_path = CFG.output_dir
                    if not os.path.exists(model_save_path):
                        os.makedirs(model_save_path)
                    model.save_pretrained(model_save_path)                              
                    print(f'save at {model_save_path}')
    
    torch.cuda.empty_cache()          


