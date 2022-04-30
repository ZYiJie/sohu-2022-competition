# _*_ coding:utf-8 _*_
import json
import os
import wandb
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.cuda.amp import autocast, GradScaler
# from pytorch_pretrained_bert import BertTokenizer, BertAdam
from pytorch_pretrained_bert import BertAdam
from typing import List
from sklearn.metrics import f1_score, classification_report, accuracy_score
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.load_datasets import get_train_data, get_test_data

warnings.filterwarnings('ignore')


def split_train_dataset(input_ids: List[List[int]], input_types: List[List[int]],
                        input_masks: List[List[int]], labels: List[List[int]], batch_size: int, ratio: float) -> (
DataLoader, DataLoader):
    random_order = list(range(len(input_ids)))
    np.random.shuffle(random_order)

    input_ids_train = np.array([input_ids[i] for i in random_order[:int(len(input_ids) * ratio)]])
    input_types_train = np.array([input_types[i] for i in random_order[:int(len(input_ids) * ratio)]])
    input_masks_train = np.array([input_masks[i] for i in random_order[:int(len(input_ids) * ratio)]])
    y_train = np.array([labels[i] for i in random_order[:int(len(input_ids) * ratio)]])

    input_ids_test = np.array(
        [input_ids[i] for i in random_order[int(len(input_ids) * ratio):]])
    input_types_test = np.array(
        [input_types[i] for i in random_order[int(len(input_ids) * ratio):]])
    input_masks_test = np.array(
        [input_masks[i] for i in random_order[int(len(input_ids) * ratio):]])
    y_test = np.array([labels[i] for i in random_order[int(len(input_ids) * ratio):]])

    train_data = TensorDataset(torch.LongTensor(input_ids_train),
                               torch.LongTensor(input_types_train),
                               torch.LongTensor(input_masks_train),
                               torch.LongTensor(y_train))
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True)

    valid_data = TensorDataset(torch.LongTensor(input_ids_test),
                               torch.LongTensor(input_types_test),
                               torch.LongTensor(input_masks_test),
                               torch.LongTensor(y_test))
    valid_sampler = SequentialSampler(valid_data)
    valid_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size, drop_last=True)
    return train_loader, valid_loader


def split_test_dataset(input_ids: List[List[int]], input_types: List[List[int]],
                       input_masks: List[List[int]], batch_size: int) -> (
DataLoader, DataLoader):
    input_ids_test = np.array(input_ids)
    input_types_test = np.array(input_types)
    input_masks_test = np.array(input_masks)
    assert len(input_ids_test) == len(input_masks_test)
    test_data = TensorDataset(torch.LongTensor(input_ids_test),
                              torch.LongTensor(input_types_test),
                              torch.LongTensor(input_masks_test))
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return test_loader


def train_step(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (x1, x2, x3, y) in enumerate(train_loader):
        x1_g, x2_g, x3_g, y_g = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        optimizer.zero_grad()
        # with autocast():
        y_pred = model([x1_g, x2_g, x3_g])
        loss = criterion(y_pred, y_g)
        # scaler.scale(loss).backward()

        loss.backward()
        optimizer.step()

        # scaler.step(optimizer)
        # scaler.update()
        if (batch_idx + 1) % 10 == 0:
            wandb.log({'train_loss': loss.item(), 'lr':optimizer.param_groups[0]["lr"], 'epoch': epoch})
        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx + 1) * len(x1),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def valid_step(model, device, valid_loader):
    model.eval()
    valid_loss = 0.0
    valid_true = []
    valid_pred = []
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (x1, x2, x3, y) in tqdm(enumerate(valid_loader)):
        x1_g, x2_g, x3_g, y_g = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with autocast():
            with torch.no_grad():
                y_pred_pa_all = model([x1_g, x2_g, x3_g])
        valid_loss += criterion(y_pred_pa_all, y_g)
        batch_true = y_g.cpu()
        batch_pred = y_pred_pa_all.detach().cpu().numpy()
        for item in batch_pred:
            valid_pred.append(item.argmax(0))
        for item in np.array(batch_true):
            valid_true.append(item)

    valid_loss /= len(valid_loader)
    print('Test set: Average loss: {:.4f}'.format(valid_loss))
    valid_true = np.array(valid_true)
    valid_pred = np.array(valid_pred)
    avg_acc = accuracy_score(valid_true, valid_pred)
    avg_f1s = f1_score(valid_true, valid_pred, average='macro')

    print('Average: Accuracy: {:.3f}%, F1Score: {:.3f}'.format(100 * avg_acc, 100 * avg_f1s))
    print(classification_report(valid_true, valid_pred))
    wandb.log({'Accuracy': 100 * avg_acc, 'F1Score':100 * avg_f1s, 'valid_loss':valid_loss})
    return avg_acc, avg_f1s, valid_loss


def test_and_save_reault(model, device, test_loader, test_entitys, test_ids, result_path):
    model.eval()
    test_pred = []

    for batch_idx, (x1, x2, x3) in tqdm(enumerate(test_loader)):
        x1_g, x2_g, x3_g = x1.to(device), x2.to(device), x3.to(device)
        with autocast():
            with torch.no_grad():
                y_pred_pa_all = model([x1_g, x2_g, x3_g])
        batch_pred = y_pred_pa_all.detach().cpu().numpy()
        for item in batch_pred:
            test_pred.append(item.argmax(0))
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


def main(args):
    global train_entitys, test_entitys, test_ids
    epochs = args.epochs
    padded_size = args.max_sequence_input
    batch_size = args.batch_size
    pretrain_path = args.pretrained_path
    save_state = args.save_state
    learning_rate = args.learning_rate
    output_path = args.save_model_path
    ratio = args.ratio
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)


    if not os.path.exists(output_path):
        os.makedirs(output_path)

    train_corpus, train_labels, train_entitys = get_train_data(input_file=args.train_input_path)
    test_corpus, test_entitys, test_ids = get_test_data(input_file=args.test_input_path)
    print("load train raw corpus, size:{}".format(len(train_corpus)))
    print("load test raw corpus, size:{}".format(len(test_corpus)))
    
    token_train = tokenizer(train_corpus, train_entitys, max_length=padded_size, 
                            truncation='longest_first', padding='max_length', 
                            return_tensors='np')
    train_input_ids, train_input_types, train_input_masks = token_train['input_ids'], token_train['token_type_ids'], token_train['attention_mask']
    
    token_test = tokenizer(test_corpus, test_entitys, max_length=padded_size, 
                            truncation='longest_first', padding='max_length', 
                            return_tensors='np')
    test_input_ids, test_input_types, test_input_masks = token_test['input_ids'], token_test['token_type_ids'], token_test['attention_mask']

    # Model

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint_path is not None:
        model = torch.load(args.checkpoint_path)
    else:
        from src.modeling.modeling_bert_classifier import BertClassifier
        model = BertClassifier(bert_path=pretrain_path).to(DEVICE)
    model = model.cuda(device=DEVICE)
    print("+++ model init on {} +++".format(DEVICE))

    # optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_group_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_group_parameters,
                         lr=learning_rate,
                         warmup=0.1,
                         t_total=int(len(train_input_ids) * 0.8) * epochs
                         )
    print("+++ optimizer init +++")
    
    global scaler 
    scaler = GradScaler()
    
    # wandb log
    wandb.init(project='sohu-2022-SentimentClassification') 
    
    # main train
    best_acc = 0.0
    best_epoch = 0
    train_loader, valid_loader = split_train_dataset(train_input_ids, train_input_types, train_input_masks,
                                                     train_labels,
                                                     batch_size,
                                                     ratio)
    test_loader = split_test_dataset(test_input_ids, test_input_types, test_input_masks, batch_size)
    for epoch in range(1, epochs + 1):
        train_step(model, DEVICE, train_loader, optimizer, epoch)
        acc, fis, loss = valid_step(model, DEVICE, valid_loader)
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch
        bert_classifier_path = os.path.join(output_path, 'bert_classifier_epoch{}.pth'.format(epoch))
        if save_state:
            torch.save(model.state_dict(), bert_classifier_path)
        else:
            torch.save(model, bert_classifier_path)
    print("+++ bert train done +++")

    # valid
    bert_classifier_path = os.path.join(output_path, 'bert_classifier_epoch{}.pth'.format(best_epoch))
    print("最佳轮次：", best_epoch)
    if save_state:
        model.load_state_dict(torch.load(bert_classifier_path))
    else:
        model = torch.load(bert_classifier_path)
    test_and_save_reault(model, DEVICE, test_loader, test_entitys, test_ids, args.result_path)
    print("+++ bert valid done +++")

def pred(args):
    padded_size = args.max_sequence_input
    batch_size = args.batch_size
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)



    test_corpus, test_entitys, test_ids = get_test_data(input_file=args.test_input_path)
    print("load test raw corpus, size:{}".format(len(test_corpus)))

    token_test = tokenizer(test_corpus, test_entitys, max_length=padded_size, 
                            truncation='longest_first', padding='max_length', 
                            return_tensors='np')
    test_input_ids, test_input_types, test_input_masks = token_test['input_ids'], token_test['token_type_ids'], token_test['attention_mask']
    test_loader = split_test_dataset(test_input_ids, test_input_types, test_input_masks, batch_size)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("+++ load model from %s +++" % args.checkpoint_path)
    model = torch.load(args.checkpoint_path)
    test_and_save_reault(model, DEVICE, test_loader, test_entitys, test_ids, args.result_path)

    print("+++ bert pred done +++")
