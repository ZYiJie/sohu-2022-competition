# _*_ coding:utf-8 _*_
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from pytorch_pretrained_bert import BertModel

warnings.filterwarnings('ignore')


class BertClassifier(nn.Module):
    def __init__(self, bert_path: str, hidden_size=768, mid_size=256):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.BatchNorm1d(mid_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mid_size, 5),
        )

    def forward(self, x):
        context = x[0]
        types = x[1]
        mask = x[2]

        _, pooled = self.bert(context, token_type_ids=types,
                              attention_mask=mask,
                              output_all_encoded_layers=False)
        context_embedding = self.dropout(pooled)

        output = self.classifier(context_embedding)
        output = F.softmax(output, dim=1)
        return output
