# _*_ coding:utf-8 _*_
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from transformers import AutoModel

warnings.filterwarnings('ignore')


class BertClassifier(nn.Module):
    def __init__(self, bert_path: str, hidden_size=768, mid_size=256):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_path)
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

        out = self.bert(context, token_type_ids=types,
                              attention_mask=mask,
                              output_hidden_states=True)
        # first-last-avg
        first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
        last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]                   
        first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
        last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
        avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
        pooled = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
        context_embedding = self.dropout(pooled)

        output = self.classifier(context_embedding)
        output = F.softmax(output, dim=1)
        return output
