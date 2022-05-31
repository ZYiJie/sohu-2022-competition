# 情感分析Baseline

### 目录结构

```
│  README.md
│  train_and_eval.py
├─data
│  ├─generate_data.py  从原始数据生成训练用数据
│  │
│  └─raw_data   用于存放原始数据
├─models   用于存放预训练模型
├─result   用于存放模型和结果文件
│  └─model_save
└─src
    ├─modeling
    │  └─modeling_bert_classifier.py
    │          
    ├─modeltrain
    │  └─train_bert.py
    │          
    └─utils
        └─load_datasets.py
```

### 运行环境

```
torch==1.7.1

pytorch-pretrained-bert == 0.6.2
```

### 运行

1. 从竞赛官网获取原始数据，并存放在```data/raw_data/```目录下
2. 运行`data/genereate_data.py`从原始数据中生成训练用数据，生成的数据为```data/generated_train_data.txt```和```data/generated_test_data.txt```
3. 获取Bert预训练模型([预训练模型地址](https://huggingface.co/hfl/chinese-roberta-wwm-ext))，并放存放在`models/`目录下
4. 运行`train_and_eval.py`
4. 运行结束后，结果文件存放在```result/section1.txt```，模型存放在```result/model_save/```目录下