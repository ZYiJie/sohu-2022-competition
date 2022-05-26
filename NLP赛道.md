# 2022-Sohu-Sentiment-Analysis-Solution

### 问题分析

本问题属于 **Aspect Based Sentiment Analysis(ABSA)** 任务，给定一段新闻以及若干实体，要求预测对应实体的情感特征（-2~2）

- 难点一：过长文本表示

- 难点二：情感标签极不均衡（样本整体0标签极多，2和-2标签极少）

- 难点三：实体的情感标签分布不均（不同实体在不同语境中情感特征不同，且该分布差异化大）

### 现有解决方案

1. Baseline：拆分样本，每篇文章对应一个实体，进行 text pair 分类任务
   - 缺点：分割了实体序列，无法捕获实体的情感特征间的信息；训练&推理效率低下

2. Prompt Finetune：不拆分样本，为每篇文章构造 prompt ，使用 MLM Model 预测 mask token，然后将该 token 映射回情感标签
   - 缺点：待预测字段在最前面，可能远离实体的上下文；
   - 基于谷歌的 *The Power of Scale for Parameter-Efficient Prompt Tuning* 一文可以看到，在强监督学习的任务中，精心设计了 finetune task 的模型表现可以强于 prompt-tuning的。

### 我们的解决方案

#### 数据预处理（难点一）

设计了两种content采样策略

1. **mean span content sample**

   思路：对于每一个实体采样其前后n个字符，最后对采样结果进行合并&拼接
   n = maxLen / entityCnt / 2

2. **BM25 sample**

   思路：基于BM25检索算法，把实体名的拼接作为query，把content的每个句子作为candidate，计算候选句对于query的相关性，保留相关最高的句子

#### 模型设计

1. **Sequence Labeling**
   将ABSA问题构造成序列标注问题，具体来说，可以看成已知实体位置，对实体type进行分类；本问题中的type即为情感标签

   优点：
      - 有效建模实体上下文（预测结果为实体在content中多处的情感标签）
      - 有效建模实体情感标签间的信息

2. **Mask Entity Labeling**
   - 思路：case中发现，序列标注模型有时会过于依赖实体本身的特征，而不是实体的上下文特征（难点三）。对此，我希望模型可以不那么依赖具体实体来对其情感特征进行判断，而仅去关注实体的上下文

   - 做法：在tokenizer中加入 `[et0]` ~ `[et30]` 标记，将原文中的实体替换成这些标记后再送入序列标注模型进行预测。

   - 我们认为，这样的操作可以达到类似于Bert预训练`[mask]`标记的效果；经过大量语料的训练后，`[et]` 标记所得到embedding便可以很好的表征该位置实体的情感特征。

   - 优点：降低模型对实体的特征的依赖，关注上下文特征；增强鲁棒性。
  
#### 训练策略

观察到样本中的情感特征标签分布极不均衡（难点二），我们设计了 **Weighted Loss** & **Weighted Batch** 两种策略来进行优化。

1. Weighted CrossEntropyLoss

   在CrossEntropyLoss函数的weight参数中设置label权重

   对比 `non-weight CrossEntropyLoss` `Weighted CrossEntropyLoss` `Focal Loss`后发现，加权交叉熵效果最好。

2. Weighted Batch

   在CrossEntropyLoss函数中给每个batch的loss乘上一个weight矩阵，用以衡量batch中不同样本的重要性。

   weight 计算：基于训练数据实体标签的统计特征
   - 实体词频权重：词频越小的实体越重要
   - 样本标签方差权重：样本标签分布越不均的实体越重要
   - 实体情感偏差权重：实体平均情感值越偏离0越重要
      （样本权重为其所有实体权重的均值）


#### 整体方案
   目前训练的结果：
   - roberta-large + 90%train & 10%eval + Sequence Labeling 
   - ernie-zh + 90%train & 10%eval + Mask Entity Labeling
   - 两者按 0.8 : 0.2 比例集成，线上成绩为0.69885
   - 发现预处理代码中存在bug，修改后需要重新训练，应该能有小提升


### 模型代码

修改 `CFG` 中的 `train_file` 训练集路径、`model` 预训练模型路径后：
python train_eval.py 开始 Sequence Labeling 模型训练
python train_eval_random.py 开始 Mask Entity Labeling 模型训练


