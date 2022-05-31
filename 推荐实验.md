# sohu-2022-competition
 2022搜狐校园 情感分析×推荐排序 算法

---

### **情感分析**

##### 训练优化：
1. 加快收敛：warm=0.2；epoch=20
2. 使用1/10数据做训练测试优化，迭代算法
3. amp训练：速度x2 性能-1.6（roberta-base+小数据量）；但是推理时使用amp不降精度（roberta-large+全数据）

##### 数据优化

1. content采样策略：在256个字范围内，采样出关于实体词的尽可能多的相关上下文
(√√) 提取每个实体前后n个字（n = maxLen / 实体出现次数 / 2）
2. 训练集构造策略
(√) 随机从原始数据中取（服从原始数据分布）
(x) 均衡各类标签样本（对高频标签样本随机采样，对低频标签样本重复采样）
(?) 样本去重：相似度计算、句子最近邻算法
(?) 按照实体的频率以及标签方差进行采样

##### 模型优化
1. 预训练模型：
   - baseline：chinese-roberta-wwm
   - 百度 ernie-gram-zh
   - roberta-large
   - CirBERTa-Chinese-Base

2. 模型结构：
   - first-last-avg
   - prompt

---

### **推荐**

##### 数据预处理

- 通过`pvId`随机划分 train & valid，比例19:1

##### 特征构造
| 特征         | 描述                   | 操作               | 效果     |
| :----------- | :--------------------- | :----------------- | :------- |
| `suv`        | 用户ID                 |                    | baseline |
| `itemId`     | 当前文章ID             |                    | baseline |
| `city`       | 城市                   |                    | baseline |
| `hour`       | 小时数                 | identity, 24       | **有**   |
| `min`        | 分钟数                 | identity, 1440     | 无       |
| `logTs`      | 时间戳                 | hash_bucket, 1000  | 无       |
| `osType`     | 操作系统               | hash_bucket, 11    | 无       |
| `deviceType` | 设备类型               | hash_bucket, 4     | **有**   |
| `province`   | 省份                   | hash_bucket, 30    | **有**   |
| `seqLen`     | 历史记录长度           | hash_bucket, 300   | **有**   |
| `sum`        | 历史共现词频和         | hash_bucket, 2000  | **有**   |
| `nor_sum`    | 带时间间隔的共现词频和 | hash_bucket, 20000 | **有**   |
| `posCnt`     | itemId被点击频率       | hash_bucket, 3000  | 无       |


##### 结果

- 直接切分train & valid

| 使用特征 | 测试成绩 | 线上成绩 |
| :------- | :------- | :------- |
| 1-3      | 61.555   | 53.000   |
| 1-4      | 60.167   | 53.495   |

- 按`pvId`随机划分train & valid

| 使用特征          | 测试成绩 | 线上成绩 |
| :---------------- | :------- | :------- |
| 1-3               | 54.484   | 53.000   |
| +hour             | 54.689   | None     |
| +min              | 54.286   | None     |
| +                      | 54.251   | None     |
| +osType           | 54.299   | None     |
| +deviceType       | 54.792   | None     |
| +hour +deviceType | 56.166   | 54.331   |
| +seqLen +sum      | 57.340   | None     |
| +province         | 54.847   | None     |
| 以上所有有效特征  | 59.407   | 57.372   |

| 使用特征(上表最优基础上)                    | 测试成绩 | 线上成绩 |
| :------------------------------------------ | :------- | :------- |
| sum->nor_sum                                | 57.628   |          |
| +mean +std                                  | 59.312   |          |
| -seqLen                                     | 59.112   |          |
| +posCnt                                     | 57.845   | 56.328   |
| DeepFM                                      | 60.537   | 58.410   |
| IFM                                         | 61.198   |          |
| DIFM                                        | 61.586   |          |
| AutoInt                                     | 60.803   |          |
| DIFM + historItemId                         | 62.133   |          |
| DIFM + historItemId +emd=32                 | 63.019   |          |
| DIFM + historItemId +emd=128 +bachsize=2048 | 63.143   |          |

| 使用特征(上表最优基础上)      | 测试成绩 | 线上成绩 |
| :---------------------------- | :------- | :------- |
| -suv -deviceType              | 63.439   |          |
| -suv -deviceType  +entity_cnt | 63.469   | 59.228   |
| -suv   +entity_cnt            | 63.413   | 59.069   |
