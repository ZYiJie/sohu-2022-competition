
## **1. 环境配置**

- pandas>=1.0.5
- tensorflow>=1.14.0
- python3

## **2. 运行配置**

- CPU/GPU均可
- 最小内存要求
    - 特征/样本生成：3G
    - 模型训练及评估：6G

- 耗时
    - 测试环境：Apple M1 Pro 2021 内存16G
    - 特征/样本生成：70 s
    - 模型训练及评估：80 s 
    
## **3. 目录结构**

- comm.py: 数据集生成
- baseline.py: 模型训练，评估，提交
- evaluation.py: gauc 评估
- data/: 数据，特征，模型
    - rec_data/: 初赛数据集
    - feature/: 特征
    - train/：离线训练数据集
    - evaluate/：评估数据集
    - submit/：在线预估结果提交
    - model/: 模型文件

## **4. 运行流程**
- 在代码项目下新建data文件夹，下载比赛数据集，把数据集中rec_data文件夹放入data下
- 生成特征/样本：python comm.py （自动新建data目录下用于存储特征、样本和模型的各个目录）
- 训练+评估+生成提交文件：python baseline.py 
- 评分代码: python evaluation.py

## **5. 模型及特征**
- 模型：DNN
- 参数：
    - batch_size: 512
    - emb_dim: 16
    - num_epochs: 1
    - learning_rate: 0.01
- 特征：
    - dnn 特征: userid, itemId
  
## **6. 模型结果**
- valid gAUC 0.60
- test gAUC 0.53


