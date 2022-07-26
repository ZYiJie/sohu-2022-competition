# 2022搜狐校园 情感分析 × 推荐排序 算法大赛决赛

# 模型训练-提交指南

比赛要求选手在"最终提交"页面上传**压缩为tar.gz格式的Docker镜像**. 这篇指南会一步步引导选手创建自己的Docker镜像。

**请选手附带提交一份自己的运行指南(一定要包括预训练模型下载地址)和编写好的`run.sh`文件。**

## 准备工作

我们将使用Docker来封装选手的模型, 这样以便选手可以任选自己熟悉的编程语言和环境来复现自己的模型. 在 [这里](https://www.runoob.com/docker/ubuntu-docker-install.html) 查看如何在自己的系统里安装Docker.
## 目录结构

**请选手们保持`train_nlp.py，train_rec.py`文件名为主文件名，我们将在docker中运行这三个同名文件。名字不要变动！！！请将python文件中input，output，模型保存位置更改为传参的形式，我们将获取测试**

```
│  README.md
│  Dockerfile	    和主文件同级，用来构建Docker镜像
│  run.sh	    	用来运行训练过程
│  train_rec.py	    训练和测试任务二模型，按照自己需求修改
└──run_nlp.py	    训练和测试任务一数据，按照自己需求修改
```

## 运行环境
在本readme的目录下打开终端

```
cd sohu2022-competition-dockerimage-submit-train
```

## 构建镜像
```
docker build -t ${MyImageName} .
```

_请注意命令最后的"." 意味着以当前所在路径build镜像。 这里应该与Dockerfile这个文件为同一个路径。${MyImageName}是选手自己设置的镜像名称_

## 测试生成的镜像

### 任务一部分

#### 任务一训练并测试结果 train_nlp.py

假设已经在测试的输入路径`${TestDir_nlp}`（外部文件夹）中存放了文件`nlp_train.txt`,`nlp_test.txt`,`model`预训练模型文件夹，`nlp_train.txt`,`nlp_test.txt`格式同官网中任务一的训练和测试数据，可以直接使用来测试

希望在输出路径`${TestDir_nlp}`（外部文件夹）产生预测文件`section1.txt`和训练好的模型`nlp_model.pt`

**如需要其他参数，可以直接在文件内部定义**

那么用来训练和产生测试文件的命令如下：

```
docker run -it --gpus all \
  -v ${TestDir_nlp}:/data/nlp \
  ${MyImageName} \
  python train_nlp.py \
  --train_input /data/nlp/nlp_train.txt \
  --pretrained_path /data/nlp/model
  --test_input /data/nlp/nlp_test.txt \
  --output /data/nlp/section1.txt \
  --model_save_path /data/nlp/nlp_model.pt
```

- docker参数解释

  - -v 同步docker内部和外部文件夹，`:` 表示将外部文件挂载到内部文件的具体目录

  - --input，--output，为执行`main.py`时的参数
- 命令生效后，将在`${TestDir_nlp}`（外部文件夹）路径产生预测的结果文件

**我们会使用这上述脚本评价对选手上传的镜像任务一部分进行训练，和复赛线上提交进行对比，选手可以使用训练和复赛测试文件自己先进行对比。**

### 任务二 部分

#### 生成任务二测试结果  train_rec.py

假设已经在测试的输入路径`${TestDir_rec}`（外部文件夹）中存放了文件`rec_train.csv`，`rec_test.csv`，`nlp_model.pt`任务一训练好的模型用来生成情感特征。`rec_train.csv`，`rec_test.csv`格式同官网中任务一的训练和测试数据，可以直接使用来测试

并且希望在输出路径`${TestDir_rec}`（外部文件夹）产生预测文件`section2.txt`和训练好的模型`rec_model.pt`

**如需要其他参数，可以直接在文件内部定义**

Docker命令如下

```
docker run -it --gpus all \
  -v ${TestDir_rec}:/data/rec \
  ${MyImageName} \
  python train_rec.py \
  --train_input /data/rec/rec_train.csv \
  --test_input /data/rec/rec_test.csv \
  --nlp_model_path /data/rec/nlp_model.pt \
  --model_save_path /data/rec/rec_model.pt \
  --output /data/rec/section2.txt
```

## 保存镜像并压缩

```
docker save ${MyImageName}:latest | gzip > ${MyImageName}.tar.gz
```

_之后可以将保存好的 `${MyImageName}.tar.gz` 文件在"最终提交"页面上传，注意最终上传的版本号是否正确_

# Shell编写

如果有特殊参数设置，请选手编写`run.sh`文件，并且按照自己的需求仅修改以`dockr run`开头的命令，我们将直接运行`run.sh`文件。

**请选手附带提交一份自己的运行指南和编写好的`run.sh`文件。**

# 命名规范

请各只队伍按`队伍名前两个字的小写全拼音_排名_train`的方式命名镜像
例如，队伍名“狐狸Fox”，排名第11，镜像名称为`hulifox_11_train`

