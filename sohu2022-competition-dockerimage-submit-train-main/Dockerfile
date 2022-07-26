# 更改docker原始镜像，可以在https://hub.docker.com/中寻找适配自己的docker原始镜像,注意cuda版本
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime


ENV DEBIAN_FRONTEND=noninteractive

# 设置环境变量，使其支持中文

ENV LANG "en_US.UTF-8"

# 将代码等复制到/app目录下并将/app设置为工作目录

COPY . /app

WORKDIR /app

# 可以在本地构建镜像时指定源
# 注意修改requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
