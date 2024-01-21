## 环境准备

### 创建conda环境

```
# 如果你是在 InternStudio 平台，则从本地 clone 一个已有 pytorch 2.0.1 的环境：
/root/share/install_conda_env_internlm_base.sh xtuner0.1.9
# 如果你是在其他平台：
conda create --name xtuner0.1.9 python=3.10 -y
```

看到下面提示，说明环境创建完毕

![1705811523798](assets/1705811523798.png)

### 激活conda环境

>  conda activate xtuner0.1.9

### 拉取xtuner源码

```
mkdir -p ~/xtuner019 && cd ~/xtuner019
# 拉取 0.1.9 的版本源码
git clone -b v0.1.9 https://gitee.com/Internlm/xtuner


```

## 安装XTuner

进入源码目录，并从源码安装 XTuner

> cd xtuner && pip install -e '.[all]'

![1705812072008](assets/1705812072008.png)

直接在命令行输入`xtuner`查看`xtuner`命令如何使用

![1705815559371](assets/1705815559371.png)

查看xtuner版本`xtuner version`

![1705816982505](assets/1705816982505.png)

xtuner使用简介

```
# 1. 列出xtuner所有内置配置文件:
xtuner list-cfg
# 2. 复制内置配置文件到指定目录:
xtuner copy-cfg $CONFIG $SAVE_FILE
# 3-1. GPU单卡微调大模型:
xtuner train $CONFIG
# 3-2. GPU多卡微调大模型:
NPROC_PER_NODE=$NGPUS NNODES=$NNODES NODE_RANK=$NODE_RANK PORT=$PORT ADDR=$ADDR xtuner dist_train $CONFIG $GPUS
# 4-1. 把pth模型转换成HuggingFace模型:
xtuner convert pth_to_hf $CONFIG $PATH_TO_PTH_MODEL $SAVE_PATH_TO_HF_MODEL
# 4-2. 将HuggingFace的适配器合并到预训练的LLM:
xtuner convert merge $NAME_OR_PATH_TO_LLM $NAME_OR_PATH_TO_ADAPTER $SAVE_PATH
# 4-3. 把HuggingFace的LLM拆分为最小的碎片:
xtuner convert split $NAME_OR_PATH_TO_LLM $SAVE_PATH
# 5. 使用HuggingFace的型号和适配器与LLM聊天:
xtuner chat $NAME_OR_PATH_TO_LLM --adapter $NAME_OR_PATH_TO_ADAPTER --prompt-template $PROMPT_TEMPLATE --system-template $SYSTEM_TEMPLATE
# 6-1. 预处理arxiv数据集:
xtuner preprocess arxiv $SRC_FILE $DST_FILE --start-date $START_DATE --categories $CATEGORIES
# 7-1. 日志处理数据集:
xtuner log-dataset $CONFIG
# 7-2. 验证自定义数据集的配置文件正确性：
xtuner check-custom-dataset
```

我们查看xtuner的内置配置文件

> xtuner list-cfg

![1705816922418](assets/1705816922418.png)

### 配置文件名的解释

以`internlm_chat_7b_qlora_oasst1_e3`配置文件举例

| 模型名         | internlm_chat_7b     |
| -------------- | -------------------- |
| 使用算法       | qlora                |
| 数据集         | oasst1               |
| 把数据集跑几次 | 跑3次：e3 (epoch 3 ) |

* 无 chat比如 `internlm-7b` 代表是基座(base)模型

## 模型数据准备

### 配置文件

```
# 创建一个微调 oasst1 数据集的工作路径，进入
mkdir ~/ft-oasst1 && cd ~/ft-oasst1
# 复制xtuner内置的配置文件到 ~/ft-oasst1 目录下
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 ~/ft-oasst1
```

![1705817661231](assets/1705817661231.png)

### 模型下载

方式一：我这里直接使用InternStudio平台提供的模型

> cp -r /root/share/temp/model_repos/internlm-chat-7b ~/ft-oasst1/

![1705817711350](assets/1705817711350.png)

方式二：从ModelScope下载模型文件

```
# 创建一个目录，放模型文件，防止散落一地
mkdir ~/ft-oasst1/internlm-chat-7b

# 装一下拉取模型文件要用的库
pip install modelscope

# 从 modelscope 下载下载模型文件
cd ~/ft-oasst1
apt install git git-lfs -y
git lfs install
git lfs clone https://modelscope.cn/Shanghai_AI_Laboratory/internlm-chat-7b.git -b v1.0.3
```

### 数据集准备

> cp -r /root/share/temp/datasets/openassistant-guanaco ~/ft-oasst1

![1705818486893](assets/1705818486893.png)

一切准备完毕之后，可以用`tree`命令查看`~/ft-oasst1`目录下的文件组成

![1705818611439](assets/1705818611439.png)

### 修改配置文件

将`/root/ft-oasst1/internlm_chat_7b_qlora_oasst1_e3_copy.py`文件里的模型和数据集修改为本地路径

* pretrained_model_name_or_path = './internlm-chat-7b'
* data_path = './openassistant-guanaco'

![1705818871799](assets/1705818871799.png)

配置文件的常用参数介绍

| 参数名              | 解释                                                   |
| ------------------- | ------------------------------------------------------ |
| **data_path**       | 数据路径或 HuggingFace 仓库名                          |
| max_length          | 单条数据最大 Token 数，超过则截断                      |
| pack_to_max_length  | 是否将多条短数据拼接到 max_length，提高 GPU 利用率     |
| accumulative_counts | 梯度累积，每多少次 backward 更新一次参数               |
| evaluation_inputs   | 训练过程中，会根据给定的问题进行推理，便于观测训练状态 |
| evaluation_freq     | Evaluation 的评测间隔 iter 数                          |

## 开始微调

### 训练

执行训练命令，并通过deepspeed进行训练加速

> xtuner train ~/ft-oasst1/internlm_chat_7b_qlora_oasst1_e3_copy.py --deepspeed deepspeed_zero2

此时预计要9个小时才能训练完

![1705819490345](assets/1705819490345.png)

时间太久了，我就把 `max_length` 和 `batch_size` 这两个参数调大。这里我是用的A100(1/4)*2

* max_length=2048
* batch_size=8

这个时候训练的时间就变成两个半小时了，那就静候佳音吧

![1705820875574](assets/1705820875574.png)

