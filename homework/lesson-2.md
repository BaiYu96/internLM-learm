# 书生大模型Demo体验作业

## 使用HuggingFace下载模型

使用 Hugging Face 官方提供的 `huggingface-cli` 命令行工具。安装依赖:

```shell
pip install -U huggingface_hub
```

安装完毕之后就可以直接在命令行使用`huggingface-cli`命令了

我们使用` huggingface-cli download -h`命令查看`huggingface-cli`的下载命令参数

![1704894333690](assets/1704894333690.png)

通过命令行下载`internlm-chat-20b`的`config.json`文件

```python
huggingface-cli download --resume-download --local-dir /root/temp internlm/internlm-chat-20b config.json
```

在开发机上毫不意外的报错了，因为huggingface被墙了。

![1704898913929](assets/1704898913929.png)

在我本地电脑试了就可以下载

![1704899169190](assets/1704899169190.png)

## 用InternLM-7B生成小故事

### 代码准备

首先 `clone` 代码，在 `/root` 路径下新建 `code` 目录，然后切换路径, clone 代码.

```shell
mkdir -p /root/code
cd /root/code
git clone https://gitee.com/internlm/InternLM.git
```

切换 commit 版本，与教程 commit 版本保持一致。

```shell
cd InternLM
git checkout 3028f07cb79e5b1d7342f4ad8d11efad3fd13d17
```

在`/root/code/InternLM/`目录下，新建一个`cli_demo.py`文件

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name_or_path = "/root/model/Shanghai_AI_Laboratory/internlm-chat-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("User  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break
    response, history = model.chat(tokenizer, input_text, history=messages)
    messages.append((input_text, response))
    print(f"robot >>> {response}")
```

### 代码运行

```bash
cd code/InternLM/
python cli_demo.py
```

运行代码即可体验`internlm-7b`大模型的简单交互，我用它写了一个小故事

我的prompt如下

> 人物及关系：白玉和妮妮，白玉是黑客，妮妮是猫
> 事件及冲突：人猫相伴
> 风格及题材：一场星际战争给他们的生活带来了巨大的变数，请在故事中融入了一些现代元素和对人性的思考，加入小动物的暖心片段。

```
标题：人猫相伴，一场星际战争的启示

在一个遥远的星球上，有一只叫做“白玉”的猫，他是一位著名的黑客。他有着超凡的计算机技能和敏锐的观察力，经常在网络上大显身手，引起了其他星球的关注。

一天，白玉收到了一封神秘的邀请函，邀请他去参加一个名为“星际争霸”的比赛。白玉很高兴，因为他一直梦想着能够在宇宙中展示自己的技术。他接受了邀请，并开始准备参加比赛。

在比赛期间，白玉遇到了一个名叫“妮妮”的猫咪。妮妮是一只非常聪明和机智的猫，她的技能在比赛中也相当出色。白玉和妮妮成为了好朋友，他们一起努力训练，希望能够在比赛中取得好成绩。

然而，一场突如其来的星际战争打乱了他们的计划。一场可怕的机器军团入侵了他们的星球，并威胁到了人类的生存。白玉和妮妮决定挺身而出，为保护他们的星球而战。

他们与敌人展开了一场艰苦的战斗，最终，他们成功地摧毁了机器军团。这场战争结束后，白玉和妮妮成为了英雄，他们的勇气和勇气成为了人们的楷模。

在这场战争结束后，白玉和妮妮决定不再隐藏他们的技能和才华。他们开始为星球上的人们服务，保护他们的家园，并在和平与和谐中生活。他们发现，虽然他们来自不同的星球，但他们都有着相同的目标：保护自己和他们的家园。

故事的结尾，白玉和妮妮成为了最好的朋友，并继续在星际中一起冒险，探索新的世界和挑战。

```

![1704892813889](assets/1704892813889.png)

其实生成的内容，我们不难发现，是有逻辑上的问题的，因为白玉不是一只猫，这可能是模型上还有一些参数需要调整。

## 浦语·灵笔



## Lagent 工具