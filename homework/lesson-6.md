## 书生浦语internlm2-chat-7b在 [C-Eval](https://cevalbenchmark.com/index.html#home) 基准任务上的评估

执行下面命令

```
python run.py --datasets ceval_gen \
--hf-path /root/share/model_repos/internlm2-chat-7b/ \
--tokenizer-path /root/share/model_repos/internlm2-chat-7b/ \
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
--model-kwargs trust_remote_code=True device_map='auto' \
--max-seq-len 2048 \
--max-out-len 16 \
--batch-size 4 \
--num-gpus 1 \
--debug
```

![1706196483864](assets/1706196483864.png)

![1706199076518](assets/1706199076518.png)

![1706201101837](assets/1706201101837.png)

