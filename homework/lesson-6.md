## 书生浦语internlm2-chat-7b在 [C-Eval](https://cevalbenchmark.com/index.html#home) 基准任务上的评估

检查是否存在问题。在 `--debug` 模式下，任务将按顺序执行，并实时打印输出。

```
python run.py --datasets ceval_gen \
--hf-path /share/temp/model_repos/internlm2-chat-7b/ \
--tokenizer-path /share/temp/model_repos/internlm2-chat-7b/ \
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
--model-kwargs trust_remote_code=True device_map='auto' \
--max-seq-len 2048 \
--max-out-len 16 \
--batch-size 4 \
--num-gpus 1 \
--debug
```

![1706196483864](assets/1706196483864.png)

