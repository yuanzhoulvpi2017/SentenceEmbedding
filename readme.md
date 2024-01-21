## SentenceEmbedding
1. 一个非常**轻量级**的文本转向量训练代码。
2. 集合了[`bge`](https://github.com/FlagOpen/FlagEmbedding)项目、[`m3e`](https://github.com/wangyuxinwhy/uniem)项目的优点(非常主观的判断)，取长补短，形成的属于自己风格的代码。



### 操作流程
#### 下载模型

两种模型：
1. 一种是类似于bert的模型，从这里下载模型[https://huggingface.co/hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)
2. 一种是llama结构的模型，这里使用了[qwen-1_8_chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat)模型,点击链接下载。


#### 准备数据
1. 将数据准备成json格式，参考`bge`的数据要求
```json
{"query": str, "pos": List[str], "neg": List[str]}
```

2. 将所有的数据，可以都放在一个文件夹中

#### 开始训练

1. 如果是使用类似于bert的模型，参考`hz_run_embedding.sh`脚本，进行训练
2. 如果是使用类似于llama的模型，参考`hz_run_embedding_qwen.sh`脚本，进行训练

