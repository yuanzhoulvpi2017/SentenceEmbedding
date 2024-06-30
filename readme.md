# 使用ddp进行sentence-embedding训练

1. ddp：分布式数据并行，这个技术已经很久了，也不是啥高级的方法。
- 1.1 假设你有4张卡。在训练的时候，
- 1.2 每张卡都有一份完整的模型权重。
- 1.3 可以说，在同一个时间，4张卡同时处理数据。
- 1.4 一般在训练bert或者小模型的时候，会使用这个办法，现在训练大模型，都是使用deepspeed了。
2. 本仓库并不是为了介绍ddp这个东西，而是分享transformers的Trainer的使用经验（个人的踩坑经历）：
- 2.1 在使用huggingface的transformers的Trainer的时候，尝试了很多方法，都没办法进行ddp。
- 2.2 无意间，看到github上这个issue，尝试了一下，竟然可以了！
- 2.3 肯定是Trainer这个类的bug，虽然我也不知道到底是啥问题，就先分享一下。（后面，如果有同学做Trainer的二次开发，可以参考一下）。
- 2.4 关联的官方issue[RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.LongTensor [1, 128]] is at version 3; expected version 2 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck! ](https://github.com/huggingface/transformers/issues/23087)


## 案例介绍
1. 使用bert模型，做embedding训练。
2. 对bert输出的embedding维度做降维，从768，降低到128
3. 主要还是分享训练方法（绕开Trainer的ddp训练小bug）


## 模型下载
1. 查看`model`文件夹下的`readme.md`


## 数据构造
1. 参考`tiny_data`文件。


## 如何遇到这个bug
之前，教大家在使用Trainer做训练的时候，会教大家对Trainer的`compute_loss`做修改。比如下面的代码：
1. 这个compute_loss里面，有太多的变量了。这些变量都是参与梯度传播的。
2. 但是这么写，在进行ddp的时候，变量全部被覆盖掉了，（各个进程都有一份对应的变量，虽然对应的value不一样），导致就没办法进行ddp。（注意：这个只是我的猜测，没有做具体的验证）
```python
class HzTrainer(Trainer):
    def compute_loss(
        self,
        model: EmbeddingModel | EmbeddingModel4Loss,
        inputs,
        **kwargs,
    ):
        query = inputs["query"]
        pos = inputs["pos"]

        for i in query.keys():
            query[i] = query[i].to(model.device)

        for i in query.keys():
            pos[i] = pos[i].to(model.device)


        text_embeddings = self.embedding_model(**query)

        text_pos_embeddings = self.embedding_model(
            **pos
        )

        batch_size = text_embeddings.size(0)
        sim_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_pos_embeddings.unsqueeze(0),
            dim=-1,
        )
        sim_matrix = sim_matrix / self.temperature#0.05  # self.args.model_train_temperature
        sim_matrix_diag = sim_matrix.diag()

        sim_diff_matrix = sim_matrix_diag.unsqueeze(1) - sim_matrix
        diag_mask = torch.eye(sim_matrix.size(
            0), dtype=torch.bool, device=sim_matrix.device)
        sim_diff_matrix = sim_diff_matrix.masked_fill(diag_mask, 1e9)

        loss = -torch.log(torch.sigmoid(sim_diff_matrix)
                          ).sum() / (batch_size**2 - batch_size)
        return loss


```


## 如何修复这个bug
1. 按照上面的issue给到的建议，我们把loss放到模型的forward里面就行。
2. 其实也那怪，只要看过transformers里面的model，他们的loss都是在forward方法里面的。
3. 所以，代码最终就变成了这样。

```python

# Trainer的compute_loss变得更加简单
class HzTrainer(Trainer):
    def compute_loss(
        self,
        model: EmbeddingModel | EmbeddingModel4Loss,
        inputs,
        **kwargs,
    ):
        query = inputs["query"]
        pos = inputs["pos"]
        loss = model(query, pos)
        return loss
```


```python
# 添加了一个model，专门用来计算loss的
class EmbeddingModel4Loss(nn.Module):
    def __init__(self, model_name_or_path:str,device:str, temperature:float) -> None:
        super(EmbeddingModel4Loss,self).__init__()
        self.embedding_model = EmbeddingModel(
            model_name_or_path=model_name_or_path, device=device)
        self.temperature = temperature
        
    
    def forward(self, query, pos):
        text_embeddings = self.embedding_model(**query)

        text_pos_embeddings = self.embedding_model(
            **pos
        )

        batch_size = text_embeddings.size(0)
        sim_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_pos_embeddings.unsqueeze(0),
            dim=-1,
        )
        sim_matrix = sim_matrix / self.temperature#0.05  # self.args.model_train_temperature
        sim_matrix_diag = sim_matrix.diag()

        sim_diff_matrix = sim_matrix_diag.unsqueeze(1) - sim_matrix
        diag_mask = torch.eye(sim_matrix.size(
            0), dtype=torch.bool, device=sim_matrix.device)
        sim_diff_matrix = sim_diff_matrix.masked_fill(diag_mask, 1e9)

        loss = -torch.log(torch.sigmoid(sim_diff_matrix)
                          ).sum() / (batch_size**2 - batch_size)
        return loss
    
    def save_pretrained(self, dir_name):
        self.embedding_model.save_pretrained(dir_name)

```



