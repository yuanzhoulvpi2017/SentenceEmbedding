from transformers import Trainer
from .model import EmbeddingModel, EmbeddingModel4Qwen
from typing import Optional
import torch
import os


class HzTrainer(Trainer):
    def compute_loss(
        self, model: EmbeddingModel | EmbeddingModel4Qwen, inputs, **kwargs
    ):
        query = inputs["query"]
        pos = inputs["pos"]
        neg = inputs["neg"]

        text_embeddings = model(query, max_len=self.args.query_max_len)

        if isinstance(model, EmbeddingModel4Qwen):
            text_pos_embeddings = model(
                pos, max_len=self.args.passage_max_len, is_query=False
            )
            text_neg_embeddings = model(
                neg, max_len=self.args.passage_max_len, is_query=False
            )
        else:
            text_pos_embeddings = model(
                pos,
                max_len=self.args.passage_max_len,
            )
            text_neg_embeddings = model(
                neg,
                max_len=self.args.passage_max_len,
            )

        sim_pos_vector = torch.cosine_similarity(
            text_embeddings, text_pos_embeddings, dim=-1
        )
        sim_pos_vector = sim_pos_vector / self.args.temperature
        sim_neg_matrix = torch.cosine_similarity(
            text_embeddings.unsqueeze(1),
            text_neg_embeddings.unsqueeze(0),
            dim=-1,
        )
        sim_neg_matrix = sim_neg_matrix / self.args.temperature
        sim_diff_matrix = sim_pos_vector.unsqueeze(1) - sim_neg_matrix
        loss = -torch.log(torch.sigmoid(sim_diff_matrix)).mean()
        return loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.model.save_pretrained(output_dir)
        self.model.tokenizer.save_pretrained(output_dir)
