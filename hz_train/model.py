from transformers import AutoTokenizer, AutoModel
from typing import List
import torch
from torch import nn 


class EmbeddingModel(nn.Module):
    def __init__(self, model_name_or_path: str, device: str = "cuda") -> None:
        super(EmbeddingModel, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.device = device
        # self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModel.from_pretrained(self.model_name_or_path)

        if self.device == "cuda":
            self.model.to("cuda")
            
    

    def forward(
        self,
        text: List[str],
        max_len: int,
    ) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]

        tokenizer_output = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        if self.device == "cuda":
            for i in tokenizer_output.keys():
                tokenizer_output[i] = tokenizer_output[i].cuda()
        model_output = self.model(**tokenizer_output)
        embedding_value = model_output.last_hidden_state[:, 0]
        return embedding_value
