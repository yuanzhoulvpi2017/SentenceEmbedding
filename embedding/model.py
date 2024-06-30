from transformers import AutoModel
import torch
from torch import nn
from pathlib import Path
import json


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        input_feature: int = None,
        output_feature: int = None,
    ) -> None:
        super(EmbeddingModel, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.device = device

        self.model = AutoModel.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)

        map_json = Path(self.model_name_or_path).joinpath("linear_map.json")
        if map_json.exists():
            with open(map_json, encoding="utf-8", mode="r") as fin:
                map_data = fin.readlines()[0]
                map_data = json.loads(map_data)
            self.input_dim, self.output_dim = (
                map_data.get("input_feature"),
                map_data.get("output_feature"),
            )
        else:
            self.input_dim, self.output_dim = (input_feature, output_feature)
        self.linear = nn.Linear(self.input_dim, self.output_dim)

        linear_weight = Path(self.model_name_or_path).joinpath("linear.pt")
        if linear_weight.exists():
            self.linear.load_state_dict(torch.load(linear_weight))

        self.linear.to(self.device)

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        # **tokenizer_output: dict,
    ) -> torch.Tensor:
        # with torch.autograd.set_detect_anomaly(True):
        #     print(tokenizer_output.keys())
        #     for i in tokenizer_output.keys():
        #         tokenizer_output[i] = tokenizer_output[i].to(self.device)

        model_output = self.model(
            input_ids=input_ids.to(self.device), token_type_ids=token_type_ids.to(self.device), attention_mask=attention_mask.to(self.device))
        embedding_value = model_output.last_hidden_state[:, 0]
        embedding_value = self.linear(embedding_value.clone())

        return embedding_value

    def save_pretrained(self, save_path: str):
        self.model.save_pretrained(save_path)
        torch.save(self.linear.state_dict(), Path(
            save_path).joinpath("linear.pt"))

        with open(
            Path(save_path).joinpath("linear_map.json"), encoding="utf-8", mode="w"
        ) as fin:
            fin.write(
                json.dumps(
                    {"input_feature": self.input_dim,
                        "output_feature": self.output_dim}
                )
            )

    def only_encode(
        self, tokenizer_output: dict,
        to_numpy: bool = True,
    ):
        with torch.no_grad():

            embedding_value = self.forward(**tokenizer_output)
            if to_numpy:
                embedding_value = embedding_value.cpu().detach().numpy()
                return embedding_value
            else:
                return embedding_value



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
