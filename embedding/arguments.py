from dataclasses import dataclass, field
from typing import List, Tuple
from transformers import TrainingArguments


@dataclass
class HzTrainArguments(TrainingArguments):
    # model_train_temperature: float = 0.05
    pass


@dataclass
class ModelDataarguments:
    model_name_or_path: str
    data_dir: str
    cache_dir_data: str
    query_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    temperature: float = 0.05
