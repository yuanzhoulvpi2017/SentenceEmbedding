import logging
import math
import os
import random
from dataclasses import dataclass
from itertools import chain
from typing import Any, List, Optional, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BatchEncoding, DataCollatorWithPadding, PreTrainedTokenizer

from .arguments import ModelDataarguments

logger = logging.getLogger(__name__)


def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []
    # all_file_size = []

    for root, dir, file_name in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"

            all_file_list.append(standard_path)

    return all_file_list


def load_dataset_from_path(
    data_path: Optional[str] = None, cache_dir: Optional[str] = "cache_data"
) -> Dataset:
    all_file_list = get_all_datapath(data_path)
    data_files = {"train": all_file_list}
    extension = all_file_list[0].split(".")[-1]

    logger.info("load files %d number", len(all_file_list))

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
    )["train"]
    return raw_datasets


class TrainDatasetForEmbedding(Dataset):
    def __init__(self, args: ModelDataarguments):
        self.dataset = load_dataset_from_path(data_path=args.data_dir,
                                              cache_dir=args.cache_dir_data)

        # self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> dict[str, str]:
        query = self.dataset[item]

        res = {"text": query['text'], 'text_pos': query['text_pos']}
        return res


class HzEmbeddingCollator:
    def __init__(self, tokenizer, modeldatargs: ModelDataarguments) -> None:
        self.tokenizer = tokenizer
        self.modeldataargs = modeldatargs

    def __call__(self, features) -> dict[str, List[str]]:
        query = []
        pos = []

        for i in features:
            query.append(i["text"])
            pos.append(i["text_pos"])

        query_output = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.modeldataargs.query_max_len,
            return_tensors="pt",
        )
        pos_output = self.tokenizer(
            pos,
            padding=True,
            truncation=True,
            max_length=self.modeldataargs.passage_max_len,
            return_tensors="pt",
        )

        res = {"query": query_output, "pos": pos_output}
        return res
