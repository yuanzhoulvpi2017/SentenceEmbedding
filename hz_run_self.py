import logging
import os

from transformers import HfArgumentParser

from hz_train.arguments import HzTrainArguments, ModelDataarguments
from hz_train.data import HzEmbeddingCollator, TrainDatasetForEmbedding
from hz_train.model import EmbeddingModel, EmbeddingModel4Qwen
from hz_train.trainer import HzTrainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelDataarguments, HzTrainArguments))
    modeldata_args, training_args = parser.parse_args_into_dataclasses()
    modeldata_args: ModelDataarguments
    training_args: HzTrainArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model Data parameters %s", modeldata_args)

    if "qwen" in modeldata_args.model_name_or_path :
        model = EmbeddingModel4Qwen(model_name_or_path=modeldata_args.model_name_or_path)
    else:
        model = EmbeddingModel(model_name_or_path=modeldata_args.model_name_or_path)

    dataset = TrainDatasetForEmbedding(args=modeldata_args)

    trainer = HzTrainer(
        model=model,
        args=training_args,
        data_collator=HzEmbeddingCollator(),
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
