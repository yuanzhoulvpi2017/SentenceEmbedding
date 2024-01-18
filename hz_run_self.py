from hz_train.arguments import ModelDataarguments, HzTrainArguments

from hz_train.data import TrainDatasetForEmbedding, HzEmbeddingCollator

from hz_train.model import EmbeddingModel

from hz_train.trainer import HzTrainer
from transformers import HfArgumentParser
import logging
import os

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
