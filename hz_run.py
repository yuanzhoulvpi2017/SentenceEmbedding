import logging

from transformers import HfArgumentParser, AutoTokenizer

from embedding.arguments import HzTrainArguments, ModelDataarguments
from embedding.data import HzEmbeddingCollator, TrainDatasetForEmbedding
from embedding.model import EmbeddingModel, EmbeddingModel4Loss
from embedding.trainer import HzTrainer
# import debugpy

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
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
        level=logging.INFO if training_args.local_rank in [
            -1, 0] else logging.WARN,
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

    model = EmbeddingModel4Loss(
        model_name_or_path=modeldata_args.model_name_or_path, device='cuda', temperature=modeldata_args.temperature)
 
    tokenizer = AutoTokenizer.from_pretrained(modeldata_args.model_name_or_path)
    dataset = TrainDatasetForEmbedding(args=modeldata_args)
    collator = HzEmbeddingCollator(
        tokenizer=tokenizer, modeldatargs=modeldata_args)

    trainer = HzTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
