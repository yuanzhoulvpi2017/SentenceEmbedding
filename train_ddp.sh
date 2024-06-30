# # HOST_NODE_ADDR = 29402
# CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node 1 \
#     --master_port 25641
# CUDA_VISIBLE_DEVICES=1,2,3
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 25641 hz_run.py \
    --output_dir modeloutput_epoch3 \
    --model_name_or_path modelfile/test002 \
    --data_dir data/train_data001 \
    --learning_rate 3e-5 \
    --fp16 False \
    --num_train_epochs 3 \
    --per_device_train_batch_size 12 \
    --dataloader_drop_last True \
    --query_max_len 512 \
    --passage_max_len 512 \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --cache_dir_data data/cache_data \
    --temperature 0.05 \
    --remove_unused_columns False

# --train_group_size 8 \
# --normlized True \
