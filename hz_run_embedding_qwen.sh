# torchrun --nnodes 1 --nproc-per-node 4

# deepspeed --include localhost:0,1,2,3
# CUDA_VISIBLE_DEVICES=1,2,3 python


deepspeed --include localhost:0,1 hz_run_self.py \
    --deepspeed ds_zero2_no_offload.json \
    --embedding_model_name qwen2 \
    --output_dir modeloutput \
    --model_name_or_path model/Qwen1.5-0.5B-Chat \
    --data_dir data/random_neg \
    --cache_dir_data cache_data \
    --learning_rate 2e-5 \
    --fp16 true \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --query_max_len 64 \
    --passage_max_len 512 \
    --remove_unused_columns False \
    --save_strategy epoch \
    --save_total_limit 3 \
    --temperature 0.05 \
    --logging_steps 5 #

    # --save_steps 5000 \

