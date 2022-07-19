#!/bin/sh

python -u run.py \
    --is_training 0 \
    --device cuda \
    --dataset_name action \
    --train_data_paths data/kth_action \
    --valid_data_paths data/kth_action \
    --save_dir checkpoints/kth_seemore_test \
    --model_name seemore \
    --reverse_input 1 \
    --img_width 128 \
    --img_channel 1 \
    --input_length 10 \
    --total_length 20 \
    --total_length_test 30 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.01 \
    --reverse_scheduled_sampling 1 \
    --batch_size 4 \
    --pretrained_model path_to_seemore_stage_2.ckpt
 