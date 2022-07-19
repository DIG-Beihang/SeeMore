#!/bin/sh

python -u run.py \
    --is_training 0 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths /data/moving-mnist-example/moving-mnist-train.npz \
    --valid_data_paths /data/moving-mnist-example/moving-mnist-test.npz \
    --save_dir checkpoints/mnist_seemore_test \
    --model_name seemore \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 1 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.1 \
    --reverse_scheduled_sampling 1 \
    --batch_size 8 \
    --test_interval 200 \
    --pretrained_model path_to_seemore_stage_2.ckpt.ckpt
