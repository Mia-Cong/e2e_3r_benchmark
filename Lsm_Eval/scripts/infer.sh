#!/bin/bash

python demo.py \
    --file_list "demo_images/indoor/scene2/processed_000.png" "demo_images/indoor/scene2/processed_001.png" \
    --model_path "checkpoints/pretrained_models/checkpoint-final.pth" \
    --output_path "outputs/indoor/scene2" \
    --resolution "256"
