#!/bin/bash

set -e

model_name='dust3r'
model_weights="/home/yancheng/code/3d/LSM/checkpoints/pretrained_models/LSM-checkpoint-final.pth"
output_dir="/data/3r/output/recon/lsm"
echo "$output_dir"
log_file="./recon_${model_name}.log"
# datasets=('7scenes')
# datasets=('NRGBD')
# datasets=('DTU' 'NRGBD') 
datasets=('TUM' 'SCANNET')

for data in "${datasets[@]}"; do
    echo "Processing dataset: $data"
    # (
        CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=32606 launch_recon.py \
            --weights "$model_weights" \
            --output_dir "$output_dir" \
            --model_name "$model_name" \
            --eval_dataset "$data" \
            --flow_loss_weight=0.0 --temporal_smoothing_weight=0.0
    # ) >> "$log_file" 2>&1
done
echo "Reconstruction process finished. Check logs in: $log_file"