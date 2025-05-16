#!/bin/bash

set -e

workdir='.'
model_name='ours'
ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"

#datasets=('7scenes')
#datasets=('NRGBD')
datasets=('DTU')

output_dir="${workdir}/eval_results/mv_recon_dense/${model_name}_${ckpt_name}"
echo "$output_dir"    

for data in "${datasets[@]}"; do
    accelerate launch --num_processes 8 --main_process_port 29501 eval/mv_recon/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --model_name "$model_name" \
        --eval_dataset "$data" \
        --dense
done