# dust3r

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=23604 launch.py --mode=eval_pose  \
    --pretrained="/home/yancheng/code/3d/LSM/checkpoints/pretrained_models/LSM-checkpoint-final.pth"   \
    --eval_dataset=rel10k --output_dir="/data/3r/output/rel10k/pred_pose/lsm" \
    --scene_graph_type complete \
    --flow_loss_weight=0.0 --temporal_smoothing_weight=0.0 \
    --model_path="/home/yancheng/code/3d/LSM/checkpoints/pretrained_models/LSM-checkpoint-final.pth" \
    --output_path="/data/3r/output/rel10k/pred_pose/lsm" \
    --file_list=""