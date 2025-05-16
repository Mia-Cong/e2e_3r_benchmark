export CUDA_VISIBLE_DEVICES=0,1,2,3
SAVE_DIR="checkpoints/output"

torchrun --nproc_per_node=4 train.py \
    --train_dataset "5_000 @ Scannetpp(split='train', ROOT='data/scannetpp_render', aug_crop=16, resolution=(256, 256)) + 5_000 @ Scannet(split='train', ROOT='data/scannet_processed', aug_crop=16, resolution=(256, 256))" \
    --test_dataset "100 @ Scannet(split='val', ROOT='data/scannet_processed', resolution=(256, 256), seed=777)" \
    --train_criterion "GaussianLoss() + ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2) " \
    --test_criterion "GaussianLoss() + ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2) " \
    --lr 0.0001 \
    --min_lr 1e-06 \
    --warmup_epochs 10 \
    --epochs 100 \
    --batch_size 4 \
    --accum_iter 2 \
    --save_freq 1 \
    --keep_freq 5 \
    --eval_freq 1 \
    --output_dir $SAVE_DIR
