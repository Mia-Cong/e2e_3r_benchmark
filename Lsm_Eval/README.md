# Installation

```bash
conda create -n monst3r python=3.11 cmake=3.14.0
conda activate monst3r 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# Optional: you can also install additional packages to:
# - training
# - evaluation on camera pose
# - dataset preparation
pip install -r requirements_optional.txt

# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../


# install robustmvd
pip install pypng
pip install pytoml
pip install pyqt5
pip install dill
pip install easydict
#wget https://github.com/lmb-freiburg/robustmvd/archive/refs/heads/master.zip && unzip master.zip && mv robustmvd-master robustmvd && rm master.zip 
# commit id: 627dffe
cd robustmvd
python -m pip install -e . # must be developer mode for correct importation
cd ..

```

## Note, to use package ```rmvd```, you might need to fix a couple things.

1. in ```robustmvd/rmvd/utils/vis.py```, change:


```python
text_size = draw.multiline_textsize(text=text, font=font) 
```

To 

```python
try:
    text_size = draw.multiline_textsize(text=text, font=font)  # (width, height)
except:
    # Get the bounding box of the text
    bbox = draw.multiline_textbbox(xy=(0, 0), text=text, font=font)
    # Calculate width and height from the bounding box
    text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])  # (width, height)
    
```

2. In ```robustmvd/rmvd/data/paths.toml```, change dataset paths to your local directory.

# Download Checkpoints

```bash
cd src
# download the weights
cd data
bash download_ckpt.sh
cd ..
```


# Evaluation

## Sparse depth

The evaluation code is in ```./launch_mvd.py```. To evaluate Dust3r, run:
```bash
#!/bin/bash

set -e

model_name='dust3r'
model_weights="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

#datasets=('scannet')
#datasets=('tanks_and_temples')
datasets=('kitti')
# datasets=('dtu')
# datasets=('eth3d')

for data in "${datasets[@]}"; do
    mkdir -p /data/3r/output/mvd/${model_name}/${data}
    output_dir="/data/3r/output/mvd/${model_name}/${data}"
    log_file="/data/3r/output/mvd/${model_name}/${data}/mvd_${model_name}.log"
    echo "$output_dir"    
(
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=30606 launch_mvd.py \
        --eval_dataset "$data" \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --model_name "$model_name" \
        --flow_loss_weight=0.0 --temporal_smoothing_weight=0.0
) >> "$log_file" 2>&1
done
```
Results will be saved in ```/data/3r/output/mvd/${model_name}/${data}```. To evaluate Mast3r and Monst3r, change the checkpoint path and model name.

## Multi-view Recon
The evaluation code is in ```./launch_recon.py```. To evaluate Dust3r, run:

```bash
#!/bin/bash

set -e

model_name='dust3r'
model_weights="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
output_dir="/data/3r/output/recon/dust3r"
echo "$output_dir"
log_file="./recon_${model_name}.log"

# datasets=('7scenes')
# datasets=('NRGBD')
# datasets=('DTU' 'NRGBD') 
datasets=('TUM')

for data in "${datasets[@]}"; do
    echo "Processing dataset: $data"
    # (
        CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=30606 launch_recon.py \
            --weights "$model_weights" \
            --output_dir "$output_dir" \
            --model_name "$model_name" \
            --eval_dataset "$data" \
            --flow_loss_weight 0.0 
    # ) >> "$log_file" 2>&1
done
echo "Reconstruction process finished. Check logs in: $log_file"
```

Results will be saved in ```/data/3r/output/recon/dust3r```. To evaluate Mast3r and Monst3r, change the checkpoint path and model name.


## Video depth
To evaluate Dust3r on sintel, run:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29605 launch.py --mode=eval_pose  \
    --pretrained="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=sintel --output_dir="/data/3r/output/sintel/pred_pose/dust3r" \
    --scene_graph_type complete --use_gt_mask \
    --flow_loss_weight=0.0 --temporal_smoothing_weight=0.0

```
Then, run ```./depth_metric.ipynb``` to compute the depth metrics. To evaluate Mast3r and Monst3r, change the checkpoint path and model name. To evaluate other datasets, change the ```--eval_dataset```.


## Pose for Ultrra
To evalaute Dust3r, run:
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --weights './checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth' \
    --eval_data_dir /data/3r/eval_data_release \
    --save_name dust3r
```
To evaluate Mast3r and Monst3r, change the checkpoint path and model name.

## Camera Pose Estimation
To evaluate Dust3r on sintel, run:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29605 launch.py --mode=eval_pose  \
    --pretrained="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=sintel --output_dir="/data/3r/output/sintel/pred_pose/dust3r" \
    --scene_graph_type complete --use_gt_mask \
    --flow_loss_weight=0.0 --temporal_smoothing_weight=0.0
```
To evaluate Mast3r and Monst3r, change the checkpoint path and model name. To evaluate other datasets, change the ```--eval_dataset```.



# Download Datasets

## Sintel 
```bash
cd data
bash download_sintel.sh
cd ..

python datasets_preprocess/sintel_get_dynamics.py --threshold 0.1 --save_dir dynamic_label_perfect 
```

## TUM-dynamics
(Deprecated)original:
```bash
cd data
bash download_tum_dynamics.sh
cd ..

cd datasets_preprocess
python prepare_tum.py
cd ..
```

actual (directly download processed from gdrive):
```bash
cd data
gdown https://drive.google.com/uc?id=1Uz1ETAURvHZ0DNbIKIPERbvY4D8IWD1B
unzip tum.zip
rm tum.zip
cd ..

```


## Bonn
(Deprecated)original following monst3r:
```bash
cd data
bash download_bonn.sh
cd ..

cd datasets_preprocess
python prepare_bonn.py
cd ..
```

actual (directly download processed from gdrive):
```bash
gdown https://drive.google.com/uc?id=1u7_61uNcEAFbePW-r0A9bv6UQkMyGWCw
unzip bonn.zip
rm bonn.zip

```


## KITTI
(Deprecated):
```bash
cd data
bash download_kitti.sh
cd ..

cd datasets_preprocess
python prepare_kitti.py
cd ..

```

Instead, Download Processed:
```bash
cd data
gdown https://drive.google.com/uc?id=1zK3GtTE3nTW9X7PqGgetIGjx1l6q8WEh
unzip kitti.zip
rm kitti.zip
cd ..

```

## NYUv2
```bash
cd data
bash download_nyuv2.sh
cd ..

cd datasets_preprocess
python prepare_nyuv2.py
cd ..

```

## ScanNetV2
(Deprecated)original:
```bash
cd data
bash download_scannetv2.sh
cd ..

rm -rf data/scannetv2/*.sens

cd datasets_preprocess
python prepare_scannet.py
cd ..

```

Actual:
```bash
cd data
gdown https://drive.google.com/uc?id=1GZ32l9gyVAIegSnCLDUDG9wafYiCD-J3
unzip scannetv2.zip
rm scannetv2.zip
cd ..
```

For MV-recon, download subset:
```bash
cd data
gdown https://drive.google.com/uc?id=1xK7trC6SBzWnyGNH_4LndwTgKVOGb1R7
unzip scannetv2_sub.zip
rm scannetv2_sub.zip
cd ..
```

## KITTI-Odometry 


First, Download ```kitti_ordometry.zip```to ```data```: https://drive.google.com/file/d/1Ek_ipAQHxnj6J1CwJOrKqipwoecbRSox/view?usp=sharing

Then, 
```bash
cd data
gdown https://drive.google.com/uc?id=1Ek_ipAQHxnj6J1CwJOrKqipwoecbRSox
unzip kitti_ordometry.zip
mv kitti_ordometry kitti_odometry #fix typo
rm kitti_ordometry.zip
cd ..
```

## RealEstate10K
```bash
cd data
gdown https://drive.google.com/uc?id=1ZYmpOPw-dQV0-hNiqgSQ8fhXiPBMa-Rf
unzip rel10k.zip
rm rel10k.zip
cd ..
```

## ACID
```bash
cd data
gdown https://drive.google.com/uc?id=1TVEk9ww_1tuinlIwrGw8axO9C1Lhx4uq
unzip acid.zip
rm acid.zip
cd ..
```

## CO3Dv2
```bash
cd data
gdown https://drive.google.com/uc?id=1xIqklamGEM-yfPV0_dWy88LiKN1Y7ifI
mv co3d_test co3d
rm co3d.zip
cd ..

cd datasets_preprocess
python prepare_co3d.py
cd ..
```

## Syndrone
(Deprecated)
```bash
cd data
gdown https://drive.google.com/uc?id=1q2bKOptKhH04dodK-XDv1S-ofzmI6i-C
unzip syndrone.zip
rm syndrone.zip
cd ..

```

New:
```bash
cd data
gdown https://drive.google.com/uc?id=1UxIgu_NdqA1nwV96HRdP2vfYlr-30AWq
#mv syndrone syndrone_old
unzip syndrone.zip
rm syndrone.zip
```

## Ultrra

```bash
cd data
gdown https://drive.google.com/uc?id=1QJwdHzUzdty8WQ-CXWnUN-9g7G900wie
unzip eval_data_release.zip
mv eval_data_release ultrra
rm eval_data_release.zip
cd ..

cd datasets_preprocess
python prepare_ultrra.py
cd ..

```

## 7 Scenes

```bash
cd data
bash download_7scenes.sh
cd ..

cd datasets_preprocess
python prepare_7scenes.py
cd ..

```

## ADT

```bash
cd data
gdown https://drive.google.com/uc?id=1Bmtmy05VqSXVpbcafALQon7pUg4j_gRC
unzip adt.zip
rm adt.zip
cd ..
```

## NeuralRGBD

```bash
cd data
gdown https://drive.google.com/uc?id=1V_3QM_a7A5BnqEtD_irwKMmuCuvZTUZi
unzip neural_rgbd_data.zip
rm neural_rgbd_data.zip
cd ..
```

## DTU

```bash
cd data
gdown https://drive.google.com/uc?id=17qte59UNEW-kwza0FfVO8jD6hbWGVtwG
unzip dtu_test_mvsnet.zip
rm dtu_test_mvsnet.zip
cd ..
```

## TUM_rgbd for mv_recon
This is different from TUM_dynamic

```bash
cd data
gdown https://drive.google.com/uc?id=1x58bsUogoa4tSTXlvcNsmz-20PxxoYkB
unzip tum_rgbd.zip
rm tum_rgbd.zip
cd ..

```

## PointOdyssey
```bash
cd data
gdown https://drive.google.com/uc?id=1E1MrxXRq_Ea_YcORcNS-OF3RoUbtbEUH
unzip PointOdyssey.zip
rm PointOdyssey.zip
cd ..
```

## EHT3D
```bash
cd data
gdown https://drive.google.com/uc?id=1lkf-Eh1e7dsOshgaL_gqFFWGb9JFZ2sD
unzip eth3d_high_res_test_subset.zip
rm eth3d_high_res_test_subset.zip
cd ..

```

## Download Sparse Depth Datasets
```bash
cd data
mkdir -p sparse_depth
cd sparse_depth

# scannet
gdown https://drive.google.com/uc?id=16H28Z-sEgTXzsbMkKM_Ps3U7VjZC5FFC
unzip scannet.zip
rm scannet.zip

# tanks_and_temples
gdown https://drive.google.com/uc?id=1vP1lPYJrIOFLJlOM6K1icOVe7h8tpZwd
unzip tanks_and_temples.zip
rm tanks_and_temples.zip

# KITTI
gdown https://drive.google.com/uc?id=1xmfpgzGfr6WCB-7CSaP1UywC9HY-hC8H
unzip kitti.zip
rm kitti.zip

# ETH3D
gdown https://drive.google.com/uc?id=19YJYMtB7mnVbjLQEqRJHBKukLTaeuaAd
unzip eth3d.zip
rm eth3d.zip

# DTU
gdown https://drive.google.com/uc?id=1Md2E0DFtLv0Y-DeUdk4lXGYvYW5QvXO3
unzip dtu.zip
rm dtu.zip

cd ../..
```