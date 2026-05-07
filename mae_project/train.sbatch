#!/bin/bash
#SBATCH --job-name=MAE_pretrain_ch9
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

conda run -n SDOenv python train.py \
    --zarr_path /home/gpatane/Dataset/zarr_file_magnetogram_1024_ORDINATO.zarr \
    --checkpoint_dir /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p16/ \
    --image_size 1024 \
    --patch_size 16 \
    --mask_ratio 0.25 \
    --batch_size 16 \
    --accum_steps 8 \
    --epochs 100 \
    --lr 1.5e-4 \
    --warmup_epochs 10 \
    --weight_decay 0.05 \
    --grad_clip 1.0 \
    --mixed_precision \
    --compile \
    --wandb_enabled \
    --resume_from /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p16/best_model.pth
