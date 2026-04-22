#!/bin/bash
#SBATCH --job-name=Recon_8patch_resume
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

conda run -n SDOenv python train.py \
    --patch_size 8 \
    --resume_from /home/gpatane/SDO_downstream/mae_project/checkpoints/best_8_patch/best_model.pth \
    --checkpoint_dir /home/gpatane/SDO_downstream/mae_project/checkpoints/best_8_patch/ \
    --wandb_enabled
    