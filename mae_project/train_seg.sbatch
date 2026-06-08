#!/bin/bash
#SBATCH --job-name=MAE_seg
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=/home/gpatane/SDO_downstream/mae_project/logs/%x-%j.out
#SBATCH --error=/home/gpatane/SDO_downstream/mae_project/logs/%x-%j.err

MAE_CKPT=/home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p8_normpix/best_model.pth
SEG_CKPT_DIR=/home/gpatane/SDO_downstream/mae_project/checkpoints_seg

mkdir -p /home/gpatane/SDO_downstream/mae_project/logs "$SEG_CKPT_DIR"

CHECKPOINT=$(ls -v "$SEG_CKPT_DIR"/checkpoint_epoch*.pth 2>/dev/null | tail -1)
[ -f "$CHECKPOINT" ] && RESUME_OPT="--resume_from $CHECKPOINT" || RESUME_OPT=""

conda run -n SDOenv python /home/gpatane/SDO_downstream/mae_project/train_seg.py --mae_checkpoint /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p8_normpix/best_model.pth --region_loss dice_tversky --dice_tversky_weight 0.5 --wandb_enabled