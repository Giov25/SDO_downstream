#!/bin/bash
#SBATCH --job-name=MAE_normpix
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

CKPT_DIR=/home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p8_normpix
CHECKPOINT=$(ls -v "$CKPT_DIR"/checkpoint_epoch*.pth 2>/dev/null | tail -1)
[ -f "$CHECKPOINT" ] && RESUME_OPT="--resume_from $CHECKPOINT" || RESUME_OPT=""

conda run -n SDOenv python /home/gpatane/SDO_downstream/mae_project/train.py \
    --zarr_path /home/gpatane/Dataset/zarr_file_magnetogram_1024_ORDINATO.zarr \
    --checkpoint_dir /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p8_normpix/ \
    --image_size 1024 \
    --patch_size 8 \
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
    --norm_pix_loss \
    --wandb_enabled \
    $RESUME_OPT
