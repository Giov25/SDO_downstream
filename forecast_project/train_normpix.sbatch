#!/bin/bash
#SBATCH --job-name=SDO_forecast_normpix
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=forecast_train-%j.out
#SBATCH --error=forecast_train-%j.err

cd /home/gpatane/SDO_downstream/forecast_project

CHECKPOINT=/home/gpatane/SDO_downstream/forecast_project/checkpoints/frozen_normpix_tb6_dd8_best.pth
[ -f "$CHECKPOINT" ] && MODE=resume || MODE=train

conda run -n SDOenv python train.py \
    --mode $MODE \
    --freeze_encoder \
    --mae_checkpoint /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p16_normpix/best_model.pth \
    --epochs 100 \
    --lr 1e-4 \
    --batch_size 16 \
    --num_workers 16 \
    --target_size 1024 \
    --num_temporal_blocks 6 \
    --decoder_depth 8 \
    --device cuda:0 \
    --mixed_precision \
    --run_name frozen_normpix_tb6_dd8 \
    --checkpoint_path $CHECKPOINT
