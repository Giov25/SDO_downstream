#!/bin/bash
#SBATCH --job-name=SDO_forecast_normpix_24h
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=logs/forecast_train-%j.out
#SBATCH --error=logs/forecast_train-%j.err

cd /home/gpatane/SDO_downstream/forecast_project

RUN_NAME=frozen_normpix_new_ultimate_p8_tb6_dd8_24h
CHECKPOINT=/home/gpatane/SDO_downstream/forecast_project/checkpoints/${RUN_NAME}_last.pth
[ -f "$CHECKPOINT" ] && MODE=resume || MODE=train

conda run -n SDOenv python train.py \
    --mode $MODE \
    --freeze_encoder \
    --mae_checkpoint /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p8_normpix/best_model.pth \
    --epochs 100 \
    --max_epochs_per_job 1 \
    --lr 1e-4 \
    --batch_size 4 \
    --num_workers 8 \
    --target_size 1024 \
    --num_temporal_blocks 6 \
    --decoder_depth 8 \
    --device cuda:0 \
    --mixed_precision \
    --run_name $RUN_NAME \
    --checkpoint_path $CHECKPOINT
