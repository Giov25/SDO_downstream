#!/bin/bash
#SBATCH --job-name=SDO_forecast
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=forecast_train-%j.out
#SBATCH --error=forecast_train-%j.err

cd /home/gpatane/SDO_downstream/forecast_project

# --- Frozen encoder (start here) ---
conda run -n SDOenv python train.py \
    --mode train \
    --freeze_encoder \
    --mae_checkpoint /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p16/best_model.pth \
    --epochs 100 \
    --lr 1e-4 \
    --batch_size 16 \
    --num_workers 16 \
    --target_size 1024 \
    --num_temporal_blocks 4 \
    --decoder_depth 6 \
    --device cuda:0 \
    --mixed_precision \
    --run_name frozen_tb4_dd6 \
    --wandb_id 7r6v9mcl \
    --checkpoint_path /home/gpatane/SDO_downstream/forecast_project/checkpoints/frozen_tb4_dd6_best.pth \
    

# --- Fine-tuning (uncomment after frozen training converges) ---
# conda run -n SDOenv python train.py \
#     --mode train \
#     --no_freeze_encoder \
#     --mae_checkpoint /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p16/best_model.pth \
#     --checkpoint_path /home/gpatane/SDO_downstream/forecast_project/checkpoints/frozen_tb4_dd6_best.pth \
#     --epochs 50 \
#     --lr 5e-5 \
#     --batch_size 1 \
#     --num_workers 4 \
#     --target_size 1024 \
#     --device cuda:0 \
#     --run_name finetuned_tb4_dd6
