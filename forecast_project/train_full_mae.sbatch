#!/bin/bash
#SBATCH --job-name=SDO_forecast_fullmae
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=shard:1
#SBATCH --output=forecast_train-%j.out
#SBATCH --error=forecast_train-%j.err

cd /home/gpatane/SDO_downstream/forecast_project

# Fase 1 — Frozen encoder, fine-tuning solo decoder MAE + delta_t_embed
conda run -n SDOenv python train.py \
    --mode resume \
    --model_type full_mae_decoder \
    --freeze_encoder \
    --mae_checkpoint /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p16/best_model.pth \
    --epochs 100 \
    --lr 1e-4 \
    --batch_size 16 \
    --num_workers 16 \
    --target_size 1024 \
    --device cuda:0 \
    --mixed_precision \
    --run_name frozen_full_mae_dec \
    --wandb_project SDO_forecast \
    --wandb_id gjghl01x \
    --checkpoint_path /home/gpatane/SDO_downstream/forecast_project/checkpoints/frozen_full_mae_dec_best.pth \
    

# Fase 2 — Fine-tuning end-to-end (decommentare dopo convergenza fase 1)
# conda run -n SDOenv python train.py \
#     --mode train \
#     --model_type full_mae_decoder \
#     --no_freeze_encoder \
#     --mae_checkpoint /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p16/best_model.pth \
#     --checkpoint_path /home/gpatane/SDO_downstream/forecast_project/checkpoints/frozen_full_mae_dec_best.pth \
#     --epochs 50 \
#     --lr 5e-5 \
#     --batch_size 4 \
#     --num_workers 8 \
#     --target_size 1024 \
#     --device cuda:0 \
#     --mixed_precision \
#     --run_name finetuned_full_mae_dec \
#     --wandb_project SDO_forecast
