#!/bin/bash
#SBATCH --job-name=Seg_Frozen_NormPix
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Encoder FROZEN con pesi MAE pretrained con norm_pix_loss=True

conda run -n SDOenv python prova_script_rapido.py \
  --mode train \
  --load_pretrained \
  --freeze_encoder \
  --model MAE_Seg_Deformer \
  --mae_checkpoint /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p16_normpix/best_model.pth \
  --model_path /home/gpatane/checkpoints/normpix/ \
  --batch_size 3 \
  --epochs 200 \
  --lr 1e-4
